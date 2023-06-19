// Copyright (c) 2023 Graphcore Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include <poputil/TileMapping.hpp>

namespace CustomOperators
{
  const popart::OperatorIdentifier NewIpuCopyId = {"custom.ops", "NewIpuCopy", 1};
} // namespace CustomOperators

class NewIpuCopyOp;
class NewIpuCopyOpx;

using DestIpuMap = std::map<popart::TensorId, popart::VGraphId>;
using DestTensorMap = std::map<popart::VGraphId, std::vector<popart::TensorId>>;

class NewIpuCopyOp : public popart::Op
{
public:
  NewIpuCopyOp(const popart::OperatorIdentifier &_opid,
               popart::VGraphId _sourceIpu,
               const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), sourceIpu(_sourceIpu) {}

  std::unique_ptr<Op> clone() const final
  {
    return std::make_unique<NewIpuCopyOp>(*this);
  }

  void setup() final
  {
    for (auto &idx_tensor : input->tensorMap())
    {
      auto idx = idx_tensor.first;
      outInfo(idx) = inInfo(idx);
    }
  }

  void setDestIpus(const DestIpuMap destIpus_) { destIpus = destIpus_; }

  void setDestTensors(const DestTensorMap destTensors_)
  {
    destTensors = destTensors_;
  }

  popart::VGraphId getDestIpu(const popart::TensorId &tenId) const
  {
    return destIpus.at(tenId);
  }

  void connectOutTensor(popart::OutIndex outIndex, popart::TensorId tenId,
                        popart::VGraphId destIpu)
  {
    destIpus.insert({tenId, destIpu});
    if (destTensors.find(destIpu) == destTensors.end())
    {
      destTensors.insert({destIpu, {tenId}});
    }
    else
    {
      std::vector<popart::TensorId> &tensorIds = destTensors.at(destIpu);
      tensorIds.push_back(tenId);
    }
    Op::connectOutTensor(outIndex, tenId);
  }

  void appendAttributes(popart::OpSerialiserBase &os) const override
  {
    Op::appendAttributes(os);
    std::set<int64_t> ipus;
    for (auto &destIpu : destIpus)
    {
      ipus.insert(destIpu.second);
    }
    os.appendAttribute("__sourceIpu", sourceIpu);
    os.appendAttribute("__destIpus", popart::logging::format("{}", ipus));
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override
  {
    Op::appendOutlineAttributes(os);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  popart::VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(popart::InIndex index,
                                   std::set<popart::OpId> &visited) const
  {
    return {sourceIpu, settings.tileSet};
  }

  popart::VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(popart::OutIndex index,
                                    std::set<popart::OpId> &visited) const
  {
    return {destIpus.at(outId(index)), settings.tileSet};
  }

private:
  DestIpuMap destIpus;
  DestTensorMap destTensors;
  popart::VGraphId sourceIpu;
};

namespace
{
  using popart::DataType;
  using popart::OpDefinition;

  static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

  static OpDefinition
      newIpuCopyOpDef({OpDefinition::Inputs({{"input", T}}),
                       OpDefinition::Outputs({{"output", T}}),
                       OpDefinition::Attributes({{"__sourceIpu", {"*"}}})});

  static popart::OpCreator<NewIpuCopyOp> newIpuCopyOpCreator(
      popart::OpDefinitions({{CustomOperators::NewIpuCopyId, newIpuCopyOpDef}}),
      [](const popart::OpCreatorInfo &info)
      {
        float __sourceIpu = info.attributes.getAttribute<popart::Attributes::Int>(
            "__sourceIpu", -1);
        return std::make_unique<NewIpuCopyOp>(info.opid, __sourceIpu,
                                              info.settings);
      },
      true);
} // namespace

class NewIpuCopyOpx : public popart::popx::Opx
{
public:
  NewIpuCopyOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex)
  {
    verifyOp<NewIpuCopyOp>(op, {CustomOperators::NewIpuCopyId});
  }

  void grow(poplar::program::Sequence &prog) const final
  {
    NewIpuCopyOp &op = getOp<NewIpuCopyOp>();

    for (auto &idx_tensor : op.output->tensorMap())
    {
      auto idx = idx_tensor.first;
      // Need to get the non virtual graph, so cannot use Opx::graph()
      auto t = poputil::copyToIpu(dv_p->lowering().graph(),
                                  getInTensor(idx),
                                  prog,
                                  static_cast<int>(op.getDestIpu(op.output->tensor(idx)->id)),
                                  debugContext("newIpuCopy"));
      setOutTensor(idx, t);
    }
  }
};

static popart::popx::OpxCreator<NewIpuCopyOpx>
    NewIpuCopyOpxCreator({CustomOperators::NewIpuCopyId});
