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


#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorinfo.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popnn/Loss.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <poputil/TileMapping.hpp>
#include <iosfwd>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// prog.add(poplar::program::PrintTensor("aa", aa));
// std::cout << "Value: a = " << a << std::endl;

using namespace popops::expr;
namespace pe = popops::expr;

namespace CustomOperators {
const popart::OperatorIdentifier BeamSearchId = {"ai.graphcore",
                                                 "BeamSearch",
                                                 1};
} // namespace CustomOperators

class BeamSearchOp : public popart::Op {
public:
  BeamSearchOp(const popart::OperatorIdentifier &_opid,
               const int beamSize,
               const popart::Op::Settings &settings_);

  std::unique_ptr<popart::Op> clone() const override;
  void setup() final;

  static popart::InIndex logitsInIndex() { return 0; }
  static popart::InIndex hypContainerInIndex() { return 1; }
  static popart::InIndex probContainerInIndex() { return 2; }
  static popart::InIndex stepInIndex() { return 3; }

  static popart::OutIndex nextIdOutIndex() { return 0; }
  static popart::OutIndex globalBestIdOutIndex() { return 1; }
  static popart::OutIndex hypContainerOutIndex() { return 2; }
  static popart::OutIndex probContainerOutIndex() { return 3; }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("beam_size", getBeamSize());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("beam_size", getBeamSize());
  }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  int getBeamSize() const { return beamSize; }

private:
  int beamSize;
};

BeamSearchOp::BeamSearchOp(const popart::OperatorIdentifier &_opid,
                           const int beamSize,
                           const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_), beamSize(beamSize) {}

std::unique_ptr<popart::Op> BeamSearchOp::clone() const {
  return std::make_unique<BeamSearchOp>(*this);
}

void BeamSearchOp::setup() {
  auto logitsInfo        = inInfo(logitsInIndex());
  auto hypContainerInfo  = inInfo(hypContainerInIndex());
  auto probContainerInfo = inInfo(probContainerInIndex());
  auto stepInfo          = inInfo(stepInIndex());

  // Shape check.
  auto logitsShape        = logitsInfo.shape();
  auto hypContainerShape  = hypContainerInfo.shape();
  auto probContainerShape = probContainerInfo.shape();
  auto stepShape          = stepInfo.shape();

  auto batchSize = logitsShape[0];
  auto beamSize  = getBeamSize();

  if (logitsShape.size() != 3) {
    throw popart::error("[Beam Search] The rank of logits should be 3."
                        " But now it is {}.",
                        logitsShape.size());
  }
  if (beamSize > logitsShape[2]) {
    throw popart::error(
        "[Beam Search] The parameter \'beam_size\'({}) should not."
        " be greater than the vocab_size(which is logits.shape[2]({})).",
        beamSize,
        logitsShape[2]);
  }
  if (hypContainerShape.size() != 3) {
    throw popart::error("[Beam Search] The rank of hyp_container should be 3."
                        " But now it is {}.",
                        hypContainerShape.size());
  }
  if (probContainerShape.size() != 2) {
    throw popart::error("[Beam Search] The rank of prob_container should be 2."
                        " But now it is {}.",
                        probContainerShape.size());
  }
  if (stepShape.size() > 1) {
    throw popart::error("[Beam Search] The rank of step should be less than 2."
                        " But now it is {}.",
                        stepShape.size());
  }
  if (hypContainerShape[0] != batchSize) {
    throw popart::error("[Beam Search] hyp_container.shape[0]({}) should be"
                        "equal to logits.shape[0]({}).",
                        hypContainerShape[0],
                        batchSize);
  }
  if (probContainerShape[0] != batchSize) {
    throw popart::error("[Beam Search] prob_container.shape[0]({}) should be"
                        "equal to logits.shape[0]({}).",
                        probContainerShape[0],
                        batchSize);
  }
  if (logitsShape[1] != beamSize) {
    throw popart::error("[Beam Search] logits.shape[1]({}) should be"
                        "equal to parameter \'beam_size\'({}).",
                        logitsShape[1],
                        beamSize);
  }
  if (hypContainerShape[2] != beamSize) {
    throw popart::error("[Beam Search] hyp_container.shape[2]({}) should be"
                        "equal to parameter \'beam_size\'({}).",
                        hypContainerShape[2],
                        beamSize);
  }
  if (probContainerShape[1] != beamSize) {
    throw popart::error("[Beam Search] prob_container.shape[1]({}) should be"
                        "equal to parameter \'beam_size\'({}).",
                        probContainerShape[1],
                        beamSize);
  }

  // Generate outInfo.
  auto probDtype  = probContainerInfo.dataType();
  auto indexDtype = hypContainerInfo.dataType();
  outInfo(nextIdOutIndex()) =
      popart::TensorInfo(indexDtype, probContainerShape);
  outInfo(globalBestIdOutIndex()) =
      popart::TensorInfo(indexDtype, probContainerShape);
  outInfo(hypContainerOutIndex()) =
      popart::TensorInfo(indexDtype, hypContainerShape);
  outInfo(probContainerOutIndex()) =
      popart::TensorInfo(probDtype, probContainerShape);
}

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T1 = {DataType::FLOAT, DataType::FLOAT16};
static OpDefinition::DataTypes T2 = {DataType::INT32};

static OpDefinition
    beamSearchOpDef({OpDefinition::Inputs({{"logits", T1},
                                           {"hyp_container", T2},
                                           {"prob_container", T1},
                                           {"step", T2}}),
                     OpDefinition::Outputs({{"next_id", T2},
                                            {"global_best_id", T2},
                                            {"hyp_container", T2},
                                            {"prob_container", T1}}),
                     OpDefinition::Attributes({{"beam_size", {"*"}}})});

static popart::OpCreator<BeamSearchOp> beamSearchOpCreator(
    popart::OpDefinitions({{CustomOperators::BeamSearchId, beamSearchOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      int beamSize =
          info.attributes.getAttribute<popart::Attributes::Int>("beam_size", 0);
      return std::make_unique<BeamSearchOp>(info.opid, beamSize, info.settings);
    },
    true);
} // namespace

static std::pair<poplar::Tensor, poplar::Tensor>
getTopKResult(poplar::Graph &graph,
              poplar::program::Sequence &prog,
              poplar::Tensor &input,
              const unsigned axis,
              const unsigned k,
              const poplar::DebugContext &debugContext) {
  auto lastDim = input.rank() - 1;
  // Poplibs topk requires input with rank = 2, axis = 1
  if (axis != lastDim) {
    input = input.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }
  auto shape        = input.shape();
  shape[lastDim]    = k;
  auto dim1Elememts = input.dim(lastDim);
  auto dim0Elems    = input.numElements() / dim1Elememts;
  input             = input.reshape({dim0Elems, dim1Elememts});

  auto indsShape = input.shape();
  indsShape[1]   = k;
  auto topKInds  = graph.addVariable(
      poplar::UNSIGNED_INT, indsShape, {debugContext, "topKInds"});
  poputil::mapTensorLinearly(graph, topKInds);

  auto topKVals = popnn::topK(
      graph, input, topKInds, k, true, prog, {debugContext, "topK"});

  topKInds = topKInds.reinterpret(poplar::INT);
  topKVals = topKVals.reshape(shape);
  topKInds = topKInds.reshape(shape);

  if (axis != lastDim) {
    topKVals = topKVals.dimShufflePartial({axis, lastDim}, {lastDim, axis});
    topKInds = topKInds.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }
  return std::make_pair(topKVals, topKInds);
}

// input.shape = (batchSize, dataDimSize, C)
// src.shape = (batchSize, srcDimSize, C)
// indices.shape = (batchSize, indexDimSize, 1)
static poplar::Tensor
getGroupScatterResult(poplar::Graph &graph,
                      poplar::program::Sequence &prog,
                      poplar::Tensor &input,
                      poplar::Tensor &src,
                      poplar::Tensor &indices,
                      const poplar::DebugContext &debugContext) {
  poplar::OptionFlags opts;
  opts.set("usedForSlice", "false");
  opts.set("usedForUpdate", "true");

  auto batchSize    = input.dim(0);
  auto dataLastDim  = input.dim(2);
  auto dataDimSize  = input.dim(1);
  auto srcDimSize   = src.dim(1);
  auto indexDimSize = indices.dim(1);

  auto multiUpdatePlan = popops::embedding::plan(graph,
                                                 input.elementType(),
                                                 batchSize,
                                                 dataDimSize,
                                                 dataLastDim,
                                                 {indexDimSize},
                                                 opts);

  popops::groupedMultiUpdate(graph,
                             input,
                             src,
                             indices,
                             {0},
                             {1},
                             prog,
                             multiUpdatePlan,
                             {},
                             debugContext);
  return input;
}

// input.shape = (batchSize, dataGatherDimSize, C)
// indices.shape = (batchSize, indexGatherDimSize, 1)
static poplar::Tensor
getGroupGatherResult(poplar::Graph &graph,
                     poplar::program::Sequence &prog,
                     poplar::Tensor &input,
                     poplar::Tensor &indices,
                     const poplar::DebugContext &debugContext) {
  poplar::OptionFlags opts;
  opts.set("usedForSlice", "true");
  opts.set("usedForUpdate", "false");

  auto batchSize          = input.dim(0);
  auto dataGatherDimSize  = input.dim(1);
  auto dataLastDim        = input.dim(2);
  auto indexGatherDimSize = indices.dim(1);

  auto multiSlicePlan = popops::embedding::plan(graph,
                                                input.elementType(),
                                                batchSize,
                                                dataGatherDimSize,
                                                dataLastDim,
                                                {indexGatherDimSize},
                                                opts);

  auto result = popops::groupedMultiSlice(
      graph, input, indices, {0}, {1}, prog, multiSlicePlan, {}, debugContext);
  return result;
}

class BeamSearchOpx : public popart::popx::Opx {
public:
  BeamSearchOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<BeamSearchOp>(op, {CustomOperators::BeamSearchId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<BeamSearchOp>();

    poplar::Tensor logits        = getInTensor(op.logitsInIndex());
    poplar::Tensor hypContainer  = getInTensor(op.hypContainerInIndex());
    poplar::Tensor probContainer = getInTensor(op.probContainerInIndex());
    poplar::Tensor step          = getInTensor(op.stepInIndex());

    auto hypDtype  = hypContainer.elementType();
    auto probDtype = probContainer.elementType();

    const unsigned beamSize = op.getBeamSize();

    auto batchSize = logits.shape()[0];
    auto maxLength = hypContainer.shape()[1];

    // topKResult.shape = (batchSize, beamSize, beamSize)
    auto topKResult   = getTopKResult(graph(),
                                    prog,
                                    logits,
                                    logits.rank() - 1,
                                    beamSize,
                                    debugContext("getTopKResult"));
    auto beamBestProb = topKResult.first;
    auto beamBestId   = topKResult.second;

    poplar::program::Sequence firstStepProg(debugContext("firstStep"));
    poplar::program::Sequence loopStepProg(debugContext("loopStep"));

    auto nextId = graph().addVariable(hypContainer.elementType(),
                                      probContainer.shape(),
                                      debugContext("nextId"));
    poputil::mapTensorLinearly(graph(), nextId);

    auto globalBestIdOut = graph().addVariable(hypContainer.elementType(),
                                               probContainer.shape(),
                                               debugContext("globalBestIdOut"));
    poputil::mapTensorLinearly(graph(), globalBestIdOut);
    // ======================== First Step ========================
    // nextId = beamBestId[:, 0, :]
    auto firstNextId = beamBestId.slice(0, 1, 1).reshape({batchSize, beamSize});
    firstStepProg.add(poplar::program::Copy(firstNextId, nextId));
    firstStepProg.add(poplar::program::Copy(firstNextId, globalBestIdOut));

    // hypContainer[:, 0] = nextId
    auto hypContainerSlice =
        hypContainer.slice(0, 1, 1).reshape({batchSize, beamSize});
    firstStepProg.add(poplar::program::Copy(firstNextId, hypContainerSlice));

    // probContainer = beamBestProb[:, 0, :]
    auto beamBestProbSlice =
        beamBestProb.slice(0, 1, 1).reshape({batchSize, beamSize});
    firstStepProg.add(poplar::program::Copy(beamBestProbSlice, probContainer));

    // ======================== Loop Step ========================
    // tempHypContainer.shape = (batchSize, maxLength, beamSize, beamSize)
    auto broadcastHyp = hypContainer.expand({3}).broadcast(beamSize, 3);
    auto tempHypContainer =
        graph().addVariable(hypContainer.elementType(),
                            broadcastHyp.shape(),
                            debugContext("tempHypContainer"));
    poputil::mapTensorLinearly(graph(), tempHypContainer);
    loopStepProg.add(poplar::program::Copy(broadcastHyp, tempHypContainer));

    // tempProbContainer.shape = (batchSize, beamSize, beamSize)
    auto tempProbContainer = probContainer.expand({2}).broadcast(beamSize, 2);

    // tempProbContainer += beamBestProb
    popops::addInPlace(graph(), beamBestProb, tempProbContainer, loopStepProg);

    // tempHypContainer[:, step] = beamBestId
    tempHypContainer =
        tempHypContainer.reshape({batchSize, maxLength, beamSize * beamSize});
    beamBestId = beamBestId.reshape({batchSize, 1, 1, beamSize * beamSize});
    auto stepOffset = step.reshape({1, 1, 1})
                          .broadcast(batchSize, 0)
                          .reinterpret(poplar::UNSIGNED_INT);
    tempHypContainer =
        getGroupScatterResult(graph(),
                              loopStepProg,
                              tempHypContainer,
                              beamBestId,
                              stepOffset,
                              debugContext("getGroupScatterResult"));

    tempHypContainer =
        tempHypContainer.reshape({batchSize, maxLength, beamSize * beamSize});
    beamBestProb = beamBestProb.reshape({batchSize, beamSize * beamSize});

    // globalTopKResult.shape = (batchSize, beamSize)
    auto globalTopKResult = getTopKResult(graph(),
                                          loopStepProg,
                                          beamBestProb,
                                          1,
                                          beamSize,
                                          debugContext("getTopKResult"));
    auto globalBestProb   = globalTopKResult.first;
    auto globalBestId     = globalTopKResult.second;
    loopStepProg.add(poplar::program::Copy(globalBestId, globalBestIdOut));

    // Use group gather. Group size = batchSize. Gather axis = 1.
    //   - input.shape   = (batchSize, beamSize * beamSize, 1)
    //   - indices.shape = (batchSize, beamSize, 1)
    //   - result.shape  = (batchSize, beamSize)
    globalBestId = globalBestId.reinterpret(poplar::UNSIGNED_INT)
                       .reshape({batchSize, beamSize, 1});
    beamBestId = beamBestId.reshape({batchSize, beamSize * beamSize, 1});
    auto loopNextId =
        getGroupGatherResult(graph(),
                             loopStepProg,
                             beamBestId,
                             globalBestId,
                             debugContext("getGroupGatherResult/loopNextId"));
    loopStepProg.add(poplar::program::Copy(loopNextId, nextId));

    // Use group gather. Group size = batchSize * maxLength. Gather axis = 1.
    //   - input.shape   = (batchSize * maxLength, beamSize * beamSize, 1)
    //   - indices.shape = (batchSize * maxLength, beamSize, 1)
    //   - result.shape  = (batchSize * maxLength, beamSize)
    tempHypContainer = tempHypContainer.reshape(
        {batchSize * maxLength, beamSize * beamSize, 1});
    globalBestId = globalBestId.expand({1})
                       .broadcast(maxLength, 1)
                       .reshape({batchSize * maxLength, beamSize, 1});
    auto loopHypContainer =
        getGroupGatherResult(
            graph(),
            loopStepProg,
            tempHypContainer,
            globalBestId,
            debugContext("getGroupGatherResult/loopHypContainer"))
            .reshape({batchSize, maxLength, beamSize});
    loopStepProg.add(poplar::program::Copy(loopHypContainer, hypContainer));

    auto loopProbContainer = globalBestProb;
    loopStepProg.add(poplar::program::Copy(loopProbContainer, probContainer));
    prog.add(poplar::program::If(step.reshape({}),
                                 loopStepProg,
                                 firstStepProg,
                                 debugContext("stepBranch")));

    setOutTensor(BeamSearchOp::nextIdOutIndex(), nextId);
    setOutTensor(BeamSearchOp::globalBestIdOutIndex(), globalBestIdOut);
    setOutTensor(BeamSearchOp::hypContainerOutIndex(), hypContainer);
    setOutTensor(BeamSearchOp::probContainerOutIndex(), probContainer);
  }
};

static popart::popx::OpxCreator<BeamSearchOpx>
    beamSearchOpxCreator({CustomOperators::BeamSearchId});