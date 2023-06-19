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

#pragma once
#include <popart/error.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/shapeinference.hpp>


namespace CustomOperators {
const popart::OperatorIdentifier KVCacheId = {"ai.graphcore", "KVCache", 1};
} // namespace CustomOperators

class KVCacheOp;
class KVCacheOpx;

class KVCacheOp : public popart::Op {
public:
  KVCacheOp(const popart::OperatorIdentifier &_opid,
            int max_len, int step_len, int sequence_axis, bool fp8, std::string const debug_str,
            const popart::Op::Settings &settings_);

  std::unique_ptr<popart::Op> clone() const override;
  void setup() final;

  static popart::InIndex stepInIndex() { return 0; }
  static popart::InIndex inputInIndex() { return 1; }
  static popart::InIndex sequenceAxisInIndex() { return 2; }
  static popart::OutIndex outIndex() { return 0; }

  int getMaxLen() const { return max_len; }
  int getStepLen() const { return step_len; }
  int getSequenceAxis() const { return sequence_axis; }
  bool saveFP8() const { return save_fp8; }
  std::string const getDebugStr() const  { return debug_str_; };
  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  bool isOutlineable() const override { return false; }

private:
  int max_len;
  int step_len;
  int sequence_axis;
  bool save_fp8;
  std::string debug_str_;
};

KVCacheOp::KVCacheOp(const popart::OperatorIdentifier &_opid,
                     int max_len, int step_len, int sequence_axis, bool fp8, std::string const debug_str,
                     const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_), max_len(max_len), step_len(step_len), sequence_axis(sequence_axis), save_fp8(fp8), debug_str_(debug_str) {}

std::unique_ptr<popart::Op> KVCacheOp::clone() const {
  return std::make_unique<KVCacheOp>(*this);
}

void KVCacheOp::setup() {
  auto oShape = inShape(inputInIndex());
  oShape[sequence_axis] = max_len;

  outInfo(outIndex()) =
      popart::TensorInfo(inInfo(inputInIndex()).dataType(), oShape);
}

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T1 = {DataType::INT32};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition KVCacheOpDef({OpDefinition::Inputs({{"step", T1}, {"input", T2}}),
                                  OpDefinition::Outputs({{"output", T2}}),
                                  OpDefinition::Attributes({{"max_len", {"*"}}, {"step_len", {"*"}}, {"sequence_axis", {"*"}}, {"fp8", {"*"}}, {"debug_str", {"*"}}})
                                });

static popart::OpCreator<KVCacheOp> kvCacheOpCreator(
    popart::OpDefinitions({{CustomOperators::KVCacheId, KVCacheOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      int max_len =
          info.attributes.getAttribute<popart::Attributes::Int>("max_len", 0);
      int step_len = 
          info.attributes.getAttribute<popart::Attributes::Int>("step_len", 1);
      int sequence_axis = 
          info.attributes.getAttribute<popart::Attributes::Int>("sequence_axis", 0);
      int fp8 = 
          info.attributes.getAttribute<popart::Attributes::Int>("fp8", 0);
      std::string  debug_str = info.attributes.getAttribute<popart::Attributes::String>("debug_str", "kv_cache");
      return std::make_unique<KVCacheOp>(info.opid, max_len, step_len, sequence_axis, fp8, debug_str, info.settings);
    },
    true);
} // namespace