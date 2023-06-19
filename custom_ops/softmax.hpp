// Copyright (c) 2022 Graphcore Ltd.
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

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>


namespace CustomOperators {
const popart::OperatorIdentifier SoftmaxId = {"ai.graphcore", "SoftmaxV3", 1};
} // namespace CustomOperators

class SoftmaxOp;
class SoftmaxOpx;

class SoftmaxOp : public popart::Op {
public:
  SoftmaxOp(const popart::OperatorIdentifier &_opid,
            int axis, int stableMode,
            const popart::Op::Settings &settings_);

  std::unique_ptr<popart::Op> clone() const override;
  void setup() final;

  static popart::InIndex inputInIndex() { return 0; }
  static popart::OutIndex outIndex() { return 0; }

  int getAxis() const { return axis; }

  int getStableMode() const { return stableMode; }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

private:
  int axis;
  int stableMode;
};

SoftmaxOp::SoftmaxOp(const popart::OperatorIdentifier &_opid,
                     int axis, int stableMode,
                     const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_), axis(axis), stableMode(stableMode) {}

std::unique_ptr<popart::Op> SoftmaxOp::clone() const {
  return std::make_unique<SoftmaxOp>(*this);
}

void SoftmaxOp::setup() {
  auto rank = inRank(0);
  auto axis = getAxis() > 0 ? getAxis() : rank + getAxis();
  if (rank < 2) {
    throw popart::error("[Softmax] Invalid input tensor of rank {},"
                        " which should be 2 or higher",
                        inRank(inputInIndex()));
  }

  if (axis != rank - 1) {
    throw popart::error("[Softmax] Only support axis = {}(or -1),"
                        "but now it is {}.",
                        rank - 1,
                        axis);
  }

  auto inputShape = inShape(inputInIndex());
  // Output has the same shape as input
  outInfo(outIndex()) =
      popart::TensorInfo(inInfo(inputInIndex()).dataType(), inputShape);
}

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition softmaxOpDef({OpDefinition::Inputs({{"input", T}}),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({{"axis", {"*"}},{"stable_mode", {"*"}}})});

static popart::OpCreator<SoftmaxOp> softmaxOpCreator(
    popart::OpDefinitions({{CustomOperators::SoftmaxId, softmaxOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      int axis =
          info.attributes.getAttribute<popart::Attributes::Int>("axis", 0);
      int stableMode =
          info.attributes.getAttribute<popart::Attributes::Int>("stable_mode", 0);
      return std::make_unique<SoftmaxOp>(info.opid, axis, stableMode, info.settings);
    },
    true);
} // namespace

