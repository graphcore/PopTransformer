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

#include <stdint.h>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/shapeinference.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <string>
#include <vector>
#include <cstddef>

#include "softmax.hpp"


poplar::Tensor getSoftmax(poplar::Graph &graph,
                          const poplar::Tensor &input,
                          const int stableMode,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext) {
  auto reduceNum = input.dim(input.rank() - 1);
  auto batchSize = input.numElements() / reduceNum;
  auto inputRemap = graph.addVariable(
      input.elementType(), {batchSize, reduceNum}, {debugContext, "inputRemap"});
  poputil::mapTensorLinearly(graph, inputRemap, 1, reduceNum);
  prog.add(poplar::program::Copy(input.reshape(inputRemap.shape()), inputRemap));
  if (stableMode != 0) {
    auto max = popops::reduce(graph,
                              inputRemap,
                              {inputRemap.rank() - 1},
                              popops::Operation::MAX,
                              prog,
                              {debugContext, "Max"});
    max = max.expand({inputRemap.rank() - 1}).broadcast(reduceNum, inputRemap.rank() - 1);
    popops::subInPlace(graph, inputRemap, max, prog, {debugContext, "Sub"});

    popops::expInPlace(graph, inputRemap, prog, {debugContext, "Exp"});
  } else {
    popops::expInPlace(graph, inputRemap, prog, {debugContext, "Exp"});
  }
  auto sumF = popops::reduce(graph,
                             inputRemap,
                             poplar::FLOAT,
                             {inputRemap.rank() - 1},
                             popops::Operation::ADD,
                             prog,
                             {debugContext, "Exp"});
  popops::invInPlace(graph, sumF, prog, {debugContext, "Inv"});

  auto sum = (inputRemap.elementType() == poplar::HALF )
                 ? popops::cast(
                       graph, sumF, poplar::HALF, prog, {debugContext, "Cast"})
                 : sumF;
  auto oneOverSum =
      sum.expand({inputRemap.rank() - 1}).broadcast(reduceNum, inputRemap.rank() - 1);
  popops::mulInPlace(graph, inputRemap, oneOverSum, prog, {debugContext, "Mul"});
  return inputRemap.reshape(input.shape());
}

class SoftmaxOpx : public popart::popx::Opx {
public:
  SoftmaxOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<SoftmaxOp>(op, {CustomOperators::SoftmaxId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<SoftmaxOp>();
    poplar::Tensor input = getInTensor(op.inputInIndex());
    auto stableMode = op.getStableMode();

    auto out = getSoftmax(graph(), input, stableMode, prog, debugContext("getSoftmax"));
    setOutTensor(SoftmaxOp::outIndex(), out);
  }
};

static popart::popx::OpxCreator<SoftmaxOpx>
    softmaxOpxCreator({CustomOperators::SoftmaxId});
static popart::RegisterShapeInferenceFunction
    softmaxOpShapeInference(CustomOperators::SoftmaxId,
                            [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });

