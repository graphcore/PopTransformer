// Copyright (c) 2023 Graphcore Ltd.

#include <popart/error.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Zero.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <iostream>

// Int4ToHalfOp can convert int4 quantized input to half data type.

using namespace popops::expr;
namespace pe = popops::expr;

namespace CustomOperators {
const popart::OperatorIdentifier Int4ToHalfId = {"ai.graphcore",
                                                 "Int4ToHalf",
                                                 1};
} // namespace CustomOperators
class Int4ToHalfOp;
class Int4ToHalfOpx;

class Int4ToHalfOp : public popart::Op {
public:
  Int4ToHalfOp(const popart::OperatorIdentifier &_opid,
               const int axis,
               const int remap,
               const popart::Op::Settings &settings_);

  std::unique_ptr<popart::Op> clone() const override;
  void setup() final;

  int getAxis() const { return axis; }

  int getRemap() const { return remap; }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

private:
  int axis;
  int remap;
};

Int4ToHalfOp::Int4ToHalfOp(const popart::OperatorIdentifier &_opid,
                           const int axis,
                           const int remap,
                           const popart::Op::Settings &settings_)
    : popart::Op(_opid, settings_), axis(axis), remap(remap) {}

std::unique_ptr<popart::Op> Int4ToHalfOp::clone() const {
  return std::make_unique<Int4ToHalfOp>(*this);
}

void Int4ToHalfOp::setup() {
  if (inRank(0) != 2) {
    throw popart::error("[Int4ToHalfOp] The rank of input should be 2."
                        " But now it is {}.",
                        inRank(0));
  }

  if (inRank(1) != 1) {
    throw popart::error("[Int4ToHalfOp] The rank of scale should be 1."
                        " But now it is {}.",
                        inRank(1));
  }

  if (axis == 0) {
    if (inShape(0)[1] != inShape(1)[0]) {
      throw popart::error(
          "[Int4ToHalfOp] input.shape[1]({}) should be equal to "
          " scale.shape[0]({})",
          inShape(0)[1],
          inShape(1)[0]);
    }
  } else if (axis == 1) {
    if (inShape(0)[0] != inShape(1)[0]) {
      throw popart::error(
          "[Int4ToHalfOp] input.shape[0]({}) should be equal to "
          " scale.shape[0]({})",
          inShape(0)[0],
          inShape(1)[0]);
    }
  } else {
    throw popart::error("[Int4ToHalfOp] \'axis\' should be 0 or 1."
                        " But now it is {}.",
                        axis);
  }

  auto shape = inShape(0);
  auto n     = shape[0];
  auto m     = shape[1];
  if (axis == 0) {
    outInfo(0) = popart::TensorInfo(popart::DataType::FLOAT16, {2 * n, m});
  } else {
    outInfo(0) = popart::TensorInfo(popart::DataType::FLOAT16, {2 * m, n});
  }
}

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T1 = {DataType::INT8};

static OpDefinition::DataTypes T2 = {DataType::FLOAT16};

static OpDefinition int4ToHalfOpDef(
    {OpDefinition::Inputs({{"input", T1}, {"scale", T2}}),
     OpDefinition::Outputs({{"output", T2}}),
     OpDefinition::Attributes({{"axis", {"*"}}, {"remap", {"*"}}})});

static popart::OpCreator<Int4ToHalfOp> int4ToHalfOpCreator(
    popart::OpDefinitions({{CustomOperators::Int4ToHalfId, int4ToHalfOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      int axis =
          info.attributes.getAttribute<popart::Attributes::Int>("axis", 0);
      int remap =
          info.attributes.getAttribute<popart::Attributes::Int>("remap", 1);
      return std::make_unique<Int4ToHalfOp>(
          info.opid, axis, remap, info.settings);
    },
    true);
} // namespace

// Since int8(int4+int4) range in [0, 256), create a (256, 2) loopup table.
// One int8 value acrodding to two half value(high data and low data).
// Return a (numTiles, 512) table tensor for every tile.
poplar::Tensor getInt4toHalfTable(poplar::Graph &graph,
                                  poplar::program::Sequence &prog) {
  std::vector<float> table(512, 0);
  uint8_t idx = 0;
  for (unsigned i = 0; i < 256; i++) {
    auto high        = float(char(idx) >> 4);
    auto low         = float(char(idx << 4) >> 4);
    table[2 * i]     = high;
    table[2 * i + 1] = low;
    idx++;
  }
  auto tableTensor =
      graph.addConstant<float>(poplar::HALF, {512}, table.data());
  graph.setTileMapping(tableTensor, 0);

  const auto target   = graph.getTarget();
  const auto numTiles = target.getTilesPerIPU();
  auto tableBroadCast =
      graph.addVariable(poplar::HALF, {numTiles, 512}, "int4ToHalfTable");
  for (unsigned tile = 0; tile < numTiles; tile++) {
    auto tableSlice = tableBroadCast.slice(tile, tile + 1, 0).flatten();
    graph.setTileMapping(tableSlice, tile);
  }
  prog.add(poplar::program::Copy(tableTensor.expand({0}).broadcast(numTiles, 0),
                                 tableBroadCast));

  return tableBroadCast;
}

poplar::Tensor getInt4toHalfAxis0(poplar::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  poplar::Tensor &input,
                                  poplar::Tensor &scale,
                                  const int remap,
                                  const poplar::DebugContext &debugContext) {
  auto n = input.dim(0);
  auto m = input.dim(1);
  input  = input.flatten();

  const auto target     = graph.getTarget();
  const auto numTiles   = target.getTilesPerIPU();
  const auto numWorkers = target.getNumWorkerContexts();

  auto getInt4ToHalfCS = graph.addComputeSet("GetInt4ToHalfCS");
  auto vertexName      = "GetInt4ToHalf";
  auto mapping         = graph.getTileMapping(input);
  auto highData = graph.clone(poplar::HALF, input, {debugContext, "highData"});
  auto lowData  = graph.clone(poplar::HALF, input, {debugContext, "lowData"});

  auto scaleRemap1 = graph
                         .clone(poplar::HALF,
                                input.reshape({n, m}).slice(0, 1, 0),
                                {debugContext, "scaleRemap1"})
                         .flatten();
  prog.add(poplar::program::Copy(scale.flatten(), scaleRemap1));
  auto scaleRemap =
      graph.clone(poplar::HALF, input, {debugContext, "scaleRemap"});
  prog.add(
      poplar::program::Copy(scaleRemap1.broadcast(n, 0).flatten(), scaleRemap));

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(input, thisTileMap);
    auto maxElemsForRpt = target.getRptCountMax() * 4;
    auto vectorWidth    = target.getVectorWidth(poplar::HALF);
    auto vertexRegions =
        poputil::splitRegionsBetweenWorkers(target,
                                            tileContiguousRegions,
                                            2 * vectorWidth,
                                            4 * vectorWidth,
                                            UINT_MAX,
                                            maxElemsForRpt);

    for (const auto &regions : vertexRegions) {
      const auto numRegions = regions.size();
      if (numRegions == 1) {
        auto inputThisWorker = poplar::concat(input.slices(regions)).flatten();
        auto highDataThisWorker =
            poplar::concat(highData.slices(regions)).flatten();
        auto lowDataThisWorker =
            poplar::concat(lowData.slices(regions)).flatten();
        auto scaleThisWorker =
            poplar::concat(scaleRemap.slices(regions)).flatten();
        const auto totalNum = inputThisWorker.numElements();
        poplar::VertexRef getInt4ToHalfVertex =
            graph.addVertex(getInt4ToHalfCS,
                            vertexName,
                            {
                                {"size", totalNum},
                                {"input", inputThisWorker},
                                {"highData", highDataThisWorker},
                                {"lowData", lowDataThisWorker},
                                {"scale", scaleThisWorker},
                            });
        graph.setTileMapping(getInt4ToHalfVertex, tile);
      } else {
        for (const auto &r : regions) {
          auto inputThisWorker = poplar::concat(input.slices(r)).flatten();
          auto highDataThisWorker =
              poplar::concat(highData.slices(r)).flatten();
          auto lowDataThisWorker = poplar::concat(lowData.slices(r)).flatten();
          auto scaleThisWorker = poplar::concat(scaleRemap.slices(r)).flatten();
          const auto totalNum  = inputThisWorker.numElements();
          poplar::VertexRef getInt4ToHalfVertex =
              graph.addVertex(getInt4ToHalfCS,
                              vertexName,
                              {
                                  {"size", totalNum},
                                  {"input", inputThisWorker},
                                  {"highData", highDataThisWorker},
                                  {"lowData", lowDataThisWorker},
                                  {"scale", scaleThisWorker},
                              });
          graph.setTileMapping(getInt4ToHalfVertex, tile);
        }
      }
    }
  }
  prog.add(poplar::program::Execute(getInt4ToHalfCS));
  auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                          const poplar::Target &device) {
    return std::uint64_t(0);
  };
  graph.registerPerfEstimator(vertexName, zeroEstimator);

  highData    = highData.reshape({n, 1, m});
  lowData     = lowData.reshape({n, 1, m});
  auto output = poplar::concat({highData, lowData}, 1).reshape({2 * n, m});
  if (remap != 1) {
    return output;
  } else {
    auto outputRemap = graph.addVariable(poplar::HALF, output.shape());
    poputil::mapTensorLinearly(graph, outputRemap);
    prog.add(poplar::program::Copy(output, outputRemap));
    return outputRemap;
  }
}

poplar::Tensor getInt4toHalfAxis1(poplar::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  poplar::Tensor &input,
                                  poplar::Tensor &scale,
                                  const int remap,
                                  const size_t inputChannel,
                                  const poplar::DebugContext &debugContext) {
  auto n = input.dim(0);
  auto m = input.dim(1);
  input  = input.flatten();

  const auto target     = graph.getTarget();
  const auto numTiles   = target.getTilesPerIPU();
  const auto numWorkers = target.getNumWorkerContexts();

  auto getInt4ToHalfCS = graph.addComputeSet("GetInt4ToHalfCS");
  auto vertexName      = "GetInt4ToHalf1";
  auto mapping         = graph.getTileMapping(input);

  auto output =
      graph.addVariable(poplar::HALF, {n * m, 2}, {debugContext, "output"});

  auto table = getInt4toHalfTable(graph, prog);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;
    auto tableThisTile = table.slice(tile, tile + 1, 0).flatten();
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(input, thisTileMap);
    auto maxElemsForRpt = target.getRptCountMax() * 4;
    auto vectorWidth    = target.getVectorWidth(poplar::HALF);
    auto vertexRegions =
        poputil::splitRegionsBetweenWorkers(target,
                                            tileContiguousRegions,
                                            2 * vectorWidth,
                                            4 * vectorWidth,
                                            UINT_MAX,
                                            maxElemsForRpt);

    for (const auto &regions : vertexRegions) {
      const auto numRegions = regions.size();
      if (numRegions == 1) {
        auto inputThisWorker = poplar::concat(input.slices(regions)).flatten();
        auto outputThisWorker =
            poplar::concat(output.slices(regions)).flatten();
        const auto totalNum = inputThisWorker.numElements();
        poplar::VertexRef getInt4ToHalfVertex =
            graph.addVertex(getInt4ToHalfCS,
                            vertexName,
                            {{"size", totalNum},
                             {"input", inputThisWorker},
                             {"output", outputThisWorker},
                             {"table", tableThisTile}});
        graph.setTileMapping(outputThisWorker, tile);
        graph.setTileMapping(getInt4ToHalfVertex, tile);
      } else {
        for (const auto &r : regions) {
          auto inputThisWorker  = poplar::concat(input.slices(r)).flatten();
          auto outputThisWorker = poplar::concat(output.slices(r)).flatten();
          const auto totalNum   = inputThisWorker.numElements();
          poplar::VertexRef getInt4ToHalfVertex =
              graph.addVertex(getInt4ToHalfCS,
                              vertexName,
                              {{"size", totalNum},
                               {"input", inputThisWorker},
                               {"output", outputThisWorker},
                               {"table", tableThisTile}});
          graph.setTileMapping(outputThisWorker, tile);
          graph.setTileMapping(getInt4ToHalfVertex, tile);
        }
      }
    }
  }
  prog.add(poplar::program::Execute(getInt4ToHalfCS));
  auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                          const poplar::Target &device) {
    return std::uint64_t(0);
  };
  graph.registerPerfEstimator(vertexName, zeroEstimator);

  output = output.reshape({n, 2 * m}).dimShufflePartial({0}, {1});
  if (remap != 1) {
    popops::mulInPlace(graph, output, scale.reshape({1, n}), prog);
    return output;
  } else {
    auto matmulRHS =
        poplin::createMatMulGroupedInputRHS(graph,
                                            poplar::HALF,
                                            poplar::HALF,
                                            {1, inputChannel, 2 * m},
                                            {1, 2 * m, n},
                                            {debugContext, "RemapToMatmulRHS"},
                                            {});
    matmulRHS = matmulRHS.reshape({2 * m, n});
    prog.add(poplar::program::Copy(output, matmulRHS));
    popops::mulInPlace(graph, matmulRHS, scale.reshape({1, n}), prog);
    return matmulRHS;
  }
}

class Int4ToHalfOpx : public popart::popx::Opx {
public:
  Int4ToHalfOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<Int4ToHalfOp>(op, {CustomOperators::Int4ToHalfId});
  }

  void grow(poplar::program::Sequence &prog) const final {
    // graph().addCodelets("./custom_ops/build/int4_to_half_codelets.gp");
    graph().addCodelets("../../custom_ops/int4_to_half_codelets.cpp");
    auto op = getOp<Int4ToHalfOp>();

    auto axis            = op.getAxis();
    auto remap           = op.getRemap();
    poplar::Tensor input = getInTensor(0);
    poplar::Tensor scale = getInTensor(1);

    if (axis == 0) {
      auto output = getInt4toHalfAxis0(graph(),
                                       prog,
                                       input,
                                       scale,
                                       remap,
                                       debugContext("getInt4toHalfAxis0"));
      setOutTensor(0, output);
    } else {
      poplar::Tensor ref = getInTensor(2);
      auto inputChannel  = ref.dim(ref.rank() - 2);
      auto output = getInt4toHalfAxis1(graph(),
                                       prog,
                                       input,
                                       scale,
                                       remap,
                                       inputChannel,
                                       debugContext("getInt4toHalfAxis1"));
      setOutTensor(0, output);
    }
  }
};

static popart::popx::OpxCreator<Int4ToHalfOpx>
    int4ToHalfOpxCreator({CustomOperators::Int4ToHalfId});