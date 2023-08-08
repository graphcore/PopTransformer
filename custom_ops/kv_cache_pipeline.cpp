// Copyright (c) 2023 Graphcore Ltd.
#include <stdint.h>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/Quarter.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include "popops/Reduce.hpp"
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <string>
#include <vector>
#include <cstddef>
#include "kv_cache_pipeline.hpp"

using namespace popops::expr;
namespace pe = popops::expr;

class KVCachePipelineOpx : public popart::popx::Opx {
public:
  KVCachePipelineOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<KVCachePipelineOp>(op, {CustomOperators::KVCachePipelineId});
  }

  poplar::Tensor CastToUint8(poplar::program::Sequence &prog,
                             poplar::Tensor &input) const {
    poplar::Tensor scale = graph().addConstant(poplar::INT, {}, 1);
    graph().setTileMapping(scale, 0);
    poplar::QuarterMetadata::Format fp8_format =
        poplar::QuarterMetadata::Format::F143;
    auto qm =
        poplar::createVariableMetadataTensor(graph(), fp8_format, scale, prog);
    auto res_cast_fp8 =
        popops::cast(graph(), input, poplar::QUARTER, qm, prog, debugContext());
    auto out = res_cast_fp8.reinterpret(poplar::UNSIGNED_CHAR);
    return out;
  }

  poplar::Tensor CastUint8ToFP16(poplar::program::Sequence &prog,
                                 poplar::Tensor &input) const {
    poplar::Tensor scale = graph().addConstant(poplar::INT, {}, 1);
    graph().setTileMapping(scale, 0);
    poplar::QuarterMetadata::Format fp8Format =
        poplar::QuarterMetadata::Format::F143;
    auto qm =
        poplar::createVariableMetadataTensor(graph(), fp8Format, scale, prog);
    auto inputFP8 =
        graph().clone(poplar::QUARTER,
                      qm,
                      input,
                      debugContext("inputFP8"),
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

    prog.add(poplar::program::Copy(
        input, inputFP8.reinterpret(poplar::UNSIGNED_CHAR)));

    auto out =
        popops::cast(graph(), inputFP8, poplar::HALF, prog, debugContext());
    return out;
  }

  void MapTensor(poplar::Tensor &input) const {
    auto op                      = getOp<KVCachePipelineOp>();
    poplar::Target const &target = graph().getTarget();
    unsigned int tilesPerIPU     = target.getTilesPerIPU();
    auto lastDim                 = input.dim(input.rank() - 1);
    auto channelCnt              = input.numElements() / lastDim;
    auto shape                   = input.shape();
    input                        = input.reshape({channelCnt, lastDim});
    for (int i = 0; i < channelCnt; i++) {
      graph().setTileMapping(input[i], i * tilesPerIPU / channelCnt);
    }
    input = input.reshape(shape);
  }

  std::pair<poplar::Tensor, poplar::Tensor>
  getStepAndBps(poplar::program::Sequence &prog,
                poplar::Tensor &step,
                int maxBps) const {
    auto stepRemap =
        graph().addVariable(step.elementType(), step.shape(), "stepRemap");
    auto stepTrue =
        graph().addVariable(step.elementType(), step.shape(), "stepTrue");
    auto bpsIdx =
        graph().addVariable(step.elementType(), step.shape(), "bpsIdx");

    graph().setTileMapping(stepRemap, 0);
    graph().setTileMapping(stepTrue, 0);
    graph().setTileMapping(bpsIdx, 0);
    prog.add(poplar::program::Copy(step, stepRemap));
    auto getStepBpsCS = graph().addComputeSet("GetStepBpsCS");
    poplar::VertexRef getStepBpsVertex =
        graph().addVertex(getStepBpsCS,
                          "GetStepBps",
                          {
                              {"maxBps", maxBps},
                              {"step", stepRemap.flatten()},
                              {"stepTrue", stepTrue.flatten()},
                              {"bpsIdx", bpsIdx.flatten()},
                          });
    graph().setTileMapping(getStepBpsVertex, 0);
    prog.add(poplar::program::Execute(getStepBpsCS));
    auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                            const poplar::Target &device) {
      return std::uint64_t(0);
    };
    graph().registerPerfEstimator("GetStepBps", zeroEstimator);
    return {stepTrue, bpsIdx};
  }

  std::vector<poplar::Tensor> getKvCaches(poplar::program::Sequence &prog,
                                          int maxLen,
                                          int maxBps,
                                          int sequenceAxis,
                                          bool save_fp8,
                                          bool transpose,
                                          poplar::Tensor &input,
                                          poplar::Tensor &stepTrue) const {
    std::vector<poplar::Tensor> kvCaches(maxBps);
    auto kvCacheShape          = input.shape();
    kvCacheShape[sequenceAxis] = maxLen;

    auto rank = input.rank() - 1;

    for (int i = 0; i < maxBps; i++) {
      kvCaches[i] =
          graph().addVariable(input.elementType(),
                              kvCacheShape,
                              debugContext("kvCache" + std::to_string(i)));
      MapTensor(kvCaches[i]);
    }
    auto kvCachesCat = poplar::concat(kvCaches, 0).flatten();
    if (save_fp8) {
      kvCachesCat = kvCachesCat.reinterpret(poplar::CHAR);
    }

    const auto target     = graph().getTarget();
    const auto numTiles   = target.getTilesPerIPU();
    const auto numWorkers = target.getNumWorkerContexts();

    auto stepTrueRemap = graph().addVariable(
        stepTrue.elementType(), {numTiles, 1}, "stepTrueRemap");
    for (unsigned tile = 0; tile != numTiles; ++tile) {
      graph().setTileMapping(stepTrueRemap.slice(tile, tile + 1, 0), tile);
    }
    prog.add(poplar::program::Copy(
        stepTrue.expand({0}).broadcast(numTiles, 0).flatten(),
        stepTrueRemap.flatten()));

    auto getkvCachesInitCS = graph().addComputeSet("GetkvCachesInitCS");
    auto vertexName = "GetkvCachesInit_" + kvCachesCat.elementType().toString();
    auto mapping    = graph().getTileMapping(kvCachesCat);
    for (unsigned tile = 0; tile != numTiles; ++tile) {
      const auto thisTileMap = mapping[tile];
      if (thisTileMap.empty())
        continue;
      auto stepThisTile = stepTrueRemap.slice(tile, tile + 1, 0).flatten();
      const auto tileContiguousRegions =
          graph().getSortedContiguousRegions(kvCachesCat, thisTileMap);
      auto maxElemsForRpt = target.getRptCountMax() * 4;
      auto vectorWidth    = target.getVectorWidth(poplar::HALF);
      auto vertexRegions =
          poputil::splitRegionsBetweenWorkers(target,
                                              tileContiguousRegions,
                                              vectorWidth,
                                              2 * vectorWidth,
                                              UINT_MAX,
                                              maxElemsForRpt);

      for (const auto &regions : vertexRegions) {
        const auto numRegions = regions.size();
        if (numRegions == 1) {
          auto inputThisWorker =
              poplar::concat(kvCachesCat.slices(regions)).flatten();
          const auto totalNum = inputThisWorker.numElements();
          poplar::VertexRef getkvCachesInitVertex =
              graph().addVertex(getkvCachesInitCS,
                                vertexName,
                                {
                                    {"size", totalNum},
                                    {"step", stepThisTile},
                                    {"dataSlice", inputThisWorker},
                                });
          graph().setTileMapping(getkvCachesInitVertex, tile);
        } else {
          for (const auto &r : regions) {
            auto inputThisWorker =
                poplar::concat(kvCachesCat.slices(r)).flatten();
            const auto totalNum = inputThisWorker.numElements();
            poplar::VertexRef getkvCachesInitVertex =
                graph().addVertex(getkvCachesInitCS,
                                  vertexName,
                                  {
                                      {"size", totalNum},
                                      {"step", stepThisTile},
                                      {"dataSlice", inputThisWorker},
                                  });
            graph().setTileMapping(getkvCachesInitVertex, tile);
          }
        }
      }
    }
    prog.add(poplar::program::Execute(getkvCachesInitCS));
    auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                            const poplar::Target &device) {
      return std::uint64_t(0);
    };
    graph().registerPerfEstimator(vertexName, zeroEstimator);
    return kvCaches;
  }

  void getBpsKvCache(poplar::program::Sequence &prog,
                     int idx,
                     bool save_fp8,
                     bool transpose,
                     poplar::Tensor &kvCache,
                     poplar::Tensor &kvCacheClone,
                     poplar::Tensor &output,
                     poplar::Tensor &bpsIdxRemap) const {
    const auto target     = graph().getTarget();
    const auto numTiles   = target.getTilesPerIPU();
    const auto numWorkers = target.getNumWorkerContexts();

    if (!transpose) {
      auto kvCacheFlat      = kvCache.flatten();
      auto kvCacheCloneFlat = kvCacheClone.flatten();
      auto outputFlat       = output.flatten();
      if (save_fp8) {
        kvCacheFlat      = kvCacheFlat.reinterpret(poplar::CHAR);
        kvCacheCloneFlat = kvCacheCloneFlat.reinterpret(poplar::CHAR);
        outputFlat       = outputFlat.reinterpret(poplar::CHAR);
      }

      auto getBpsKvCacheCS = graph().addComputeSet("GetBpsKvCacheCS");
      auto vertexName = "GetBpsKvCache_" + kvCacheFlat.elementType().toString();
      auto mapping    = graph().getTileMapping(kvCacheFlat);
      for (unsigned tile = 0; tile != numTiles; ++tile) {
        const auto thisTileMap = mapping[tile];
        if (thisTileMap.empty())
          continue;
        auto bpsIdxThisTile = bpsIdxRemap.slice(tile, tile + 1, 0).flatten();
        const auto tileContiguousRegions =
            graph().getSortedContiguousRegions(kvCacheFlat, thisTileMap);
        auto maxElemsForRpt = target.getRptCountMax() * 4;
        auto vectorWidth    = target.getVectorWidth(poplar::HALF);
        auto vertexRegions =
            poputil::splitRegionsBetweenWorkers(target,
                                                tileContiguousRegions,
                                                vectorWidth,
                                                2 * vectorWidth,
                                                UINT_MAX,
                                                maxElemsForRpt);

        for (const auto &regions : vertexRegions) {
          const auto numRegions = regions.size();
          if (numRegions == 1) {
            auto kvCacheThisWorker =
                poplar::concat(kvCacheFlat.slices(regions)).flatten();
            auto kvCacheCloneThisWorker =
                poplar::concat(kvCacheCloneFlat.slices(regions)).flatten();
            auto outputThisWorker =
                poplar::concat(outputFlat.slices(regions)).flatten();
            const auto totalNum = kvCacheThisWorker.numElements();
            poplar::VertexRef getBpsKvCacheVertex =
                graph().addVertex(getBpsKvCacheCS,
                                  vertexName,
                                  {
                                      {"size", totalNum},
                                      {"idx", idx},
                                      {"bpsIdx", bpsIdxThisTile},
                                      {"kvCache", kvCacheThisWorker},
                                      {"kvCacheClone", kvCacheCloneThisWorker},
                                      {"output", outputThisWorker},
                                  });
            graph().setTileMapping(getBpsKvCacheVertex, tile);
          } else {
            for (const auto &r : regions) {
              auto kvCacheThisWorker =
                  poplar::concat(kvCacheFlat.slices(r)).flatten();
              auto kvCacheCloneThisWorker =
                  poplar::concat(kvCacheCloneFlat.slices(r)).flatten();
              auto outputThisWorker =
                  poplar::concat(outputFlat.slices(r)).flatten();
              const auto totalNum = kvCacheThisWorker.numElements();
              poplar::VertexRef getBpsKvCacheVertex = graph().addVertex(
                  getBpsKvCacheCS,
                  vertexName,
                  {
                      {"size", totalNum},
                      {"idx", idx},
                      {"bpsIdx", bpsIdxThisTile},
                      {"kvCache", kvCacheThisWorker},
                      {"kvCacheClone", kvCacheCloneThisWorker},
                      {"output", outputThisWorker},
                  });
              graph().setTileMapping(getBpsKvCacheVertex, tile);
            }
          }
        }
      }
      prog.add(poplar::program::Execute(getBpsKvCacheCS));
      auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                              const poplar::Target &device) {
        return std::uint64_t(0);
      };
      graph().registerPerfEstimator(vertexName, zeroEstimator);
    } else {
      auto rank        = kvCache.rank() - 1;
      auto kvCacheFlat = kvCache.flatten(0, rank);
      auto outputFlat  = output.flatten(0, rank);

      if (save_fp8) {
        kvCacheFlat = kvCacheFlat.reinterpret(poplar::CHAR);
        outputFlat  = outputFlat.reinterpret(poplar::CHAR);
      }

      auto channelCnt = kvCacheFlat.dim(0);
      auto lastDim    = kvCacheFlat.dim(1);
      std::vector<unsigned> tileStart(numTiles, 0);
      std::vector<unsigned> tileCount(numTiles, 0);
      size_t tileIdxLast = 1;
      for (unsigned i = 0; i < channelCnt; i++) {
        auto tileId = i * numTiles / channelCnt;
        if (tileIdxLast != tileId) {
          tileStart[tileId] = i;
        }
        tileCount[tileId] += 1;
        tileIdxLast = tileId;
      }
      auto getBpsKvCacheLastDimCS =
          graph().addComputeSet("GetBpsKvCacheLastDimCS");
      auto vertexName =
          "GetBpsKvCacheLastDim_" + kvCacheFlat.elementType().toString();
      for (unsigned i = 0; i < numTiles; i++) {
        if (tileCount[i] == 0) {
          continue;
        }
        auto bpsIdxThisTile = bpsIdxRemap.slice(i, i + 1, 0).flatten();

        for (unsigned j = 0; j < tileCount[i]; j++) {
          poplar::VertexRef getBpsKvCacheLastDimVertex =
              graph().addVertex(getBpsKvCacheLastDimCS,
                                vertexName,
                                {
                                    {"size", lastDim},
                                    {"idx", idx},
                                    {"bpsIdx", bpsIdxThisTile},
                                    {"kvCache", kvCacheFlat[tileStart[i] + j]},
                                    {"output", outputFlat[tileStart[i] + j]},
                                });
          graph().setTileMapping(getBpsKvCacheLastDimVertex, i);
        }
      }
      prog.add(poplar::program::Execute(getBpsKvCacheLastDimCS));
      auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                              const poplar::Target &device) {
        return std::uint64_t(0);
      };
      graph().registerPerfEstimator(vertexName, zeroEstimator);
    }
  }

  void grow(poplar::program::Sequence &prog) const final {
    auto op                      = getOp<KVCachePipelineOp>();
    std::string const &debug_str = op.getDebugStr();
    poplar::Tensor step          = getInTensor(op.stepInIndex());
    poplar::Tensor input         = getInTensor(op.inputInIndex());

    auto maxLen       = op.getMaxLen();
    auto maxBps       = op.getMaxBps();
    auto sequenceAxis = op.getSequenceAxis();
    auto save_fp8     = op.saveFP8();
    auto transpose    = op.getTranspose();

    if (save_fp8) {
      input = CastToUint8(prog, input); // cast to int8
    }

    graph().addCodelets("../../custom_ops/kv_cache_codelet.cpp");

    auto stepAndBps = getStepAndBps(prog, step, maxBps);
    auto stepTrue   = stepAndBps.first;
    auto bpsIdx     = stepAndBps.second;

    auto originAxis = sequenceAxis;
    if (transpose) {
      auto rank    = input.rank() - 1;
      input        = input.dimShufflePartial({unsigned(sequenceAxis), rank},
                                      {rank, unsigned(sequenceAxis)});
      sequenceAxis = rank;
    }
    auto kvCaches = getKvCaches(
        prog, maxLen, maxBps, sequenceAxis, save_fp8, transpose, input, step);

    auto kvCacheClone = graph().clone(kvCaches[0], "kvCacheClone");
    auto output       = graph().clone(kvCaches[0], "kcCacheOut");
    popops::zero(graph(), output, prog);
    prog.add(poplar::program::Copy(input, output.slice(0, 1, sequenceAxis)));

    const auto target   = graph().getTarget();
    const auto numTiles = target.getTilesPerIPU();
    auto bpsIdxRemap =
        graph().addVariable(bpsIdx.elementType(), {numTiles, 1}, "bpsIdxRemap");
    for (unsigned tile = 0; tile != numTiles; ++tile) {
      graph().setTileMapping(bpsIdxRemap.slice(tile, tile + 1, 0), tile);
    }
    prog.add(poplar::program::Copy(
        bpsIdx.expand({0}).broadcast(numTiles, 0).flatten(),
        bpsIdxRemap.flatten()));

    for (int i = 0; i < maxBps; i++) {
      auto kvCacheSlice =
          kvCaches[i].slice(0, op.getMaxLen() - op.getStepLen(), sequenceAxis);
      auto kvCacheConcat = poplar::concat({input, kvCacheSlice}, sequenceAxis);

      if (!transpose) {
        prog.add(poplar::program::Copy(kvCacheConcat, kvCacheClone));
      }

      getBpsKvCache(prog,
                    i,
                    save_fp8,
                    transpose,
                    kvCaches[i],
                    kvCacheClone,
                    output,
                    bpsIdxRemap);
    }

    if (save_fp8) {
      output = CastUint8ToFP16(prog, output);
    }

    if (transpose) {
      auto rank = output.rank() - 1;
      output    = output.dimShufflePartial({unsigned(originAxis), rank},
                                        {rank, unsigned(originAxis)});
    }
    setOutTensor(KVCachePipelineOp::outIndex(), output);
  }
};

static popart::popx::OpxCreator<KVCachePipelineOpx>
    KVCachePipelineOpxCreator({CustomOperators::KVCachePipelineId});
