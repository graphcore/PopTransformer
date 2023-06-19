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
#include <popops/Zero.hpp>
#include <popops/Cast.hpp>
#include <poputil/TileMapping.hpp>
#include <string>
#include <vector>
#include <cstddef>
#include "kv_cache.hpp"
#include "TileMappingCommon.hpp"


class KVCacheOpx : public popart::popx::Opx {
public:
  KVCacheOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<KVCacheOp>(op, {CustomOperators::KVCacheId});
  }

  poplar::Tensor CastToUint8(poplar::program::Sequence &prog, poplar::Tensor &input) const
  {
    poplar::Tensor scale = graph().addConstant(poplar::INT, {}, 1);
    graph().setTileMapping(scale, 0);
    poplar::QuarterMetadata::Format fp8_format = poplar::QuarterMetadata::Format::F143;
    auto qm = poplar::createVariableMetadataTensor(graph(), fp8_format, scale, prog);
    auto res_cast_fp8 = popops::cast(graph(), input, poplar::QUARTER, qm, prog, debugContext());
    auto out = res_cast_fp8.reinterpret(poplar::UNSIGNED_CHAR);
    return out;
  }


  poplar::Tensor CastUint8ToFP16(poplar::program::Sequence &prog, poplar::Tensor &input) const 
  {
    poplar::Tensor scale = graph().addConstant(poplar::INT, {}, 1);
    graph().setTileMapping(scale, 0);
    poplar::QuarterMetadata::Format fp8Format = poplar::QuarterMetadata::Format::F143;
    auto qm = poplar::createVariableMetadataTensor(graph(), fp8Format, scale, prog);
    auto inputFP8 = graph().clone(poplar::QUARTER,
                                qm,
                                input,
                                debugContext("inputFP8"),
                                poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

    prog.add(poplar::program::Copy(input, inputFP8.reinterpret(poplar::UNSIGNED_CHAR)));
   
    auto out = popops::cast(graph(),
                        inputFP8,
                        poplar::HALF,
                        prog,
                        debugContext());
    return out;
  }


  void MapTensor(poplar::Tensor &input) const
  {
    auto op = getOp<KVCacheOp>();
    poplar::Target const& target = graph().getTarget();
    unsigned int tilesPerIPU = target.getTilesPerIPU();
    auto lastDim = input.dim(input.rank() - 1);
    auto channelCnt  = input.numElements() / lastDim;
    auto shape = input.shape();
    input = input.reshape({channelCnt, lastDim});
    for (int i = 0; i < channelCnt; i++) {
      graph().setTileMapping(input[i], i * tilesPerIPU / channelCnt);
    }
    input = input.reshape(shape);
  }


  void grow(poplar::program::Sequence &prog) const final {
    auto op = getOp<KVCacheOp>();
    auto save_fp8 = op.saveFP8();
    std::string const& debug_str=op.getDebugStr();
    // scalar, int32
    poplar::Tensor step = getInTensor(op.stepInIndex());
    // tensor, shape: [bs, 2, max_len, hidden_size]
    poplar::Tensor input = getInTensor(op.inputInIndex());
    if (save_fp8){
      input = CastToUint8(prog, input); //cast to int8
    }
    auto out_shape = input.shape();
    out_shape[op.getSequenceAxis()] = op.getMaxLen();

    auto kv_cache = graph().addVariable(input.elementType(),
                                        out_shape,
                                        debugContext(debug_str));
    MapTensor(kv_cache);

    auto kv_cache_clone = graph().clone(kv_cache);

    poplar::program::Sequence firstStepProg(debugContext("firstStep"));
    poplar::program::Sequence loopStepProg(debugContext("loopStep"));

    popops::zero(graph(), kv_cache_clone, firstStepProg);
    loopStepProg.add(poplar::program::Copy(kv_cache, kv_cache_clone));

    prog.add(poplar::program::If(step,
                                 loopStepProg,
                                 firstStepProg,
                                 debugContext("stepBranch")));

    if (op.getSequenceAxis() == input.rank() - 2 || !save_fp8) {
      auto kv_cache_slice = kv_cache_clone.slice(
          0, op.getMaxLen() - op.getStepLen(), op.getSequenceAxis());
      auto kv_cache_concat = poplar::concat(
          {input, kv_cache_slice}, op.getSequenceAxis());

      prog.add(poplar::program::Copy(kv_cache_concat, kv_cache));
    } else if (op.getSequenceAxis() == input.rank() - 1) {
      graph().addCodelets("./custom_ops/kv_cache_codelet.cpp");
      const auto numTiles = graph().getTarget().getTilesPerIPU();
      auto cacheShape = kv_cache.shape();
      auto rank = input.rank();
      kv_cache = kv_cache.flatten(0, rank - 1);
      kv_cache_clone = kv_cache_clone.flatten(0, rank - 1);
      input = input.flatten(0, rank - 1);
      auto channelCnt = kv_cache.dim(0);
      auto lastDim = kv_cache.dim(1);
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
      auto getKVCacheCS = graph().addComputeSet("KVCacheCS");
      auto vertexName = "KVCache";
      for (unsigned i = 0; i < numTiles; i++) {
        if (tileCount[i] == 0) {
          continue;
        }

        for (unsigned j = 0; j < tileCount[i]; j++) {
          poplar::VertexRef getKVCacheVertex =
              graph().addVertex(getKVCacheCS,
                              vertexName,
                              {
                                  {"size", lastDim},
                                  {"input", input[tileStart[i] + j]},
                                  {"kvCacheClone", kv_cache_clone[tileStart[i] + j]},
                                  {"kvCache", kv_cache[tileStart[i] + j]},
                              });
          graph().setTileMapping(getKVCacheVertex, i);
        }
      }
      prog.add(poplar::program::Execute(getKVCacheCS));
      auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                              const poplar::Target &device) {
        return std::uint64_t(0);
      };
      graph().registerPerfEstimator(vertexName, zeroEstimator);
      kv_cache = kv_cache.reshape(cacheShape);

    }
    if (save_fp8) {
      kv_cache = CastUint8ToFP16(prog, kv_cache);
    }
    setOutTensor(KVCacheOp::outIndex(), kv_cache);
  }
};

static popart::popx::OpxCreator<KVCacheOpx>
    KVCacheOpxCreator({CustomOperators::KVCacheId});