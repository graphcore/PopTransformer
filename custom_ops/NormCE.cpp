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

#include <popart/popx/opxmanager.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>
#include <poplar/ArrayRef.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <poputil/Util.hpp>
#include <popart/shapeinference.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/region.hpp>
#include <popnn/GroupNorm.hpp>
#include <functional>
#include "NormCE.hpp"
#include "TileMappingCommon.hpp"
#include "NormCEImpl.hpp"


void inPlaceNorm(poplar::Graph &graph,
                 poplar::program::Sequence &prog,
                 poplar::Tensor &input,
                 poplar::Tensor &output,
                 poplar::Tensor &mean,
                 poplar::Tensor &power,
                 poplar::Tensor &bias,
                 poplar::Tensor &scale,
                 const float epsilon,
                 const poplar::DebugContext &debugContext) {
  const auto target     = graph.getTarget();
  const auto numTiles   = target.getTilesPerIPU();
  const auto numWorkers = target.getNumWorkerContexts();
  auto getInPlaceNormCS = graph.addComputeSet("GetInPlaceNormCS");
  auto vertexName       = "GetInPlaceNorm";
  auto mapping          = graph.getTileMapping(input);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty()) {
      continue;
    }
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(input, thisTileMap);
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
        auto inputThisWorker = poplar::concat(input.slices(regions)).flatten();
        auto outThisWorker   = poplar::concat(output.slices(regions)).flatten();
        auto biasThisWorker  = poplar::concat(bias.slices(regions)).flatten();
        auto scaleThisWorker = poplar::concat(scale.slices(regions)).flatten();
        const auto totalNum  = inputThisWorker.numElements();
        poplar::VertexRef getInPlaceNormVertex =
            graph.addVertex(getInPlaceNormCS,
                            vertexName,
                            {
                                {"size", totalNum},
                                {"input", inputThisWorker},
                                {"bias", biasThisWorker},
                                {"scale", scaleThisWorker},
                                {"mean", mean.flatten()},
                                {"power", power.flatten()},
                                {"epsilon", epsilon},
                                {"output", outThisWorker},
                            });
        graph.setTileMapping(getInPlaceNormVertex, tile);
      } else {
        for (const auto &r : regions) {
          auto inputThisWorker = poplar::concat(input.slices(r)).flatten();
          auto outThisWorker   = poplar::concat(output.slices(r)).flatten();
          auto biasThisWorker  = poplar::concat(bias.slices(r)).flatten();
          auto scaleThisWorker = poplar::concat(scale.slices(r)).flatten();
          const auto totalNum  = inputThisWorker.numElements();
          poplar::VertexRef getInPlaceNormVertex =
              graph.addVertex(getInPlaceNormCS,
                              vertexName,
                              {
                                  {"size", totalNum},
                                  {"input", inputThisWorker},
                                  {"bias", biasThisWorker},
                                  {"scale", scaleThisWorker},
                                  {"mean", mean.flatten()},
                                  {"power", power.flatten()},
                                  {"epsilon", epsilon},
                                  {"output", outThisWorker},
                              });
          graph.setTileMapping(getInPlaceNormVertex, tile);
        }
      }
    }
  }
  prog.add(poplar::program::Execute(getInPlaceNormCS));
  auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                          const poplar::Target &device) {
    return std::uint64_t(0);
  };
  graph.registerPerfEstimator(vertexName, zeroEstimator);
}

static poplar::Tensor
calSmallBatchNorm(poplar::Graph &graph,
                  poplar::program::Sequence &prog,
                  poplar::Tensor &input,
                  poplar::Tensor &scale,
                  poplar::Tensor &bias,
                  float epsilon,
                  bool stable_algo,
                  const poplar::DebugContext &debugContext) {
  auto batchSize = input.dim(0);
  auto dimSize   = input.dim(1);

  poplar::OptionFlags flags{{"groupNormStridedChannelGrouping", "false"}};

  // Calculate the normalization
  // TODO: Need to find a better way to choose instead of magic number.
  if (batchSize < 16) {
    auto scaleTensor = graph.addConstant(
        poplar::FLOAT, {}, float(1.0 / dimSize), {debugContext, "scaleTensor"});
    graph.setTileMapping(scaleTensor, 0);
    popops::ReduceParams meanParams{popops::Operation::ADD, false, scaleTensor};
    popops::ReduceParams powerParams{
        popops::Operation::SQUARE_ADD, false, scaleTensor};
    auto mean =
        poputil::createBroadcastOperand(graph,
                                        input,
                                        poplar::FLOAT,
                                        0,
                                        true,
                                        {debugContext, "/mean/ReduceResult"});
    auto power =
        graph.clone(poplar::FLOAT, mean, {debugContext, "/power/ReduceResult"});
    std::vector<poplar::Tensor> outputs = {std::move(mean), std::move(power)};
    std::vector<popops::SingleReduceOp> reductions = {
        popops::SingleReduceOp{/*in     = */ input,
                               /*dims   = */ {1},
                               /*params = */ std::move(meanParams),
                               /*debugName = */ "mean"},
        popops::SingleReduceOp{/*in     = */ input,
                               /*dims   = */ {1},
                               /*params = */ std::move(powerParams),
                               /*debugName = */ "power"}};
    popops::reduceMany(
        graph, reductions, outputs, prog, {debugContext, "/reduceMany"});
    mean  = std::move(outputs[0]);
    power = std::move(outputs[1]);

    auto biasPre  = bias.reshape({1, dimSize});
    auto scalePre = scale.reshape({1, dimSize});
    if (batchSize > 1) {
      biasPre  = biasPre.broadcast(batchSize, 0);
      scalePre = scalePre.broadcast(batchSize, 0);
    }
    auto output     = graph.clone(input, "output");
    auto biasRemap  = graph.clone(input, "biasRemap");
    auto scaleRemap = graph.clone(input, "biasRemap");
    prog.add(poplar::program::Copy(biasPre, biasRemap));
    prog.add(poplar::program::Copy(scalePre, scaleRemap));

    for (unsigned i = 0; i < batchSize; i++) {
      auto meanSlice   = mean.slice(i, i + 1, 0).flatten();
      auto powerSlice  = power.slice(i, i + 1, 0).flatten();
      auto inputSlice  = input.slice(i, i + 1, 0).flatten();
      auto outputSlice = output.slice(i, i + 1, 0).flatten();
      auto biasSlice   = biasRemap.slice(i, i + 1, 0).flatten();
      auto scaleSlice  = scaleRemap.slice(i, i + 1, 0).flatten();

      inPlaceNorm(graph,
                  prog,
                  inputSlice,
                  outputSlice,
                  meanSlice,
                  powerSlice,
                  biasSlice,
                  scaleSlice,
                  epsilon,
                  {debugContext, "getNormResult"});
    }
    return output;
  } else {
    // Calculate the mean and the inverse standard deviation
    poplar::Tensor mean;
    poplar::Tensor invStdDev;
    std::tie(mean, invStdDev) =
        popnn::gn::groupNormStatistics(graph,
                                       input,
                                       epsilon,
                                       prog,
                                       static_cast<unsigned int>(1),
                                       false,
                                       stable_algo,
                                       poplar::FLOAT,
                                       {debugContext, "getMeanAndInvStdDev"},
                                       flags);

    auto result = popnn::gn::groupNormalise(graph,
                                            input,
                                            scale,
                                            bias,
                                            mean,
                                            invStdDev,
                                            prog,
                                            {debugContext, "getNormResult"},
                                            flags);
    return result.first;
  }
}

static poplar::Tensor add_graph_prog(poplar::Graph&              graph, 
                                   poplar::program::Sequence&  prog,
                                   poplar::Tensor const&       input,
                                   poplar::Tensor const&       scale,
                                   poplar::Tensor const&       bias,
                                   bool                      after_matmul,
                                   int64_t                   grain_size,
                                   float                     epsilon,
                                   int64_t                   num_groups,
                                   bool                      stable_algo,
                                   bool                      isBwd,
                                   std::string const&        debug_str)
{
  poplar::Target const& target      = graph.getTarget();
  unsigned int          numTiles    = target.getNumTiles();
  unsigned int          tilesPerIPU = target.getTilesPerIPU();
  poplar::Tensor          src         = input;
  size_t                input_rank  = src.rank();
  if(input_rank < 2){
    throw poplar::poplar_error("input_tensor.rank should be >= 2 in normCEOp, and current rank is: " + std::to_string(input_rank));
  }
  size_t       x_dim_size   = src.dim(input_rank - 1);
  size_t       y_dim_size   = src.numElements() / x_dim_size;
  auto         src_reshape  = src.reshape({ 
                                    y_dim_size, 
                                    x_dim_size
                                });
  bool         need_regroup = true;
  int regroup_size = 1;
  if(0 == (x_dim_size & 15)){
    regroup_size = 16;
  }else if(0 == (x_dim_size & 7)){
    regroup_size = 8;
  }else if(0 == (x_dim_size & 3)){
    regroup_size = 4;
  }else{
    need_regroup = false;
  }

  bool    regroup_res = false;
  if(true == after_matmul){
    if(regroup_size > 1){
      poplar::Tensor&  src_reshape_poplar = src_reshape;
      const auto       inGrouping = poputil::detectDimGroupings(graph, src_reshape_poplar);  
      if(!inGrouping.empty()){
        if((inGrouping[0].first == 1) &&
           ((0 == (inGrouping[0].second % regroup_size)) ||
           (inGrouping[0].second > regroup_size) ||
           (8 == inGrouping[0].second))){
            regroup_res = true;
        }else{
          auto             input_tilemapping = graph.getTileMapping(src_reshape);
          poplar::Tensor   regroup           = popops::rearrange::regroupIfBeneficial(graph, 
                                                                                      src_reshape, 
                                                                                      regroup_size, 
                                                                                      prog, 
                                                                                      { debug_str + std::string("/normCE_regroup_out0") });
          auto              regroup_tilemapping = graph.getTileMapping(regroup);
          if(input_tilemapping != regroup_tilemapping){
            regroup = regroup.reshape(src.shape());
            src     = regroup, graph;
            regroup_res = true;
          }
        }
      }
    }

    if(false == regroup_res){
      size_t  grain_size_after_matmul = grain_size;
      if(0 == (x_dim_size & 15)){
        grain_size_after_matmul = 16;
      }else if(0 == (x_dim_size & 7)){
        grain_size_after_matmul = 8;
      }else if(0 == (x_dim_size & 3)){
        grain_size_after_matmul = 4;
      }else{
        throw poplar::poplar_error("[NormCE]current tensor's last dim size is not 4x/8x/16x");
      }
      src_reshape = src.reshape({ 
                                  y_dim_size, 
                                  x_dim_size / grain_size_after_matmul,
                                  grain_size_after_matmul 
                                });
      src_reshape = src_reshape.dimShuffle( { 1, 0, 2 } );
      auto  matmul_out_remap = graph.addVariable(src.elementType(), 
                                                    { 
                                                        x_dim_size / grain_size_after_matmul, 
                                                        y_dim_size, 
                                                        grain_size_after_matmul 
                                                    }, 
                                                    debug_str + std::string("/normCE_matmul_out_remap"));
      SplitChannelInfo splitInfo    = splitChannelByGroup((x_dim_size / grain_size_after_matmul) * y_dim_size,  
                                                            1, numTiles, tilesPerIPU);
      auto  matmulOutRemapReshape = matmul_out_remap.reshape({ (x_dim_size / grain_size_after_matmul) * y_dim_size, 
                                                                                    (size_t)grain_size_after_matmul });
      std::vector<size_t> const& tileStart  = std::get<0>(splitInfo);
      std::vector<size_t> const& tileCount  = std::get<1>(splitInfo);
      for (unsigned i = 0; i < numTiles; ++i)
      {
            if(0 == tileCount[i])
            continue;
            
            poplar::Tensor curOut = matmulOutRemapReshape.slice(tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
            graph.setTileMapping(curOut, i);
      }
      prog.add(poplar::program::Copy(src_reshape, matmul_out_remap));
      matmul_out_remap = matmul_out_remap.dimShuffle( { 1, 0, 2 } );
      matmul_out_remap = matmul_out_remap.reshape( {y_dim_size, x_dim_size} );
      matmul_out_remap = matmul_out_remap.reshape(src.shape());
      src              = matmul_out_remap;
    }
  }else if(true == need_regroup){
    poplar::Tensor&  src_poplar = src;
    const auto       inGrouping = poputil::detectDimGroupings(graph, src_poplar);  
    if(!inGrouping.empty()){
      if((inGrouping[0].first == (input_rank - 1)) &&
         ((0 == (inGrouping[0].second % regroup_size)) ||
         (inGrouping[0].second > regroup_size) ||
         (8 == inGrouping[0].second))){
         regroup_res = true;
      }else{
        auto             input_tilemapping = graph.getTileMapping(src_poplar);
        poplar::Tensor   regroup           = popops::rearrange::regroupIfBeneficial(graph, 
                                                                                    src_poplar, 
                                                                                    regroup_size, 
                                                                                    prog, 
                                                                                    { debug_str + std::string("/normCE_regroup_out1") });
        auto              regroup_tilemapping = graph.getTileMapping(regroup);
        if(input_tilemapping != regroup_tilemapping){
          regroup = regroup.reshape(src.shape());
          src     = regroup, graph;
          regroup_res = true;
        }
      }
    }
  }

  poplar::Tensor  out;
  if(false == isBwd){
    auto res = celib::normCEInf(graph, 
                         prog, 
                         src,
                         scale,
                         bias,
                         epsilon,
                         num_groups,
                         stable_algo,
                         debug_str + "/NormCEInf");
    out = res;
    out = out.reshape(src.shape());
  }

  if(true == regroup_res){
    prog.add(poplar::program::WriteUndef(src));
  }

  return out;
}

NormCEOp::NormCEOp(OperatorIdentifier const& opid, 
                   Op::Settings const&       settings_, 
                   bool                      fwd_after_matmul,
                   bool                      bwd_after_matmul,
                   int64_t                   fwd_grain_size,
                   int64_t                   bwd_grain_size,
                   float                     epsilon,
                   int64_t                   num_groups,
                   bool                      stable_algo,
                   std::string const&        debug_str):Op(opid, settings_) {
  fwd_after_matmul_ = fwd_after_matmul;
  bwd_after_matmul_ = bwd_after_matmul;
  fwd_grain_size_   = fwd_grain_size;
  bwd_grain_size_   = (0 == bwd_grain_size ? fwd_grain_size : bwd_grain_size);
  epsilon_          = epsilon;
  num_groups_       = num_groups;
  stable_algo_      = stable_algo;
  debug_str_        = debug_str;
}

std::vector<std::unique_ptr<Op>> NormCEOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<NormCEGradOp>(*this));

  return upops;
}

std::unique_ptr<Op> NormCEOp::clone() const {
  return std::make_unique<NormCEOp>(*this);
}

void NormCEOp::setup() {
  Shape data_shape = inInfo(0).shape();
  outInfo(0) = {inInfo(0).dataType(), data_shape};
}

poprithms::memory::inplace::Proposal
NormCEOp::mapInplaceProposal(const AliasModel &aliasModel,
                                 OperatorIdentifier id) const {
  return Op::mapInplaceProposal(aliasModel, id);
}

ReplicatedTensorShardingIndices
NormCEOp::getReplicatedTensorShardingIndices() const {
  return {{{0}, {0}}};
}

void NormCEOp::growAliasModel(AliasModel &m) const {
  Op::growAliasModel(m);
}

std::vector<std::tuple<OperatorIdentifier, float>>
NormCEOp::inplacePriorityDefault() const {
  return {};
}

std::unique_ptr<Op>
NormCEOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  return Op::getInplaceVariant(operator_id);
}
//register op
static OpDefinition::DataTypes NormCEOpDataTensorType = { DataType::FLOAT16,
                                                          DataType::FLOAT };

static OpDefinition
    normCEOpDef({
      OpDefinition::Inputs
      (
        {
          {"data",   NormCEOpDataTensorType},
          {"scale",  NormCEOpDataTensorType},
          {"bias",   NormCEOpDataTensorType},
        }
      ),
      OpDefinition::Outputs
      (
        {
          {"out",  NormCEOpDataTensorType}
        }
      ),
      OpDefinition::Attributes
      (
        {
          {"fwd_after_matmul", {"*"}},
          {"bwd_after_matmul", {"*"}},
          {"fwd_grain_size",   {"*"}},
          {"bwd_grain_size",   {"*"}},
          {"epsilon",          {"*"}},
          {"num_groups",       {"*"}},
          {"stable_algo",      {"*"}},
        }
      )
    });

static OpCreator<NormCEOp> NormOpCECreator(
    OpDefinitions({{CustomOperators::normCEId, normCEOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
       OperatorIdentifier const& opid             = info.opid;
       Op::Settings const&       settings_        = info.settings;
       Attributes const&         attr             = info.attributes;
       int64_t                   fwd_after_matmul = attr.getAttribute<Attributes::Int>("fwd_after_matmul", false);
       int64_t                   bwd_after_matmul = attr.getAttribute<Attributes::Int>("bwd_after_matmul", false);
       int64_t                   fwd_grain_size   = attr.getAttribute<Attributes::Int>("fwd_grain_size", -1);
       int64_t                   bwd_grain_size   = attr.getAttribute<Attributes::Int>("bwd_grain_size", -1);
       float                     epsilon          = attr.getAttribute<Attributes::Float>("epsilon", 1e-5f);
       int64_t                   num_groups       = attr.getAttribute<Attributes::Int>("num_groups");
       int64_t                   stable_algo      = attr.getAttribute<Attributes::Int>("stable_algo");
       std::string               debug_str        = attr.getAttribute<Attributes::String>("debug_str", "normCE");
      return std::unique_ptr<Op>(new NormCEOp(opid, 
                                              settings_, 
                                              fwd_grain_size, 
                                              bwd_grain_size, 
                                              fwd_after_matmul,
                                              bwd_after_matmul,
                                              epsilon,
                                              num_groups,
                                              0 == stable_algo ? false : true,
                                              debug_str));
    },
    true);


NormCEBaseOpx::NormCEBaseOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType NormCEBaseOpx::getInputCreatorType(InIndex) const {
  NormCEOp& normOp = getOp<NormCEOp>();
  return InputCreatorType::Deadend;
}

poplar::Tensor NormCEBaseOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return tensor;
}

view::RegMap NormCEBaseOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

NormCEOutplaceOpx::NormCEOutplaceOpx(Op *op, Devicex *devicex) : NormCEBaseOpx(op, devicex) {

  verifyOp<NormCEOp>(op, CustomOperators::normCEId);
}

void NormCEOutplaceOpx::grow(poplar::program::Sequence& prog) const {
  graph().addCodelets(std::string("../../NormCECodelets.gp"));

  auto          input_tensor  = getInTensor(0);
  auto          scale_tensor  = getInTensor(1);
  auto          bias_tensor   = getInTensor(2);
  std::string   exception_str = std::string("our normCEOp just support HALF, and current elementType is: ");
  if(poplar::HALF != input_tensor.elementType()){
    std::stringstream str ;
    str << input_tensor.elementType();
    throw poplar::poplar_error(exception_str + str.str());
  }
  if(poplar::HALF != scale_tensor.elementType()){
    std::stringstream str ;
    str << input_tensor.elementType();
    throw poplar::poplar_error(exception_str + str.str());
  }
  if(poplar::HALF != bias_tensor.elementType()){
    std::stringstream str ;
    str << input_tensor.elementType();
    throw poplar::poplar_error(exception_str + str.str());
  }

  std::size_t        dim_rank      = input_tensor.rank();
  std::size_t        last_dim_size = input_tensor.dim(dim_rank - 1);
  if(0 != (last_dim_size & 3)){
    throw poplar::poplar_error("our normCEOp require 4x a last dim, and current is: " + std::to_string(last_dim_size));
  }
  if(last_dim_size >= 8192){
    throw poplar::poplar_error("normalize_ce, last_dim_size should be <= 8192, and current is: " + std::to_string(last_dim_size));
  }
  std::size_t        channel_cnt   = input_tensor.numElements() / last_dim_size;
  NormCEOp&          norm_op      = getOp<NormCEOp>();
  bool               after_matmul = (0 != norm_op.isFwdAfterMatmul() ? true : false);
  int64_t            grain_size   = norm_op.getFwdGrainSize();
  float              epsilon      = norm_op.getEpsilon();
  int64_t            num_groups   = norm_op.getNumGroups();
  if(num_groups != channel_cnt){
    throw poplar::poplar_error("our normCEOp just support normalize row by row");
  }
  bool               stable_algo  = norm_op.isStableAlgo();
  if(true == stable_algo){
    throw poplar::poplar_error("our normCEOp just support stable_algo is false");
  }

  if (channel_cnt == 1) {
     auto shape   = input_tensor.shape();
     input_tensor = input_tensor.reshape({1, last_dim_size});
     auto out     = calSmallBatchNorm(graph(),
                                      prog,
                                      input_tensor,
                                      scale_tensor,
                                      bias_tensor,
                                      epsilon,
                                      stable_algo,
                                      debugContext("calSmallBatchNorm"));
     setOutTensor(0, out.reshape(shape));
     return;
   }

  poputil::internal::registerPerfFunctions(graph(), 
                                           celib::makePerfFunctionTable());
  std::string const& debug_str    = norm_op.getDebugStr();
  auto               out_tensor   = add_graph_prog(graph(), 
                                                   prog, 
                                                   input_tensor, 
                                                   scale_tensor,
                                                   bias_tensor,
                                                   after_matmul,
                                                   grain_size,
                                                   epsilon,
                                                   num_groups,
                                                   stable_algo,
                                                   false,
                                                   debug_str);
  setOutTensor(0, out_tensor);
}

NormCEGradOp::NormCEGradOp(const NormCEOp &fwdOp)
    : popart::Op(CustomGradOperators::normCEGradId, fwdOp.getSettings()) {
  after_matmul_  = fwdOp.isBwdAfterMatmul();
  grain_size_    = fwdOp.getBwdGrainSize();
  debug_str_     = fwdOp.getDebugStr();
}

std::unique_ptr<Op> NormCEGradOp::clone() const {
  return std::make_unique<NormCEGradOp>(*this);
}

void NormCEGradOpx::grow(poplar::program::Sequence &prog) const {

  throw poplar::poplar_error("we don't support bwd for normCEOp");
}

static popart::popx::OpxCreator<NormCEOutplaceOpx> NormCEOpxCreator(CustomOperators::normCEId);

static popart::popx::OpxCreator<NormCEGradOpx> NormCEGradOpxCreator(CustomGradOperators::normCEGradId);

// static popart::RegisterShapeInferenceFunction
//     normCEOpShapeInference(CustomOperators::normCEId,
//                          [](auto &ctx) { ctx.outInfo(0) = ctx.inInfo(0); });
