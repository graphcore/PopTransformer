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

#include <poplar/Engine.hpp>
#include <poplar/DeviceManager.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Rearrange.hpp>
#include <poputil/VertexTemplates.hpp>
#include "NormCEImpl.hpp"
#include "TileMappingCommon.hpp"

namespace celib{

NormCEOutput normCE(poplar::Graph&              graph, 
                    poplar::program::Sequence&  prog,
                    poplar::Tensor &            input,
                    poplar::Tensor const&       scale,
                    poplar::Tensor const&       bias,
                    float                       epsilon,
                    int                         num_groups,
                    bool                        stable_algo,
                    std::string const&          debug_str){
  unsigned int     last_dim_size = input.dim(input.rank() - 1);
  unsigned int     channel_cnt   = input.numElements() / last_dim_size;
  poplar::Tensor   meanInvStdDevPair = graph.addVariable(poplar::FLOAT, 
                                             { 2 * channel_cnt }, debug_str + "/mean_invStdDev");
  poplar::Tensor        output = graph.addVariable(input.elementType(), 
                                             input.shape(), debug_str + "/NormCE_output");
  poplar::Tensor        scale_cpy = graph.addVariable(scale.elementType(), 
                                             { last_dim_size}, debug_str + "/NormCE_scale_cpy");
  poplar::Tensor        bias_cpy = graph.addVariable(bias.elementType(), 
                                             { last_dim_size}, debug_str + "/NormCE_bias_cpy");
  poplar::Tensor        mean;
  poplar::Tensor        invStdDev;
  poplar::Target const& target       = graph.getTarget();
  unsigned int          numTiles     = target.getNumTiles();
  unsigned int          wrk_cnt      = target.getNumWorkerContexts();
  unsigned int          tilesPerIPU  = target.getTilesPerIPU();
  poplar::ComputeSet    norm_cs      = graph.addComputeSet(debug_str + "/normCE");
  std::string           vertex_name  = std::string("celib::NormCEVertex");

  graph.setTileMapping(scale_cpy,  tilesPerIPU - 1);
  graph.setTileMapping(bias_cpy,   tilesPerIPU - 2);

  prog.add(poplar::program::Copy(poplar::concat({scale, bias}), poplar::concat({scale_cpy, bias_cpy})));

  SplitChannelInfo splitInfo  = splitChannelByGroupFull(channel_cnt,  1, numTiles, tilesPerIPU);
  auto             inReshape  = input.reshape({ channel_cnt, (size_t)last_dim_size });
  auto             outReshape = output.reshape({ channel_cnt, (size_t)last_dim_size });
  std::vector<size_t> const& tileStart  = std::get<0>(splitInfo);
  std::vector<size_t> const& tileCount  = std::get<1>(splitInfo);

  float            invScale   = 1.0f / (float)last_dim_size;
  for (unsigned i = 0; i < numTiles; ++i)
  {
    if(0 == tileCount[i])
      continue;

    const auto curIn        = inReshape.slice( tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
    const auto curOut       = outReshape.slice(tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
    const auto curMean      = meanInvStdDevPair.slice(2 * tileStart[i], 2 * tileStart[i] + 2 * tileCount[i], 0).flatten();
    graph.setTileMapping(curOut,  i);
    graph.setTileMapping(curMean, i);
    auto       norm_vertex  = graph.addVertex(norm_cs, poputil::templateVertex(vertex_name, input.elementType()));
    graph.setTileMapping(norm_vertex,  i);
    auto       work_sub_info = split_worker_1dvector(tileCount[i], last_dim_size, wrk_cnt);
    for(int j = 0 ; j < wrk_cnt ; j ++){
      work_sub_info.second[j]  = work_sub_info.second[j] / last_dim_size;
    }
    graph.connect(norm_vertex["in_"],          curIn);
    graph.connect(norm_vertex["scale_"],       scale_cpy);
    graph.connect(norm_vertex["bias_"],        bias_cpy);
    graph.connect(norm_vertex["out_"],         curOut);
    graph.connect(norm_vertex["meanDevPair_"], curMean);
    graph.setInitialValue(norm_vertex["inner_size_"], last_dim_size);
    graph.setInitialValue(norm_vertex["work_size_"],  work_sub_info.first);
    graph.setInitialValue(norm_vertex["work_ofs_"],   work_sub_info.second);
    graph.setInitialValue(norm_vertex["invScale_"],   invScale);
    graph.setInitialValue(norm_vertex["eps_"],        epsilon);
  }
  prog.add(poplar::program::Execute(norm_cs));

  //arrange mean&invStdDev
  if(poplar::HALF == input.elementType() || poplar::FLOAT == input.elementType()){
    poplar::Tensor   mean_arrange      = graph.addVariable(input.elementType(), 
                                              { channel_cnt }, debug_str + "/mean");
    poplar::Tensor   invStdDev_arrange = graph.addVariable(input.elementType(), 
                                              { channel_cnt }, debug_str + "/invStdDev");

    auto arrange_split_info = splitTensorByGrainSize(mean_arrange.numElements(), 8, numTiles, tilesPerIPU);
    std::vector<size_t> const& arrange_tileStart = std::get<0>(arrange_split_info);
    std::vector<size_t> const& arrange_tileCount = std::get<1>(arrange_split_info);
    poplar::ComputeSet    arrange_cs           = graph.addComputeSet(debug_str + "/float2real");
    std::string           arrange_vertex_name  = std::string("celib::Float2RealVertex");
    for (unsigned i = 0; i < numTiles; ++i)
    {
      if(0 == arrange_tileCount[i])
        continue;

      const auto curMean             = meanInvStdDevPair.slice( 2 * arrange_tileStart[i], 2 * arrange_tileStart[i] + 2 * arrange_tileCount[i], 0);
      const auto curMeanArrange      = mean_arrange.slice(      arrange_tileStart[i], arrange_tileStart[i] + arrange_tileCount[i], 0);
      const auto curInvStdDevArrange = invStdDev_arrange.slice( arrange_tileStart[i], arrange_tileStart[i] + arrange_tileCount[i], 0);
      graph.setTileMapping(curMeanArrange,       i);
      graph.setTileMapping(curInvStdDevArrange,  i);
      auto  arrange_vertex  = graph.addVertex(arrange_cs, poputil::templateVertex(arrange_vertex_name, input.elementType()));
      graph.setTileMapping(arrange_vertex,  i);
      graph.connect(arrange_vertex["meanDevPair_"],       curMean);
      graph.connect(arrange_vertex["meanData_"],          curMeanArrange);
      graph.connect(arrange_vertex["invStdDevData_"],     curInvStdDevArrange);
      graph.setInitialValue(arrange_vertex["data_size_"], arrange_tileCount[i]);
    }
    prog.add(poplar::program::Execute(arrange_cs));
    prog.add(poplar::program::WriteUndef(meanInvStdDevPair));
    prog.add(poplar::program::WriteUndef(scale_cpy));
    prog.add(poplar::program::WriteUndef(bias_cpy));
    mean      = mean_arrange;
    invStdDev = invStdDev_arrange;
  }else{
    throw poplar::poplar_error("just support float and half");
  }

  return { mean, invStdDev, output };
}

static std::pair<std::vector<int>, std::vector<int>> splitOneTask(int task_size, int grain_size, int work_num){
  int  task_grain_cnt       = (task_size + grain_size - 1) / grain_size;
  int  grain_cnt_per_worker = (task_grain_cnt + work_num - 1) / work_num;
  int  ele_cnt_per_worker   = grain_cnt_per_worker * grain_size;
  int  remain_cnt           = task_size;
  int  cur_ofs              = 0;
  int  work_idx             = 0;
  std::vector<int>   work_ofs(work_num, 0);
  std::vector<int>   work_size(work_num, 0);
  while(remain_cnt > 0){
    int cur_work_size   = remain_cnt > ele_cnt_per_worker ? ele_cnt_per_worker : remain_cnt;
    work_ofs[work_idx]  = cur_ofs;
    work_size[work_idx] = cur_work_size;
    remain_cnt -= cur_work_size;
    cur_ofs    += cur_work_size;
    work_idx++;
  }

  return { work_size, work_ofs };
}

static std::pair<std::vector<int>, std::vector<int>> splitTwoTask(int task_size, int grain_size, int work_num){ 
  int  task_grain_cnt       = (task_size + grain_size - 1) / grain_size;
  int  work_per_task        = work_num / 2;
  int  grain_cnt_per_worker = (task_grain_cnt + work_per_task - 1) / work_per_task;
  int  ele_cnt_per_worker   = grain_cnt_per_worker * grain_size;
  int  remain_cnt           = task_size;
  int  remain_work_cnt      = work_num;
  std::vector<int>   work_ofs(work_num, 0);
  std::vector<int>   work_size(work_num, 0);
  int  cur_ofs              = 0;
  int  work_idx             = 0;
  while(remain_cnt > 0){
    int cur_work_size   = remain_cnt > ele_cnt_per_worker ? ele_cnt_per_worker : remain_cnt;
    work_ofs[work_idx]  = cur_ofs;
    work_size[work_idx] = cur_work_size;
    remain_cnt -= cur_work_size;
    cur_ofs    += cur_work_size;
    work_idx++;
  }

  for(int i = work_per_task ; i < work_num ; i ++){
    work_ofs[i]  = task_size + work_ofs[i - work_per_task];
    work_size[i] = work_size[i - work_per_task];
  }

  return { work_size, work_ofs };
}

static std::pair<std::vector<int>, std::vector<int>> splitThreeTask(int task_size, int grain_size, int work_num){ 
  int  task_grain_cnt       = (task_size + grain_size - 1) / grain_size;
  int  work_per_task        = work_num / 3;
  int  grain_cnt_per_worker = (task_grain_cnt + work_per_task - 1) / work_per_task;
  int  ele_cnt_per_worker   = grain_cnt_per_worker * grain_size;
  int  remain_cnt           = task_size;
  int  remain_work_cnt      = work_num;
  std::vector<int>   work_ofs(work_num, 0);
  std::vector<int>   work_size(work_num, 0);
  int  cur_ofs              = 0;
  int  work_idx             = 0;
  while(remain_cnt > 0){
    int cur_work_size   = remain_cnt > ele_cnt_per_worker ? ele_cnt_per_worker : remain_cnt;
    work_ofs[work_idx]  = cur_ofs;
    work_size[work_idx] = cur_work_size;
    remain_cnt -= cur_work_size;
    cur_ofs    += cur_work_size;
    work_idx++;
  }

  for(int i = work_per_task ; i < 2 * work_per_task ; i ++){
    work_ofs[i]  = task_size + work_ofs[i - work_per_task];
    work_size[i] = work_size[i - work_per_task];
    work_ofs[i + work_per_task]  = 2 * task_size + work_ofs[i - work_per_task];
    work_size[i + work_per_task] = work_size[i - work_per_task];
  }

  return { work_size, work_ofs };
}

poplar::Tensor normCEInf(poplar::Graph&              graph, 
                         poplar::program::Sequence&  prog,
                         poplar::Tensor &            input,
                         poplar::Tensor const&       scale,
                         poplar::Tensor const&       bias,
                         float                       epsilon,
                         int                         num_groups,
                         bool                        stable_algo,
                         std::string const&          debug_str){
  if(poplar::FLOAT != input.elementType() &&
    poplar::HALF != input.elementType()){
    throw poplar::poplar_error("normCEInf just support float and half");
  }
  unsigned int     last_dim_size = input.dim(input.rank() - 1);
  unsigned int     channel_cnt   = input.numElements() / last_dim_size;
  poplar::Tensor        output = graph.addVariable(input.elementType(), 
                                             input.shape(), debug_str + "/NormCE_output");
  /*
  poplar::Tensor        scale_cpy = graph.addVariable(scale.elementType(), 
                                             { last_dim_size}, debug_str + "/NormCE_scale_cpy");
  poplar::Tensor        bias_cpy = graph.addVariable(bias.elementType(), 
                                             { last_dim_size}, debug_str + "/NormCE_bias_cpy");
  */
  poplar::Tensor        mean;
  poplar::Tensor        invStdDev;
  poplar::Target const& target       = graph.getTarget();
  unsigned int          numTiles     = target.getNumTiles();
  unsigned int          wrk_cnt      = target.getNumWorkerContexts();
  unsigned int          tilesPerIPU  = target.getTilesPerIPU();

  //graph.setTileMapping(scale_cpy,  tilesPerIPU - 1);
  //graph.setTileMapping(bias_cpy,   tilesPerIPU - 2);

  //prog.add(poplar::program::Copy(poplar::concat({scale, bias}), poplar::concat({scale_cpy, bias_cpy})));

  SplitChannelInfo splitInfo  = splitChannelByGroupFull(channel_cnt,  1, numTiles, tilesPerIPU);
  auto             inReshape  = input.reshape({ channel_cnt, (size_t)last_dim_size });
  auto             regroup_remap = graph.addVariable(input.elementType(), 
                                    { 
                                      channel_cnt, (size_t)last_dim_size
                                    }, 
                                    debug_str + std::string("/normCE_input_remap"));
  std::vector<size_t> const& tileStart  = std::get<0>(splitInfo);
  std::vector<size_t> const& tileCount  = std::get<1>(splitInfo);
  for (unsigned i = 0; i < numTiles; ++i)
  {
    if(0 == tileCount[i])
      continue;
          
    auto curOut = regroup_remap.slice(tileStart[i], tileStart[i] + tileCount[i], 0).flatten();
    graph.setTileMapping(curOut, i);
  }
  prog.add(poplar::program::Copy(inReshape, regroup_remap));
  inReshape = regroup_remap;

  auto             outReshape = output.reshape({ channel_cnt, (size_t)last_dim_size });
  std::vector<size_t>        bc_tensor_idx(tileCount.size(), 0);
  int                        valid_cnt  = 0;
  for (unsigned i = 0; i < numTiles; ++i){
    if(0 == tileCount[i])
      continue;
    valid_cnt++;
  }

  poplar::Tensor scale_bc =  graph.addVariable(scale.elementType(), 
                                             { (size_t)valid_cnt, (size_t)last_dim_size}, 
                                             debug_str + "/NormCE_scale_bc");
  poplar::Tensor bias_bc  =  graph.addVariable(bias.elementType(), 
                                             { (size_t)valid_cnt, (size_t)last_dim_size}, 
                                             debug_str + "/NormCE_bias_bc");
  valid_cnt = 0;
  for (unsigned i = 0; i < numTiles; ++i){
    if(0 == tileCount[i])
      continue;
    graph.setTileMapping(scale_bc[valid_cnt].flatten(), i);
    graph.setTileMapping(bias_bc[valid_cnt].flatten(),  i);
    bc_tensor_idx[i] = valid_cnt;
    valid_cnt ++;
  }
  /*
  prog.add(poplar::program::Copy(scale_cpy.reshape({1, (size_t)last_dim_size}).broadcast(valid_cnt, 0).flatten(), 
                                 scale_bc.flatten()));
  prog.add(poplar::program::Copy(bias_cpy.reshape( {1, (size_t)last_dim_size}).broadcast(valid_cnt, 0).flatten(), 
                                 bias_bc.flatten()));
  */
  prog.add(poplar::program::Copy(scale.reshape({1, (size_t)last_dim_size}).broadcast(valid_cnt, 0).flatten(), 
                                 scale_bc.flatten()));
  prog.add(poplar::program::Copy(bias.reshape( {1, (size_t)last_dim_size}).broadcast(valid_cnt, 0).flatten(), 
                                 bias_bc.flatten()));
  valid_cnt = 0;
  std::vector<size_t>        tileTskElapsed(tileCount.size(), 0);
  float   invScale         = 1.0f / (float)last_dim_size;
  int     execute_stage    = 1;
  int     tile_remain_max  = tileCount[0];
  bool    first_stage      = true;
  bool    allFinished      = false;
  int     s                = 0;

  int     grainSize        = 2;
  if(poplar::HALF == input.elementType())
    grainSize = 4;
  int   real_grain_size    = grainSize;
  bool  interleve_mem      = false;
  if(0 == (last_dim_size&7)){
    real_grain_size = 8;
    interleve_mem   = true;
    if((0 == (last_dim_size % (real_grain_size * wrk_cnt))) && 
        ((last_dim_size % (real_grain_size * wrk_cnt)) >= 2)){
      //we accept it, or add contraint for it in future
    }else if((12 * real_grain_size * wrk_cnt) > last_dim_size){
      real_grain_size = 4;
      interleve_mem   = false;
    }
  }

  while(false == allFinished){
    poplar::ComputeSet norm_cs   = graph.addComputeSet(debug_str + std::string("/normCEInf_") + std::to_string(s));
    valid_cnt = 0;
    bool  curRoundFinished      = true;
    int   curRoundTileRemainMax = 0;
    for (unsigned i = 0; i < numTiles; ++i)
    {
      int   curTaskElapsed = tileTskElapsed[i];
      int   curTaskCnt     = tileCount[i] - curTaskElapsed;
      int   curTskTail     = (curTaskCnt % 6);

      if(curTaskCnt != curTskTail){
        curTaskCnt = curTaskCnt - curTskTail;
      }else{
        curTaskCnt = curTskTail;
        if(0 != (tile_remain_max % 6)){
          if(4 == curTskTail || 5 == curTskTail){
            curTaskCnt = 3;
          }
        }
      }

      int remain_tsk = tileCount[i] - (curTaskElapsed + curTaskCnt);
      if(curRoundTileRemainMax < remain_tsk){
        curRoundTileRemainMax = remain_tsk;
      }
      curRoundFinished = curRoundFinished & (0 == remain_tsk);

      if(0 == curTaskCnt)
        continue;

      const auto curIn  = inReshape.slice( tileStart[i] + curTaskElapsed, tileStart[i] + curTaskElapsed + curTaskCnt, 0).flatten();
      const auto curOut = outReshape.slice(tileStart[i] + curTaskElapsed, tileStart[i] + curTaskElapsed + curTaskCnt, 0).flatten();
      graph.setTileMapping(curOut,  i);
      poplar::VertexRef  norm_vertex;
      bool               nonSplit  = false;
      if(poplar::HALF == input.elementType() && (0 != (last_dim_size & 1)) && curTaskCnt > 1)
        nonSplit = true;
      if(curTaskCnt <= 3 && ((poplar::HALF  == input.elementType() && last_dim_size >= 24) ||
                              (poplar::FLOAT == input.elementType() && last_dim_size >= 12)) && 
        false == nonSplit){
        std::string  vertex_name  = std::string("celib::NormCEInfSplitVertex");
        std::pair<std::vector<int>, std::vector<int>>  work_sub_info;
        if(1 == curTaskCnt){
          norm_vertex   = graph.addVertex(norm_cs, poputil::templateVertex(vertex_name, input.elementType(), 1, interleve_mem));
          work_sub_info = splitOneTask(last_dim_size, real_grain_size, wrk_cnt);
        }else if(2 == curTaskCnt){
          norm_vertex   = graph.addVertex(norm_cs, poputil::templateVertex(vertex_name, input.elementType(), 2, interleve_mem));
          work_sub_info = splitTwoTask(last_dim_size, real_grain_size, wrk_cnt);
        }else{
          norm_vertex   = graph.addVertex(norm_cs, poputil::templateVertex(vertex_name, input.elementType(), 3, interleve_mem));
          work_sub_info = splitThreeTask(last_dim_size, real_grain_size, wrk_cnt);
        }
        graph.setInitialValue(norm_vertex["work_size_"],  work_sub_info.first);
        graph.setInitialValue(norm_vertex["work_ofs_"],   work_sub_info.second);
      }else{
        std::string           vertex_name  = std::string("celib::NormCEInfVertex");
        norm_vertex  = graph.addVertex(norm_cs, poputil::templateVertex(vertex_name, input.elementType(), interleve_mem));
        auto       work_sub_info = split_worker_1dvector(curTaskCnt, last_dim_size, wrk_cnt);
        for(int j = 0 ; j < wrk_cnt ; j ++){
          work_sub_info.second[j]  = work_sub_info.second[j] / last_dim_size;
        }
        graph.setInitialValue(norm_vertex["work_size_"],  work_sub_info.first);
        graph.setInitialValue(norm_vertex["work_ofs_"],   work_sub_info.second);
      }
      graph.setTileMapping(norm_vertex,  i);
      graph.connect(norm_vertex["in_"],                 curIn);
      graph.connect(norm_vertex["scale_"],              scale_bc[bc_tensor_idx[i]].flatten());
      graph.connect(norm_vertex["bias_"],               bias_bc[bc_tensor_idx[i]].flatten());
      graph.connect(norm_vertex["out_"],                curOut);
      graph.setInitialValue(norm_vertex["inner_size_"], last_dim_size);
      graph.setInitialValue(norm_vertex["invScale_"],   invScale);
      graph.setInitialValue(norm_vertex["eps_"],        epsilon);

      tileTskElapsed[i] += curTaskCnt;
      tile_remain_max    = curRoundTileRemainMax;
      valid_cnt ++;
    }
    if(valid_cnt > 0)
      prog.add(poplar::program::Execute(norm_cs));
    allFinished = curRoundFinished;
    s ++;
  }
  //prog.add(poplar::program::WriteUndef(scale_cpy));
  //prog.add(poplar::program::WriteUndef(bias_cpy));
  prog.add(poplar::program::WriteUndef(scale_bc));
  prog.add(poplar::program::WriteUndef(bias_bc));
  prog.add(poplar::program::WriteUndef(regroup_remap));

  return output;
}

poplar::VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NormCEVertex)(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &fpType) {
  const bool    isFloat      = fpType == poplar::FLOAT;
  const auto    total_size   = vertex.getFieldInfo("in_").size();
  const auto    inner_size   = vertex.getFieldInfo("inner_size_").getInitialValue<int>(target);
  const auto    group_num    = total_size / inner_size;
  uint64_t      reduce_flpos = (inner_size + 2 * inner_size);
  uint64_t      norm_flops   = 4 * inner_size;
  uint64_t      flops        = group_num * (reduce_flpos + norm_flops);
  uint64_t      cycles       = flops;
  if(true == isFloat){

  }else{
    uint64_t  reduce_inner_loop_cycles    = ((inner_size >> 2) << 2);
    uint64_t  reduce_tail_cycles          = (inner_size & 3) * 3 + 4;
    uint64_t  reduce_other_cycles         = 20;
    uint64_t  reduce_cycle                = group_num * 2 * (reduce_inner_loop_cycles + reduce_tail_cycles) +
                                                 reduce_other_cycles;
    uint64_t  normalize_inner_loop_cycles = ((inner_size >> 2) << 2) * 4;
    uint64_t  normalize_tail_cycles       = (inner_size & 3) * 4;
    uint64_t  normalize_cycle             = group_num * (normalize_inner_loop_cycles + normalize_tail_cycles);
    cycles = 100 + reduce_cycle + normalize_cycle;
  }

  return {cycles, flops};
}

poplar::VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(Float2RealVertex)(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &fpType) {
  const bool  isFloat      = fpType == poplar::FLOAT;
  const auto  data_size    = vertex.getFieldInfo("data_size_").getInitialValue<int>(target);
  uint64_t    flops        = data_size;
  uint64_t    cycles       = data_size;
  if(true == isFloat){

  }else{
    cycles = 20 + (data_size >> 1) * 6;
  }

  return {cycles, flops};
}

poplar::VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NormCEInfVertex)(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &fpType, bool interleave_mem) {
  const bool    isFloat      = fpType == poplar::FLOAT;
  const auto    total_size   = vertex.getFieldInfo("in_").size();
  const auto    inner_size   = vertex.getFieldInfo("inner_size_").getInitialValue<int>(target);
  const auto    group_num    = total_size / inner_size;
  uint64_t      reduce_flpos = (inner_size + 2 * inner_size);
  uint64_t      norm_flops   = 4 * inner_size;
  uint64_t      flops        = group_num * (reduce_flpos + norm_flops);
  uint64_t      cycles       = flops;
  if(true == isFloat){

  }else{
    uint64_t  reduce_inner_loop_cycles    = ((inner_size >> 2) << 2);
    uint64_t  reduce_tail_cycles          = (inner_size & 3) * 4 + 4;
    if(true == interleave_mem){
      reduce_inner_loop_cycles = ((inner_size >> 3) << 3);
      reduce_tail_cycles       = (inner_size & 7) * 4 + 8;
    }
    uint64_t  reduce_other_cycles         = 20;
    uint64_t  reduce_cycle                = group_num * 2 * (reduce_inner_loop_cycles + reduce_tail_cycles) +
                                                 reduce_other_cycles;
    uint64_t  normalize_inner_loop_cycles = ((inner_size >> 2) << 2) * 4;
    uint64_t  normalize_tail_cycles       = (inner_size & 3) * 4;
    uint64_t  normalize_cycle             = group_num * (normalize_inner_loop_cycles + normalize_tail_cycles);
    cycles = 100 + reduce_cycle + normalize_cycle;
  }

  return {cycles, flops};
}

poplar::VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(NormCEInfSplitVertex)(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &fpType, int split_size, bool interleave_mem) {
  const bool    isFloat      = fpType == poplar::FLOAT;
  const auto    total_size   = vertex.getFieldInfo("in_").size();
  const auto    inner_size   = vertex.getFieldInfo("inner_size_").getInitialValue<int>(target);
  const auto    group_num    = total_size / inner_size;
  uint64_t      reduce_flpos = (inner_size + 2 * inner_size);
  uint64_t      norm_flops   = 4 * inner_size;
  uint64_t      flops        = group_num * (reduce_flpos + norm_flops);
  uint64_t      cycles       = flops;
  if(true == isFloat){

  }else{
    uint64_t  reduce_inner_loop_cycles    = ((inner_size >> 2) << 2);
    uint64_t  reduce_tail_cycles          = (inner_size & 3) * 4 + 4;
    if(true == interleave_mem){
      reduce_inner_loop_cycles = ((inner_size >> 3) << 1);
      reduce_tail_cycles       = (inner_size & 7) * 4 + 8;
    }

    uint64_t  reduce_other_cycles         = 20;
    uint64_t  reduce_cycle                = group_num * 2 * (reduce_inner_loop_cycles + reduce_tail_cycles) +
                                                 reduce_other_cycles;
    uint64_t  normalize_inner_loop_cycles = ((inner_size >> 2) << 2) * 4;
    uint64_t  normalize_tail_cycles       = (inner_size & 3) * 4;
    uint64_t  normalize_cycle             = group_num * (normalize_inner_loop_cycles + normalize_tail_cycles);
    cycles = 100 + reduce_cycle + normalize_cycle;
  }

  return {cycles, flops};
}

poputil::internal::PerfEstimatorTable makePerfFunctionTable()
{
return {
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEVertex, poplar::HALF),
      CYCLE_ESTIMATOR_ENTRY(celib, Float2RealVertex, poplar::HALF),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfVertex, poplar::HALF, false),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfVertex, poplar::HALF, true),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 1, false),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 2, false),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 3, false),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 1, true),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 2, true),
      CYCLE_ESTIMATOR_ENTRY(celib, NormCEInfSplitVertex, poplar::HALF, 3, true),
  };
}

} //namespace celib