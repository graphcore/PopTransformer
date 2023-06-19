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

#include "TileMappingCommon.hpp"
#include <string.h>

SplitChannelInfo splitChannelByGroup(size_t   channel_cnt, 
                                     size_t   group_size, 
                                     size_t   num_tiles,
                                     size_t   tiles_per_ipu)
{
  size_t               tile_idx_last    = 0xFFFFFFFF;
  std::vector<size_t>  tile_start(num_tiles, 0);
  std::vector<size_t>  tile_count(num_tiles, 0);
  size_t               regroup_channel  = (channel_cnt / group_size);
  if(((regroup_channel*group_size) != channel_cnt))
  {
    group_size      = 1;
    regroup_channel = channel_cnt;
    for (size_t i = 0; i < channel_cnt; ++i)
    {
      size_t idx = ((unsigned long long)i * (unsigned long long)num_tiles) / ((unsigned long long)channel_cnt);
      if(tile_idx_last != idx)
        tile_start[idx] = i;
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }
  else
  {
    size_t               ipu_num          = num_tiles / tiles_per_ipu;
    size_t               channels_per_ipu = ((regroup_channel + (ipu_num - 1)) / ipu_num) * group_size;
    std::vector<size_t>  ipuChannels(ipu_num, 0);
    size_t               ipuReainChannels = channel_cnt;
    for(size_t i = 0 ; i < ipu_num ; i ++){
      ipuChannels[i]    = ipuReainChannels < channels_per_ipu ? ipuReainChannels : channels_per_ipu;
      ipuReainChannels -= ipuChannels[i];
    }

    size_t  channel_ofs = 0;
    for(size_t i = 0 ; i < ipu_num ; i ++){
      size_t   channel_per_tile = (((ipuChannels[i]/group_size) + (tiles_per_ipu - 1)) / tiles_per_ipu) * group_size;
      size_t   remain_channel   = ipuChannels[i];
      size_t   tile_idx         = i * tiles_per_ipu;
      while(remain_channel > 0)
      {
        tile_start[tile_idx] = channel_ofs;
        tile_count[tile_idx] = remain_channel < channel_per_tile ? remain_channel : channel_per_tile;
        channel_ofs        += channel_per_tile;
        remain_channel     -= tile_count[tile_idx];
        tile_idx ++;
      }
    }
  }

  return { tile_start, tile_count, group_size };
}

SplitChannelInfo splitChannelByGroupFull(size_t   channel_cnt, 
                                         size_t   group_size, 
                                         size_t   num_tiles,
                                         size_t   tiles_per_ipu)
{
  size_t               tile_idx_last    = 0xFFFFFFFF;
  std::vector<size_t>  tile_start(num_tiles, 0);
  std::vector<size_t>  tile_count(num_tiles, 0);
  size_t               regroup_channel  = (channel_cnt / group_size);
  if(((regroup_channel*group_size) != channel_cnt))
  {
    group_size      = 1;
    regroup_channel = channel_cnt;
    for (size_t i = 0; i < channel_cnt; ++i)
    {
      size_t idx = ((unsigned long long)i * (unsigned long long)num_tiles) / ((unsigned long long)channel_cnt);
      if(tile_idx_last != idx)
        tile_start[idx] = i;
      tile_count[idx] += 1;
      tile_idx_last    = idx;
    }
  }
  else
  {
    size_t               ipu_num           = num_tiles / tiles_per_ipu;
    size_t               channels_per_ipu  = ((regroup_channel + (ipu_num - 1)) / ipu_num) * group_size;
    std::vector<size_t>  ipuChannels(ipu_num, 0);
    size_t               ipuRemainChannels = channel_cnt;
    for(size_t i = 0 ; i < ipu_num ; i ++){
      ipuChannels[i]     = ipuRemainChannels < channels_per_ipu ? ipuRemainChannels : channels_per_ipu;
      ipuRemainChannels -= ipuChannels[i];
    }

    size_t  channel_ofs = 0;
    for(size_t i = 0 ; i < ipu_num ; i ++){
      size_t   channel_per_tile = ((ipuChannels[i]/group_size) / tiles_per_ipu) * group_size;
      size_t   remain_channel   = ipuChannels[i];
      size_t   tile_idx         = i * tiles_per_ipu;
      for(size_t j = 0 ; j < tiles_per_ipu ; j ++){
        tile_count[tile_idx + j] = channel_per_tile;
      }
      size_t k = 0 ;
      remain_channel = remain_channel - channel_per_tile * tiles_per_ipu;
      while(remain_channel > 0){
        tile_count[tile_idx + k] += group_size;
        remain_channel           -= group_size;
        k ++;
      }
      for(size_t j = 0 ; j < tiles_per_ipu ; j ++){
        if(0 == tile_count[tile_idx + j])
          break;
        tile_start[tile_idx + j] = channel_ofs;
        channel_ofs             += tile_count[tile_idx + j];
      }
    }
  }

  return { tile_start, tile_count, group_size };
}

SplitChannelInfo  splitTensorByGrainSize(size_t   element_cnt, 
                                         size_t   grain_size, 
                                         size_t   num_tiles,
                                         size_t   tiles_per_ipu){
  size_t               grain_cnt         = (element_cnt + grain_size - 1) / grain_size;
  size_t               ipu_num           = num_tiles / tiles_per_ipu;
  size_t               grains_per_ipu    = ((grain_cnt + (ipu_num - 1)) / ipu_num);
  std::vector<size_t>  ipuGrains(ipu_num, 0);
  std::vector<size_t>  ipuElements(ipu_num, 0);
  size_t               ipuRemainGrains   = grain_cnt;
  size_t               ipuRemainElements = element_cnt;
  std::vector<size_t>  tile_start(num_tiles, 0);
  std::vector<size_t>  tile_count(num_tiles, 0);
  for(size_t i = 0 ; i < ipu_num ; i ++){
    ipuGrains[i]       = ipuRemainGrains   < grains_per_ipu ? ipuRemainGrains : grains_per_ipu;
    ipuElements[i]     = ipuRemainElements < (grains_per_ipu*grain_size) ? ipuRemainElements : grains_per_ipu * grain_size;
    ipuRemainGrains   -= ipuGrains[i];
    ipuRemainElements -= ipuElements[i];
  }

  size_t  element_ofs = 0;
  for(size_t i = 0 ; i < ipu_num ; i ++){
    size_t   grain_per_tile = ((ipuGrains[i] + (tiles_per_ipu - 1)) / tiles_per_ipu);
    size_t   remain_element = ipuElements[i];
    size_t   tile_idx       = i * tiles_per_ipu;
    while(remain_element > 0)
    {
      tile_start[tile_idx] = element_ofs;
      tile_count[tile_idx] = remain_element < (grain_per_tile * grain_size) ? remain_element : (grain_per_tile * grain_size);
      element_ofs     += (grain_per_tile * grain_size);
      remain_element  -= tile_count[tile_idx];
      tile_idx ++;
    }
  }
  return { tile_start, tile_count, grain_size };
}

SplitWorker1DInfo  split_worker_1dvector(int channel_cnt, 
                                         int inner_size, 
                                         int work_num)
{
  std::vector<int>  splits_size(work_num, 0);
  std::vector<int>  splits_ofs(work_num, 0);

  int*  worker_start    = new int[work_num];
  int*  worker_cnt      = new int[work_num];
  int   worker_idx_last = -1;
  memset(worker_start, 0, work_num * sizeof(int));
  memset(worker_cnt, 0, work_num * sizeof(int));
  for (int i = 0; i < channel_cnt; ++i)
  {
    int idx = ((long long)i * (long long)work_num) / ((long long)channel_cnt);
    if(worker_idx_last != idx)
      worker_start[idx] = i;
    worker_cnt[idx] += 1;
    worker_idx_last   = idx;
  }

  int idx = 0;
  for (int i = 0; i < work_num; ++i)
  {
    if(0 == worker_cnt[i])
      continue;

    splits_size[idx] = worker_cnt[i];
    splits_ofs[idx]  = worker_start[i] * inner_size;

    idx ++;
  }

  delete[] worker_cnt;
  delete[] worker_start;

  return { splits_size, splits_ofs };
}

SplitWorkerInfo  split_worker_vector(int channel_cnt, 
                                     int inner_size, 
                                     int work_num)
{
  std::vector<int>  splits_size(work_num, 0);
  std::vector<int>  splits_channel_ofs(work_num, 0);
  std::vector<int>  splits_data_ofs(work_num, 0);

  std::vector<int>  worker_start(work_num, 0);
  std::vector<int>  worker_cnt(work_num, 0);
  int               worker_idx_last = -1;
  for (int i = 0; i < channel_cnt; ++i)
  {
    int idx = ((long long)i * (long long)work_num) / ((long long)channel_cnt);
    if(worker_idx_last != idx)
      worker_start[idx] = i;
    worker_cnt[idx] += 1;
    worker_idx_last  = idx;
  }

  int idx = 0;
  for (int i = 0; i < work_num; ++i)
  {
    if(0 == worker_cnt[i])
      continue;

    splits_size[idx]        = worker_cnt[i];
    splits_channel_ofs[idx] = worker_start[i];
    splits_data_ofs[idx]    = worker_start[i] * inner_size;

    idx ++;
  }

  return { splits_size, splits_channel_ofs, splits_data_ofs };
}