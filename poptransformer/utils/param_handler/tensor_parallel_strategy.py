# Copyright (c) 2023 Graphcore Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil
import numpy as np

from poptransformer import ops
from poptransformer.utils.registry import REGISTRY


def shard(param: np.ndarray, num_replicas: int, axis: int) -> np.array:
    """Shard array along a given axis"""
    if axis < 0:
        axis = len(param.shape) + axis
    if param.shape[axis] % num_replicas != 0:
        pads = [(0, 0) for i in range(len(param.shape))]
        pads[axis] = (0, num_replicas - param.shape[axis] % num_replicas)
        param = np.pad(param, pads)

    return np.ascontiguousarray(np.concatenate(np.split(param[np.newaxis, ...], num_replicas, axis=axis + 1)))

def repeat(param: np.ndarray, num_replicas: int, axis: int = 0) -> np.array:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(param, axis), num_replicas, axis=axis)

def build_sharded_weight(param, num_replicas, vocab_size, embedding_size):
    shard_size = ceil(vocab_size / num_replicas)
    n_pad = shard_size * num_replicas - param.shape[0]
    param = np.pad(param, ((0, n_pad), (0, 0)))
    param = param.reshape(num_replicas, shard_size, embedding_size)
    param = np.pad(param, ((0, 0), (0, 1), (0, 0)))
    return param

def shard_fused_qkv(param, num_replicas, axis):
    q_split, k_split, v_split = [
        np.split(part, num_replicas, axis=axis) for part in np.split(param, 3, axis=axis)
    ]
    sharded_param = np.concatenate(
        [np.concatenate([q_split[i], k_split[i], v_split[i]], axis=axis)[np.newaxis, ...]
            for i in range(num_replicas)]
    )
    sharded_param = np.ascontiguousarray(sharded_param)
    return sharded_param

def shard_multi_query_qkv(param, n_shards, axis):
    q_size = REGISTRY.get("query_size")
    kv_size = REGISTRY.get("key_value_size")
    q_split, k_split, v_split = [
        np.split(part, n_shards, axis=axis) for part in np.split(param, [q_size, q_size+kv_size], axis=axis)
    ]
    sharded_param = np.concatenate(
        [np.concatenate([q_split[i], k_split[i], v_split[i]], axis=axis)[np.newaxis, ...]
            for i in range(n_shards)]
    )
    sharded_param = np.ascontiguousarray(sharded_param)
    return sharded_param

def identity(param, num_replicas, axis):
    return param

def identity_prepare_matmul(graph, x, weight):
    return x, weight

def identity_post_process_matmul(graph, y):
    return y


TPStragetgyMap = {
    'start': {
        'weight_fn': shard,
        'weight_axis': -1,
        'bias_fn': shard,
        'prepare_matmul': identity_prepare_matmul,
        'post_process_matmul': identity_post_process_matmul},
    'end': {
        'weight_fn': shard,
        'weight_axis': 0,
        'bias_fn': repeat,
        'prepare_matmul': identity_prepare_matmul,
        'post_process_matmul': ops.replicated_all_reduce},
    'fused_qkv': {
        'weight_fn': shard_fused_qkv,
        'weight_axis': -1,
        'bias_fn': shard_fused_qkv,
        'prepare_matmul': identity_prepare_matmul,
        'post_process_matmul': identity_post_process_matmul},
    'multi_query_qkv': {
        'weight_fn': shard_multi_query_qkv,
        'weight_axis': -1,
        'bias_fn': shard_multi_query_qkv,
        'prepare_matmul': identity_prepare_matmul,
        'post_process_matmul': identity_post_process_matmul},
    'identity': {
        'weight_fn': identity,
        'weight_axis': 0,
        'bias_fn': identity,
        'prepare_matmul': identity_prepare_matmul,
        'post_process_matmul': identity_post_process_matmul},
}
