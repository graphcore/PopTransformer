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



def shard(x: np.ndarray, n_shards: int, axis: int) -> np.array:
    """Shard array along a given axis"""
    if axis < 0:
        axis = len(x.shape) + axis
    return np.ascontiguousarray(np.concatenate(np.split(x[np.newaxis, ...], n_shards, axis=axis + 1)))


def repeat(x: np.ndarray, n: int, axis: int = 0) -> np.array:
    """Repeat array along new axis inserted at position `axis`"""
    return np.repeat(np.expand_dims(x, axis), n, axis=axis)


def build_sharded_weight(weight, num_replica, vocab_size, embedding_size):
    shard_size = ceil(vocab_size / num_replica)
    n_pad = shard_size * num_replica - vocab_size
    weight = np.pad(weight, ((0, n_pad), (0, 0)))
    weight = weight.reshape(num_replica, shard_size, embedding_size)
    weight = np.pad(weight, ((0, 0), (0, 1), (0, 0)))
    return weight


def shard_fused_qkv(param, num_replicas):
    q_split, k_split, v_split = [
        np.split(part, num_replicas, axis=-1) for part in np.split(param, 3, axis=-1)
    ]
    sharded_param = np.concatenate(
        [np.concatenate([q_split[i], k_split[i], v_split[i]], axis=-1)[np.newaxis, ...]
            for i in range(num_replicas)]
    )
    sharded_param = np.ascontiguousarray(sharded_param)
    return sharded_param
