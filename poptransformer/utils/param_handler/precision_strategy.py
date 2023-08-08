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

import numpy as np
from poprt.utils import convert_float_to_uint8
from poptransformer import ops
from ..registry import REGISTRY
from .tensor_parallel_strategy import shard, repeat


def weight_fn_identity(host_layer, weight_np, weight_key, weight_fn_tp, num_replicas, weight_axis, **vs_setting):
    return weight_np

def weight_fn_int4(host_layer, weight_np, weight_key, weight_fn_tp, num_replicas, weight_axis, **vs_setting):
    if weight_np.dtype == np.int8:  # Embedding/LM are FP16 precision
        if len(weight_np.shape) == 3:   # TP:[num_replicas, shape1, shape2]
            weight_np = weight_np.transpose(0,2,1)
        elif len(weight_np.shape) == 2:
            weight_np = weight_np.transpose(1, 0)
        else:
            raise ValueError(f"weight_np can only have rank 2 or 3, but got {len(weight_np.shape)}.")
        scale_key = weight_key + '_scale'
        scale_np = host_layer.get_param_from_state_dict(scale_key, shape_list=(weight_np.shape[1],))
        if num_replicas > 1 and len(weight_np.shape)==3:
            if weight_axis == 0:
                scale_np = repeat(scale_np, num_replicas, 0)
            elif weight_axis in [-1, 1]:
                scale_np = shard(scale_np, num_replicas, 0)
            else:
                raise ValueError(f"weight_axis can only be 0,1,-1, but got {weight_axis}.")
        host_layer.add_initialized_input_tensor(scale_np, scale_key, **vs_setting)
    return weight_np

def weight_fn_fp8(host_layer, weight_np, weight_key, weight_fn_tp, num_replicas, weight_axis, **vs_setting):
    scale_key = weight_key + '_scale'
    scale_np = np.array([-1]).astype(np.int32)
    if num_replicas > 1:
        scale_np = np.repeat(np.expand_dims(scale_np, 0), num_replicas, axis=0)
    host_layer.add_initialized_input_tensor(scale_np, scale_key, **vs_setting)
    weight_np = convert_float_to_uint8(weight_np.astype(np.float32), 'F143', -1)
    return weight_np

def prepare_float32_16_matmul(graph, x, weight):
    return x, weight

def prepare_int4_matmul(graph, x, weight):
    scale = weight + '_scale'
    if scale in REGISTRY.get('main_graph').getInputTensorIds():
        weight = ops.int4_to_half(graph, weight, scale, x, axis=1)
    return x, weight

def prepare_fp8_matmul(graph, x, weight):
    scale = weight + '_scale'
    if scale in REGISTRY.get('main_graph').getInputTensorIds():
        x = ops.half_to_uint8(graph, x, scale)
    return x, weight

def prepare_fp8_weight_matmul(graph, x, weight):
    return x, weight

def matmul_identity(graph, x, weight):
    return ops.matmul(graph, x, weight)

def matmul_int4(graph, x, weight):
    return matmul_identity(graph, x, weight)

def matmul_fp8(graph, x, weight):
    scale = weight + '_scale'
    if scale in REGISTRY.get('main_graph').getInputTensorIds():
        return ops.fp8_matmul(graph, x, weight, scale, scale, 'F143', 'F143')
    return ops.matmul(graph, x, weight)

def post_process_float32_16_matmul(graph, y):
    return y

def post_process_int4_matmul(graph, y):
    return y

def post_process_fp8_matmul(graph, y):
    return y


PrecisionStrategyMap = {
    'fp16': {
        'weight_fn': weight_fn_identity,
        'prepare_matmul': prepare_float32_16_matmul,
        'matmul_fn': matmul_identity,
        'post_process_matmul': post_process_float32_16_matmul},
    'fp32': {
        'weight_fn': weight_fn_identity,
        'prepare_matmul': prepare_float32_16_matmul,
        'matmul_fn': matmul_identity,
        'post_process_matmul': post_process_float32_16_matmul},
    'int4': {
        'weight_fn': weight_fn_int4,
        'prepare_matmul': prepare_int4_matmul,
        'matmul_fn': matmul_int4,
        'post_process_matmul': post_process_int4_matmul},
    'fp8': {
        'weight_fn': weight_fn_fp8,
        'prepare_matmul': prepare_fp8_matmul,
        'matmul_fn': matmul_fp8,
        'post_process_matmul': post_process_fp8_matmul},
    'fp8_weight': {
        'weight_fn': weight_fn_fp8,
        'prepare_matmul': prepare_fp8_weight_matmul,
        'matmul_fn': matmul_fp8,
        'post_process_matmul': post_process_fp8_matmul}
}
