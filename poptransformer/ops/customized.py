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

import ctypes
import numpy as np
import poprt
so_path = '../../custom_ops.so'
ctypes.cdll.LoadLibrary(so_path)


def kv_cache(graph, step, updater, max_len, sequence_axis=1, step_len=1):
    output = graph.customOp(
        inputs=[step, updater],
        opName="KVCache",
        domain="ai.graphcore",
        opVersion=1,
        numOutputs=1,
        attributes={
            'max_len': max_len,
            'step_len':  step_len,
            'sequence_axis': sequence_axis,
            'fp8': False,
        },
    )[0]
    return output


def remap_tensor(graph, x, fwd_after_matmul=0, name='remap'):
    output = graph.customOp(
        opName='Remap',
        domain='ai.graphcore',
        opVersion=1,
        inputs=[x],
        attributes={
            'fwd_grain_size': 8,
            'bwd_grain_size': 8,
            'fwd_clone_layout': 0,
            'bwd_clone_layout': 0,
            'fwd_after_matmul': fwd_after_matmul,
            'bwd_after_matmul': 0,
            'debug_str': name
        }
    )[0]
    return output


def softmax_ce(graph, x, dim, stable_mode=True):
    output = graph.customOp(
        opName="SoftmaxV3",
        domain="ai.graphcore",
        opVersion=1,
        inputs=[x],
        attributes={'axis': dim, 'stable_mode': stable_mode}
    )[0]
    return output


# TODO: NormCE -> FastNorm
def layer_norm_ce(graph, x, gamma, beta, eps, batch_size, sequence_length, input_size):
    return _layer_norm_ce(graph, x, gamma, beta, batch_size, sequence_length, input_size, eps)


def _layer_norm_ce(
        graph, x, weight, bias, batch_size, sequence_length, input_size, epsilon,
        fwd_grain_size=-1, bwd_grain_size=-1, fwd_after_matmul=0, bwd_after_matmul=0,
        stable_algo=False, name='layer_norm_ce'):
    with graph.nameScope(name):

        data_size = np.prod([batch_size, sequence_length, input_size])
        num_groups = int(data_size // input_size)
        assert input_size & 1 == 0
        assert input_size < 8192
        fwd_grain_size = input_size if fwd_grain_size != -1 else fwd_grain_size
        bwd_grain_size = input_size if bwd_grain_size != -1 else bwd_grain_size

        output = graph.customOp(
            opName="NormCE",
            domain="ai.graphcore",
            opVersion=1,
            inputs=[x, weight, bias],
            attributes={
                'fwd_after_matmul': 1 if fwd_after_matmul else 0,
                'bwd_after_matmul': 1 if bwd_after_matmul else 0,
                'fwd_grain_size': fwd_grain_size,
                'bwd_grain_size': bwd_grain_size,
                'epsilon': epsilon,
                'num_groups': num_groups,
                'stable_algo': stable_algo,
                'debug_str': graph.getNameScope() + 'op'
            }
        )[0]
    return output


def replicated_allgather(graph, matmul_output):
    output = graph.customOp(
        opName="ReplicatedAllGather",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[matmul_output],
        attributes={})[0]  # shape is 1 dim
    return output


def int4_to_half(graph, x, scale, ref, axis=1, remap=1):
    x = graph.customOp(
            inputs=[x, scale, ref],
            opName="Int4ToHalf",
            domain="ai.graphcore",
            opVersion=1,
            attributes={
                "axis": axis,
                "remap": remap},
        )[0]
    return x

def half_to_uint8(graph, x, fp8_scale, fp8_format='F143'):
    output = graph.customOp(
        opName="CastToUint8WithoutShapeInfer",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[x, fp8_scale],
        attributes={"fp8Format": fp8_format},
    )[0]
    return output

def fp8_matmul(graph, input_l, input_r, fp8_scale_lhs, fp8_scale_rhs, fp8_format_lhs='F143', fp8_format_rhs='F143'):
    output = graph.customOp(
        opName="FP8MatMulWithoutShapeInfer",
        opVersion=1,
        domain="ai.graphcore",
        inputs=[input_l, input_r, fp8_scale_lhs, fp8_scale_rhs],
        attributes={
            "fp8FormatLHS": fp8_format_lhs,
            "fp8FormatRHS": fp8_format_rhs,
        },
    )[0]
    return output
