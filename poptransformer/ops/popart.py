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
from poptransformer.ops.customized import int4_to_half
from poptransformer.utils.registry import REGISTRY


def pad(graph, x, pad_list, mode='constant', constant_value=0.0):
    return graph.aiOnnx.pad([x], pads=pad_list, mode=mode, value=constant_value)

def randomnormal(graph, shape, dtype, scale):
    return graph.aiOnnx.randomnormal(shape=shape, dtype=dtype, scale=scale)

def random_sample(graph, logits, k, dim=1):
    _, next_token = topk(graph, logits, axis=dim, k=k)
    rand = randomnormal(graph, [1, k], 1, 1)
    rand = argmax(graph, rand, 1)
    return dynamic_slice(graph, next_token, rand, axes=[dim], sizes=[1])

def add(graph, x, y):
    return graph.aiOnnx.add([x, y])

def less(graph, x, y):
    return graph.aiOnnx.less([x, y])

def greater(graph, x, y):
    return graph.aiOnnx.greater([x, y])

def sum_op(graph, inputs):
    return graph.aiOnnx.sum(inputs)

def reduce_sum(graph, x, axes, keepdims):
    return graph.aiOnnx.reducesum([x], axes=axes, keepdims=keepdims)

def bitwiseor(graph, x, y):
    return graph.aiGraphcore.bitwiseor([x, y])

def sub(graph, x, y):
    return graph.aiOnnx.sub([x, y])

def div(graph, x, y):
    return graph.aiOnnx.div([x, y])

def matmul(graph, x, y):
    o = graph.aiOnnx.matmul([x, y])
    amp = REGISTRY.get('amp')
    partialtype = REGISTRY.get('partialtype')
    serial_factor = REGISTRY.get('serial_factor')
    serial_mode = REGISTRY.get('serial_mode')
    if amp is not None:
        graph.setAvailableMemoryProportion(o, amp)
    if partialtype is not None:
        graph.setPartialsType(o, partialtype)
    if serial_factor is not None:
        graph.setSerializeMatMul({o}, mode=serial_mode, factor=serial_factor)
    return o

def constant(graph, tensor, tensor_name='constant'):
    return graph.aiOnnx.constant(tensor, debugContext=tensor_name)

def _group_norm(graph, x, gamma, beta, eps):
    return graph.aiGraphcore.groupnormalization([x, gamma, beta], 1, eps)[0]

def group_norm(graph, x, gamma, beta, eps, batch_size, sequence_length, input_size):
    x = reshape(graph, x, [batch_size * sequence_length, input_size])
    x = _group_norm(graph, x, gamma, beta, eps)
    x = reshape(graph, x, [batch_size, sequence_length, input_size])
    return x

def gather(graph, weight, input_ids):
    return graph.aiOnnx.gather([weight, input_ids])

def grouped_gather(graph, weight, indexs, axis=0, group_size=1):
    return graph.aiGraphcore.groupedgather([weight, indexs], axis=axis, group_size=group_size)

def mul(graph, x, y):
    return graph.aiOnnx.mul([x, y])

def tanh(graph, x):
    return graph.aiOnnx.tanh([x])

def split(graph, x, num_outputs, axis, splits, name='split'):
    return graph.aiOnnx.split([x], num_outputs=num_outputs, axis=axis, split=splits, debugContext=name)

def transpose(graph, x, perm):
    return graph.aiOnnx.transpose([x], perm=perm)

def reshape(graph, x, shape):
    shape = constant(graph, np.asarray(shape, dtype=np.int32))
    return graph.aiOnnx.reshape([x, shape])

def static_slice(graph, x, starts, ends, axes):
    return graph.aiGraphcore.slice([x], starts=starts, ends=ends, axes=axes)

def dynamic_slice(graph, x, index, axes, sizes):
    return graph.aiGraphcore.dynamicslice([x, index], axes=axes, sizes=sizes)

def dynamic_update(graph, x, index, slice_tensor, axes, sizes):
    return graph.aiGraphcore.dynamicupdate([x, index, slice_tensor], axes=axes, sizes=sizes)

def cast(graph, x, popart_float_type):
    return graph.aiOnnx.cast([x], popart_float_type)

def unsqueeze(graph, x, dim_list):
    return graph.aiOnnx.unsqueeze([x], dim_list)

def squeeze(graph, x, dim_list):
    return graph.aiOnnx.squeeze([x], dim_list)

def equal(graph, x, y):
    return graph.aiOnnx.equal([x, y])

def where(graph, cond, x1, x2):
    return graph.aiOnnx.where([cond, x1, x2])

def softmax(graph, x, dim, stable_mode=None):
    if stable_mode:
        raise ValueError('set use unstable softmax in session, or use type ce ')
    return graph.aiOnnx.softmax([x], dim)

def argmax(graph, x, axis, keepdims=True):
    return graph.aiOnnx.argmax([x], axis=axis, keepdims=keepdims)

def topk(graph, x, axis, k):
    k = constant(graph, np.array(k).astype(np.int64), 'k')
    return graph.aiOnnx.topk([x, k], axis=axis)

def printtensor(graph, x, title):
    return graph.aiGraphcore.printtensor([x], title=title)

def gelu(graph, x):
    return graph.aiGraphcore.gelu([x])

def concat(graph, x, y, axis):
    return graph.aiOnnx.concat([x, y], axis=axis)

def concat_sequence(graph, inputs, axis):
    return graph.aiOnnx.concat(inputs, axis=axis)

def log_softmax(graph, x, axis=-1):
    return graph.aiOnnx.logsoftmax([x], axis)

def clip(graph, x, clip_min=10, clip_max=100):
    output = graph.aiOnnx.clip([x], max=clip_max, min=clip_min)
    return output

def expand(graph, x, shape_list):
    expand_shape = constant(graph, np.array(
        shape_list, dtype=np.int32), 'shape')
    return graph.aiOnnx.expand([x, expand_shape])

def bitwiseand(graph, x, y):
    return graph.aiGraphcore.bitwiseand([x, y])

def tile(graph, x, repeats):
    repeats = constant(graph, np.array(repeats, dtype=np.int64), 'repeats')
    return graph.aiOnnx.tile([x, repeats], 'tile')

def loop(graph, max_loop_num, loop_graph_input_list, loop_graph):
    max_loop = constant(graph, np.array(max_loop_num).astype(np.int32), 'max_loop')
    init_cond = constant(graph, np.array(True).astype(np.bool_), 'cond')
    loop_outputs = graph.aiOnnx.loop(
        [max_loop, init_cond] + loop_graph_input_list,
        len(loop_graph_input_list),
        loop_graph,
    )
    return loop_outputs

def call_sub_graph(graph, input_list, sub_graph):
    return graph.aiGraphcore.call(input_list, 1, sub_graph)

def replicated_all_reduce(graph, x):
    return graph.aiGraphcore.replicatedallreduce([x])

def conv(graph, x, weight, bias, kernel_shape, strides, pads, dilations, group):
    args = [x, weight] if bias is None else [x, weight, bias]
    o = graph.aiOnnx.conv(
        args=args,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group
    )
    amp = REGISTRY.get('amp')
    partialtype = REGISTRY.get('partialtype')
    if amp is not None:
        graph.setAvailableMemoryProportion(o, amp)
    if partialtype is not None:
        graph.setPartialsType(o, partialtype)
    return o

def relu(graph, x):
    return graph.aiOnnx.relu([x])

def sigmoid(graph, x):
    return graph.aiOnnx.sigmoid([x])

def exp(graph, x):
    return graph.aiOnnx.exp([x])

def maximum(graph, x, y):
    cond = greater(graph, x, y)
    output = where(graph, cond, x, y)
    return output

def swish(graph, x):
    return graph.aiGraphcore.swish([x])

def mean(graph, x):
    return graph.aiOnnx.mean([x])

def sqrt(graph, x):
    return graph.aiOnnx.sqrt([x])

def reciprocal(graph, x):
    return graph.aiOnnx.reciprocal([x])

def reducemean(graph, x):
    return graph.aiOnnx.reducemean([x],axes=[-1])
