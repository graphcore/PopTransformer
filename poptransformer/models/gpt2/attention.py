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

from poptransformer import ops
from poptransformer.layers import BaseLayer
from poptransformer.layers import Linear


class BaseAttention(BaseLayer):
    softmax_fn_map = {
        'aionnx': ops.softmax,
        'ce': ops.softmax_ce,
    }
    def __init__(self, context, name, input_size, num_head, cache_max_length):
        super().__init__(context, name)
        self.input_size = input_size
        self.num_head = num_head
        self.head_size = self.input_size // self.num_head
        self.cache_max_length = cache_max_length
        self.scale = input_size // num_head
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.c_attn = Linear(self.context, 'c_attn', self.input_size, self.input_size * 3)
        self.c_proj = Linear(self.context, 'c_proj', self.input_size, self.input_size)

    def forward_qkv(self, graph, x, step):
        qkv = self.c_attn(graph, x)
        temp_splits = [self.input_size, self.input_size * 2]
        q, kv = ops.split(graph, qkv, num_outputs=2, axis=2, splits=temp_splits)

        temp_q_shape = [self.batch_size, self.sequence_length, self.num_head, self.head_size]
        q = ops.reshape(graph, q, temp_q_shape)

        temp_kv_shape = [self.batch_size, self.sequence_length, 2, self.num_head, self.head_size]
        kv = ops.reshape(graph, kv, temp_kv_shape)

        q = ops.transpose(graph, q, [0, 2, 1, 3])
        kv = ops.transpose(graph, kv, [2, 0, 3, 1, 4])
        layer_past = ops.kv_cache(graph, step, kv, self.cache_max_length, 3, self.sequence_length)
        layer_past = ops.remap_tensor(graph, layer_past)
        if self.sequence_length != 1 and self.sequence_length < self.cache_max_length:
            layer_past = ops.static_slice(graph, layer_past, [0], [self.sequence_length], [3])
        k, v = ops.split(graph, layer_past, 2, axis=0, splits=[1, 1], name='split_past')
        k = ops.squeeze(graph, k, [0])
        v = ops.squeeze(graph, v, [0])
        return q, k, v

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        temp_k = ops.transpose(graph, k, [0, 1, 3, 2])
        score = ops.matmul(graph, q, temp_k)
        if self.sequence_length != 1:
            score = ops.remap_tensor(graph, score, fwd_after_matmul=True)

        score = ops.add(graph, score, attention_mask)
        softmax_fn = self.softmax_fn_map.get(softmax_type, None)
        if not softmax_fn:
            raise ValueError(f"Invalid softmax_fn {softmax_type}, options: {self.softmax_fn_map.keys()}")
        score = softmax_fn(graph, score, -1, stable_mode=self.sequence_length != 1)
        return score

    def forward_output(self, graph, score, v):
        score = ops.matmul(graph, score, v)
        score = ops.transpose(graph, score, [0, 2, 1, 3])
        score = ops.reshape(graph, score, [self.batch_size, self.sequence_length, self.input_size])
        output = self.c_proj(graph, score)
        return output

    def __call__(self, graph, x, step, attention_mask, sequence_length, softmax_type='ce'):
        with graph.nameScope(self.context):
            self.sequence_length = sequence_length
            q, k, v = self.forward_qkv(graph, x, step)
            score = self.forward_attention(graph, q, k, attention_mask, softmax_type)
            output = self.forward_output(graph, score, v)
            return output


class TPAttention(BaseAttention):

    def collect_bind_layer_weights(self):
        qkv_tp_setting = {'strategy_name': 'fused_qkv'}
        self.c_attn = Linear(self.context, 'c_attn', self.input_size, self.input_size * 3, **qkv_tp_setting)
        proj_tp_setting = {'strategy_name': 'end'}
        self.c_proj = Linear(self.context, 'c_proj', self.input_size, self.input_size, **proj_tp_setting)
        self.input_size = self.input_size // self.num_replicas
        self.num_head = self.num_head // self.num_replicas


class Attention(TPAttention, BaseAttention):
    layer_class_map = {
        'tp': TPAttention,
        'shard': BaseAttention}

    def __init__(self, context, name, input_size, num_head, cache_max_length):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, input_size, num_head, cache_max_length)

    def __call__(self, graph, x, step, attention_mask, sequence_length, softmax_type='ce'):
        return self.layer_class.__call__(self, graph, x, step, attention_mask, sequence_length, softmax_type)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        return self.layer_class.forward_attention(self, graph, q, k, attention_mask, softmax_type)

    def forward_qkv(self, graph, x, step):
        return self.layer_class.forward_qkv(self, graph, x, step)

    def forward_output(self, graph, score, v):
        return self.layer_class.forward_output(self, graph, score, v)
