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
from poptransformer.utils import shard, repeat, shard_fused_qkv
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
        self.sharded_embd = self.input_size // self.num_replicas
        self.sharded_head = self.num_head // self.num_replicas
        vs_setting = {'vs_type': 'consecutive', 'group_size': 1}

        c_attn_weight_key = '.'.join([self.context, 'c_attn.weight'])
        c_attn_weight_np = self.get_param_from_state_dict(c_attn_weight_key, [self.input_size, self.input_size * 3])
        c_attn_bias_key = '.'.join([self.context, 'c_attn.bias'])
        c_attn_bias_np = self.get_param_from_state_dict(c_attn_bias_key, [self.input_size * 3])
        c_attn_weight_np = shard_fused_qkv(c_attn_weight_np, self.num_replicas)
        c_attn_bias_np = shard_fused_qkv(c_attn_bias_np, self.num_replicas)
        self.c_attn_weight_id = self.add_initialized_input_tensor(c_attn_weight_np, c_attn_weight_key, **vs_setting)
        self.c_attn_bias_id = self.add_initialized_input_tensor(c_attn_bias_np, c_attn_bias_key, **vs_setting)

        c_proj_weight_key = '.'.join([self.context, 'c_proj.weight'])
        c_proj_weight_np = self.get_param_from_state_dict(c_proj_weight_key, [self.input_size, self.input_size])
        c_proj_bias_key = '.'.join([self.context, 'c_proj.bias'])
        c_proj_bias_np = self.get_param_from_state_dict(c_proj_bias_key, [self.input_size])
        c_proj_weight_np = shard(c_proj_weight_np, self.num_replicas, axis=0)
        c_proj_bias_np = repeat(c_proj_bias_np, self.num_replicas, axis=0)
        self.c_proj_weight_id = self.add_initialized_input_tensor(c_proj_weight_np, c_proj_weight_key, **vs_setting)
        self.c_proj_bias_id = self.add_initialized_input_tensor(c_proj_bias_np, c_proj_bias_key, **vs_setting)

    def forward_qkv(self, graph, x, step):
        x = ops.reshape(graph, x, [self.batch_size * self.sequence_length, self.input_size])
        qkv = ops.matmul(graph, x, self.c_attn_weight_id)
        qkv = ops.add(graph, qkv, self.c_attn_bias_id)
        q, k, v = ops.split(
            graph, qkv, num_outputs=3, axis=-1, splits=[self.sharded_embd] * 3, name='split_qkv')

        q = ops.reshape(graph, q, [self.batch_size, self.sequence_length, self.sharded_head, self.head_size])
        q = ops.transpose(graph, q, [0, 2, 1, 3])  # q: [B, N, L, h]
        v = ops.reshape(graph, v, [self.batch_size, self.sequence_length, self.sharded_head, self.head_size])
        v = ops.transpose(graph, v, [0, 2, 1, 3])  # v: [B, N, L, h]
        k = ops.reshape(graph, k, [self.batch_size, self.sequence_length, self.sharded_head, self.head_size])
        k = ops.transpose(graph, k, [0, 2, 3, 1])  # k: [B, N, h, L]

        if self.sequence_length == 1:
            k = ops.kv_cache(graph, step, k, self.cache_max_length, 3, self.sequence_length)
            v = ops.kv_cache(graph, step, v, self.cache_max_length, 2, self.sequence_length)
            k = ops.remap_tensor(graph, k)
            v = ops.remap_tensor(graph, v)
        return q, k, v

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        score = ops.matmul(graph, q, k)
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
        score = ops.reshape(graph, score, [self.batch_size, self.sequence_length, self.sharded_embd])
        output = ops.matmul(graph, score, self.c_proj_weight_id)
        output = graph.aiGraphcore.replicatedallreduce([output])
        output = ops.add(graph, output, self.c_proj_bias_id)
        return output


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
