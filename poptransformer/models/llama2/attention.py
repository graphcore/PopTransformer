# Copyright (c) 2023 Graphcore Ltd.
# This is a re-implementation of Llama 2 by Graphcore Ltd
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
import numpy as np

from poptransformer import ops
from poptransformer.utils import shard, repeat, shard_fused_qkv
from poptransformer.layers import BaseLayer
from poptransformer.layers import Linear

class BaseAttention(BaseLayer):
    softmax_fn_map = {
        'aionnx': ops.softmax,
        'ce': ops.softmax_ce,
    }
    def __init__(self, context, name, input_size, num_head, cache_max_length, fp8_cache=False):
        super().__init__(context, name)
        self.input_size = input_size
        self.num_head = num_head
        self.head_size = self.input_size // self.num_head
        self.cache_max_length = cache_max_length
        self.fp8_cache = fp8_cache
        self.scale = input_size // num_head
        self.collect_bind_layer_weights()

    def fixed_pos_embedding(self, graph, step, head_dim):
        inv_freq_value = np.array(
            [1.0 / (10000 ** (i / head_dim)) for i in range(0, head_dim, 2)]).astype(self.np_float_type)
        inv_freq = ops.constant(graph, inv_freq_value, 'inv_freq')
        inv_freq = ops.reshape(graph, inv_freq, [1, -1])

        ind = ops.reshape(graph, step, [-1, 1])
        ind = ops.cast(graph, ind, self.popart_float_type)

        sinusoid_inp = ops.matmul(graph, ind, inv_freq)
        return (graph.aiOnnx.sin([sinusoid_inp]), graph.aiOnnx.cos([sinusoid_inp]))

    def rotate_half(self, graph, x, n_head, head_dim, batch_size=1):
        x1, x2 = ops.split(graph, x, 2, 3, [head_dim//2, head_dim//2], "split_rotate_every_two")
        x2 = ops.mul(graph, x2, ops.constant(graph, np.array([-1]).astype(self.np_float_type)))
        x = ops.concat(graph, x2, x1, 3)
        return ops.reshape(graph, x, [batch_size, n_head, 1, head_dim])

    def apply_rotary_pos_emb(self, graph, q, k, sincos,  n_head, head_dim,batch_size=1):
        sin = ops.concat(graph, sincos[0], sincos[0],1)
        sin = ops.reshape(graph, sin, [1, 1, 1, -1])
        cos = ops.concat(graph, sincos[1], sincos[1],1)
        cos = ops.reshape(graph, cos, [1, 1, 1, -1])

        q_rotate_every_two = self.rotate_half(graph, q, n_head, head_dim, batch_size)
        q = ops.add(graph, ops.mul(graph, q, cos), ops.mul(graph, q_rotate_every_two, sin))
        k_rotate_every_two = self.rotate_half(graph, k, n_head, head_dim, batch_size)
        k = ops.add(graph, ops.mul(graph, k, cos), ops.mul(graph, k_rotate_every_two, sin))
        return q, k

    def collect_bind_layer_weights(self):
        self.q_proj = Linear(self.context, 'q_proj', self.input_size, self.input_size, use_bias=False)
        self.k_proj = Linear(self.context, 'k_proj', self.input_size, self.input_size, use_bias=False)
        self.v_proj = Linear(self.context, 'v_proj', self.input_size, self.input_size, use_bias=False)
        self.o_proj = Linear(self.context, 'o_proj', self.input_size, self.input_size, use_bias=False)

    def forward_qkv(self, graph, x, step):
        q = self.q_proj(graph, x)
        k = self.k_proj(graph, x)
        v = self.v_proj(graph, x)

        q = ops.reshape(graph, q, [self.batch_size, self.sequence_length, self.num_head, self.head_size])
        k = ops.reshape(graph, k, [self.batch_size, self.sequence_length, self.num_head, self.head_size])
        v = ops.reshape(graph, v, [self.batch_size, self.sequence_length, self.num_head, self.head_size])

        q = ops.transpose(graph, q, [0, 2, 1, 3])  # q: [B, N, L, H]
        k = ops.transpose(graph, k, [0, 2, 1, 3])  # k: [B, N, L, H]
        v = ops.transpose(graph, v, [0, 2, 1, 3])  # v: [B, N, L, H]

        sincos = self.fixed_pos_embedding(graph, step, self.head_size)
        q,k = self.apply_rotary_pos_emb(graph, q, k, sincos, self.num_head, self.head_size, self.batch_size)

        kv = ops.concat(graph, k, v, 0)  #kv: [2, B, N, L, H]
        kv = ops.reshape( graph, kv, [2, self.batch_size, self.num_head, self.sequence_length, self.head_size])

        # layer_past: [2, B, N, L, h]
        with graph.nameScope('attn_past_update'):
            layer_past = ops.kv_cache(graph, step, kv, self.cache_max_length, 3, self.sequence_length)
            layer_past_key, layer_past_value = ops.split(
                graph, layer_past, 2, axis=0, splits=[1, 1], name='split_past'
            )
            layer_past_key = ops.squeeze(graph, layer_past_key, [0])
            layer_past_value = ops.squeeze(graph, layer_past_value, [0])
            layer_past_key_temp = ops.transpose(
                graph, layer_past_key, [0, 1, 3, 2])

        return  q, layer_past_key_temp, layer_past_value


    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        w = ops.matmul(graph, q, k)
        w = ops.mul(graph, w, ops.constant(graph, np.array([1/math.sqrt(self.head_size)]).astype(self.np_float_type)))
        w = ops.add(graph, w, attention_mask)

        w = ops.cast(graph, w, 'FLOAT')
        softmax_fn = self.softmax_fn_map.get(softmax_type, None)
        if not softmax_fn:
            raise ValueError(f"Invalid softmax_fn {softmax_type}, options: {self.softmax_fn_map.keys()}")
        w = softmax_fn(graph, w, -1, stable_mode=self.sequence_length != 1)
        w = ops.cast(graph, w, self.popart_float_type)
        return w

    def forward_output(self, graph, score, v):
        a = ops.matmul(graph, score, v)
        a = ops.transpose(graph, a, [0, 2, 1, 3])
        a = ops.reshape(graph, a, [self.batch_size, self.sequence_length, -1])
        return self.o_proj(graph, a)

    def __call__(self, graph, x, step, attention_mask, sequence_length, softmax_type='ce'):
        with graph.nameScope(self.context):
            self.sequence_length = sequence_length
            q, k, v = self.forward_qkv(graph, x, step)
            score = self.forward_attention(graph, q, k, attention_mask, softmax_type)
            output = self.forward_output(graph, score, v)
            return output


class TPAttention(BaseAttention):

    def collect_bind_layer_weights(self):
        self.num_head_beforeTP = self.num_head
        self.num_head = math.ceil(self.num_head / self.num_replicas)
        assert self.num_head_beforeTP == self.num_head * self.num_replicas
        qkv_tp_settings = {
            'strategy_name': 'start',
        }
        proj_tp_setting = {
            'strategy_name': 'end',
        }
        self.q_proj = Linear(
            self.context, 'q_proj', self.input_size, self.input_size, use_bias=False, **qkv_tp_settings)
        self.k_proj = Linear(
            self.context, 'k_proj', self.input_size, self.input_size, use_bias=False, **qkv_tp_settings)
        self.v_proj = Linear(
            self.context, 'v_proj', self.input_size, self.input_size, use_bias=False, **qkv_tp_settings)
        self.o_proj = Linear(
            self.context, 'o_proj', self.input_size, self.input_size, use_bias=False, **proj_tp_setting)


class Attention(TPAttention, BaseAttention):
    layer_class_map = {
        'tp': TPAttention,
        'shard': BaseAttention}

    def __init__(self, context, name, input_size, num_head, cache_max_length, fp8_cache=False):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, input_size, num_head, cache_max_length, fp8_cache)

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
