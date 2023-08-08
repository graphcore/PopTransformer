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

import math
import numpy as np
from poptransformer import ops
from poptransformer.layers import BaseLayer
from poptransformer.layers import Linear


class BaseRotaryAttention(BaseLayer):
    softmax_fn_map = {
        "aionnx": ops.softmax,
        "ce": ops.softmax_ce,
    }

    def __init__(self, context, name, input_size, num_head, cache_max_length, layer_index, rotary_dim):
        super().__init__(context, name)
        self.input_size = input_size
        self.num_head = num_head
        self.head_size = self.input_size // self.num_head
        self.cache_max_length = cache_max_length
        self.layer_index = layer_index
        self.rotary_dim = rotary_dim
        self.scale = input_size // num_head
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.c_attn = Linear(self.context, "query_key_value",
                             self.input_size, self.input_size * 3)
        self.c_proj = Linear(self.context, "dense",
                             self.input_size, self.input_size)

    def kv_cache(self, graph, kv, layer_past, sequence_length):
        """Implement cache for key-value pairs without using custom op.

        Equivalent to `ops.kv_cache`.
        """
        with graph.nameScope("attn_past_update"):
            # layer_past: [2, B, N, L, h]
            layer_past = ops.static_slice(
                graph, layer_past, [0], [-sequence_length], [3]
            )
            layer_past = ops.concat(graph, kv, layer_past, 3)

        return layer_past

    def forward_qkv(self, graph, x, layer_past, step, position_ids, block_position_ids):
        qkv = self.c_attn(graph, x)
        qkv = ops.reshape(
            graph,
            qkv,
            [self.batch_size, self.sequence_length,
                self.num_head, 3 * self.head_size],
        )

        temp_splits = [self.head_size] * 3
        q, k, v = ops.split(graph, qkv, num_outputs=3,
                            axis=3, splits=temp_splits)

        cos1, sin1 = self.fixed_pos_embedding(graph, position_ids)
        cos2, sin2 = self.fixed_pos_embedding(graph, block_position_ids)

        q1, q2 = ops.split(
            graph, q, num_outputs=2, axis=-1, splits=[self.rotary_dim, self.rotary_dim], name="rope_split_q",)
        k1, k2 = ops.split(
            graph, k, num_outputs=2, axis=-1, splits=[self.rotary_dim, self.rotary_dim], name="rope_split_k")

        q1, k1 = self.apply_rotary_pos_emb_index(graph, q1, k1, cos1, sin1)
        q2, k2 = self.apply_rotary_pos_emb_index(graph, q2, k2, cos2, sin2)
        q = ops.concat(graph, q1, q2, axis=-1)
        k = ops.concat(graph, k1, k2, axis=-1)
        q = ops.transpose(graph, q, [0, 2, 1, 3])  # q: [B, N, L, h]
        kv = ops.concat(graph, k, v, axis=0)

        kv = ops.reshape(
            graph,
            kv,
            shape=[2, self.batch_size, self.sequence_length,
                   self.num_head, self.head_size]
        )
        kv = ops.transpose(graph, kv, perm=[0, 1, 3, 2, 4])
        layer_present = self.kv_cache(
            graph, kv, layer_past, self.sequence_length)
        layer_present = ops.remap_tensor(graph, layer_present)
        if self.sequence_length != 1 and self.sequence_length < self.cache_max_length:
            layer_present_temp = kv
        else:
            layer_present_temp = layer_present
        k, v = ops.split(
            graph, layer_present_temp, 2, axis=0, splits=[1, 1], name="split_past"
        )
        k = ops.squeeze(graph, k, [0])
        v = ops.squeeze(graph, v, [0])
        return q, k, v, layer_present

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        temp_k = ops.transpose(graph, k, [0, 1, 3, 2])
        layer_scale_coeff_value = np.array(
            [self.layer_index + 1], dtype=self.np_float_type)
        layer_scale_coeff = ops.constant(
            graph, layer_scale_coeff_value, f"softmax_scale_{self.layer_index}"
        )
        attention_scale_value = np.array(
            [1.0 / (math.sqrt(self.head_size) * (self.layer_index + 1))],
            dtype=self.np_float_type,
        )
        attention_scale = ops.constant(
            graph, attention_scale_value, f"attention_scale_{self.layer_index}"
        )
        q = ops.mul(graph, q, attention_scale)
        score = ops.matmul(graph, q, temp_k)
        if self.sequence_length != 1:
            score = ops.remap_tensor(graph, score, fwd_after_matmul=True)
        score = ops.mul(graph, score, layer_scale_coeff)
        score = ops.add(graph, score, attention_mask)
        softmax_fn = self.softmax_fn_map.get(softmax_type, None)
        if not softmax_fn:
            raise ValueError(
                f"Invalid softmax_fn {softmax_type}, options: {self.softmax_fn_map.keys()}"
            )
        score = softmax_fn(
            graph, score, -1, stable_mode=self.sequence_length != 1)
        return score

    def forward_output(self, graph, score, v):
        score = ops.matmul(graph, score, v)
        score = ops.transpose(graph, score, [0, 2, 1, 3])
        score = ops.reshape(
            graph, score, [self.batch_size, self.sequence_length, -1]
        )
        output = self.c_proj(graph, score)
        return output

    def fixed_pos_embedding(self, graph, position_id):
        # rotary_dim = hidden_size_per_head // 2
        # cos, sin = [L, rotary_dim]
        inv_freq_value = np.array(
            [
                1.0 / (10000 ** (i / self.rotary_dim))
                for i in range(0, self.rotary_dim, 2)
            ]
        ).astype(self.np_float_type)
        inv_freq = ops.constant(graph, inv_freq_value, "inv_freq")
        inv_freq = ops.reshape(graph, inv_freq, [1, -1])
        # position_id -> [L, B]
        position_id = ops.cast(
            graph, ops.reshape(graph, position_id,
                               [-1, 1]), self.popart_float_type
        )
        freqs = ops.matmul(graph, position_id, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.concat(graph, freqs, freqs, axis=-1)
        # emb -> [L, rotary_dim] -> [1, L, 1, rotary_dim]
        emb = ops.reshape(graph, emb, shape=[1, -1, 1, self.rotary_dim])

        cos, sin = graph.aiOnnx.cos([emb]), graph.aiOnnx.sin([emb])
        return cos, sin

    def rotate_half(self, graph, x):
        x1, x2 = ops.split(
            graph,
            x,
            num_outputs=2,
            axis=-1,
            splits=[self.rotary_dim // 2, self.rotary_dim // 2],
            name="rope_split",
        )
        x2 = ops.mul(graph, x2, ops.constant(
            graph, np.array([-1]).astype(self.np_float_type)))
        return ops.concat(graph, x2, x1, axis=-1)

    def apply_rotary_pos_emb_index(self, graph, q, k, cos, sin):
        # position_id: [B, L], q, k: [B, L, N, rotary_dim], sin, cos: [L, rotary_dim] -> [1, L, 1, rotary_dim]
        q = ops.add(
            graph,
            ops.mul(graph, q, cos),
            ops.mul(graph, self.rotate_half(graph, q), sin)
        )
        k = ops.add(
            graph,
            ops.mul(graph, k, cos),
            ops.mul(graph, self.rotate_half(graph, k), sin),
        )
        return q, k

    def __call__(
        self,
        graph,
        x,
        layer_past,
        position_ids,
        block_position_ids,
        step,
        attention_mask,
        sequence_length,
        softmax_type="ce"
    ):
        with graph.nameScope(self.context):
            self.sequence_length = sequence_length
            q, k, v, layer_present = self.forward_qkv(
                graph, x, layer_past, step, position_ids, block_position_ids
            )
            score = self.forward_attention(
                graph, q, k, attention_mask, softmax_type)
            output = self.forward_output(graph, score, v)
            return output, layer_present


class TPRotaryAttention(BaseRotaryAttention):
    def collect_bind_layer_weights(self):
        self.num_head_before_tp = self.num_head
        self.num_head = self.num_head // self.num_replicas
        assert self.num_head_before_tp == self.num_head * self.num_replicas, \
            f"Heads {self.num_head_before_tp} can not be exact divided by replicas {self.num_replicas}."

        qkv_tp_setting = {
            'strategy_name': 'start',
        }
        proj_tp_setting = {
            'strategy_name': 'end',
        }
        self.c_attn = Linear(
            self.context, "query_key_value", self.input_size, self.input_size * 3, **qkv_tp_setting)
        self.c_proj = Linear(
            self.context, "dense", self.input_size, self.input_size, **proj_tp_setting)


class RotaryAttention(TPRotaryAttention, BaseRotaryAttention):
    layer_class_map = {
        "tp": TPRotaryAttention,
        "shard": BaseRotaryAttention,
    }

    def __init__(self, context, name, input_size, num_head, cache_max_length, layer_index, rotary_dim):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(
            f"initializing model type: {self.layer_class.__name__}")
        super().__init__(context, name, input_size, num_head, cache_max_length, layer_index, rotary_dim)

    def __call__(
        self,
        graph,
        x,
        layer_past,
        position_ids,
        block_position_ids,
        step,
        attention_mask,
        sequence_length,
        softmax_type="ce"
    ):
        return self.layer_class.__call__(
            self, graph, x, layer_past, position_ids, block_position_ids,
            step, attention_mask, sequence_length, softmax_type,)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        return self.layer_class.forward_attention(self, graph, q, k, attention_mask, softmax_type)

    def forward_qkv(self, graph, x, layer_past, step, position_ids, block_position_ids):
        return self.layer_class.forward_qkv(
            self, graph, x, layer_past, step, position_ids, block_position_ids
        )

    def forward_output(self, graph, score, v):
        return self.layer_class.forward_output(self, graph, score, v)

    def fixed_pos_embedding(self, graph, position_id):
        return self.layer_class.fixed_pos_embedding(self, graph, position_id)

    def rotate_half(self, graph, x):
        return self.layer_class.rotate_half(self, graph, x)

    def apply_rotary_pos_emb_index(self, graph, q, k, cos, sin):
        return self.layer_class.apply_rotary_pos_emb_index(self, graph, q, k, cos, sin)
