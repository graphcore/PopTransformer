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
from poptransformer.layers.linear import BaseLinear


class BaseMultiQueryAttention(BaseLayer):
    softmax_fn_map = {
        "aionnx": ops.softmax,
        "ce": ops.softmax_ce,
    }

    def __init__(
        self,
        context,
        name,
        hidden_size,
        multi_query_group_num,
        num_attention_heads,
        cache_max_length,
        layer_number,
        add_qkv_bias,
    ):
        super().__init__(context, name)
        self.hidden_size = hidden_size
        self.multi_query_group_num = multi_query_group_num
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = (
            self.hidden_size // self.num_attention_heads
        )
        self.cache_max_length = cache_max_length
        self.layer_number = layer_number
        self.add_qkv_bias = add_qkv_bias

        # 1 stage mode
        self.input_length = 1
        self.rotary_dim = self.hidden_size_per_attention_head

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.query = Linear(
            self.context,
            "query",
            self.hidden_size,
            self.hidden_size,
            use_bias=self.add_qkv_bias,
        )
        self.key_value = Linear(
            self.context,
            "key_value",
            self.hidden_size,
            2 * self.hidden_size_per_attention_head * self.multi_query_group_num,
            use_bias=self.add_qkv_bias,
        )
        self.dense = Linear(
            self.context, "dense", self.hidden_size, self.hidden_size, use_bias=False
        )

    def fixed_pos_embedding(self, graph, position_id, dim):
        # H = 4096, n = 32, h = 128, dim = 128 // 2 = 64
        # cos, sin = [L, dim]
        inv_freq_value = np.array(
            [1.0 / (10000 ** (i / dim)) for i in range(0, dim, 2)]
        ).astype(np.float32)
        inv_freq = ops.constant(graph, inv_freq_value, "inv_freq")
        inv_freq = ops.reshape(graph, inv_freq, [1, -1])
        # position_id -> [L, B]
        position_id = ops.reshape(graph, position_id, [-1, 1])
        # Notice: fp16 precision is not suitable for large integers.
        position_id = ops.cast(graph, position_id, "FLOAT")
        freqs = ops.matmul(graph, position_id, inv_freq)
        emb = ops.concat(graph, freqs, freqs, axis=-1)
        # emb -> [L, dim] -> [1, L, 1, dim]
        emb = ops.reshape(graph, emb, shape=[1, -1, 1, dim])

        emb = ops.cast(graph, emb, self.popart_float_type)
        cos, sin = graph.aiOnnx.cos([emb]), graph.aiOnnx.sin([emb])
        return cos, sin

    def rotate_half(self, graph, x):
        x1, x2 = ops.split(
            graph,
            x,
            num_outputs=2,
            axis=-1,
            splits=[self.rotary_dim // 4, self.rotary_dim // 4],
            name="rope_split",
        )
        x2 = ops.mul(graph, x2, ops.constant(graph, np.array([-1]).astype(np.float32)))
        return ops.concat(graph, x2, x1, axis=-1)

    def apply_rotary_pos_emb(self, graph, x, apply_rotary_pos_emb, num_heads):
        # position_id: [B, L], x: [B, L, N, rotary_dim], sin, cos: [L, rotary_dim] -> [1, L, 1, rotary_dim]
        cos, sin = apply_rotary_pos_emb
        x = ops.reshape(
            graph, x, shape=[self.batch_size, self.input_length, num_heads, -1, 2]
        )
        x = ops.transpose(graph, x, perm=[0, 1, 2, 4, 3])
        x = ops.reshape(
            graph, x, shape=[self.batch_size, self.input_length, num_heads, -1]
        )
        x = ops.add(
            graph,
            ops.mul(graph, x, cos),
            ops.mul(graph, self.rotate_half(graph, x), sin),
        )
        x = ops.reshape(
            graph, x, shape=[self.batch_size, self.input_length, num_heads, 2, -1]
        )
        x = ops.transpose(graph, x, perm=[0, 1, 2, 4, 3])
        x = ops.reshape(
            graph, x, shape=[self.batch_size, self.input_length, num_heads, -1]
        )
        return x

    def rotary_embedding(self, graph, q, k, position_ids):
        with graph.nameScope("build_rotary"):
            rotary_pos_emb = self.fixed_pos_embedding(
                graph, position_ids, dim=self.rotary_dim // 2
            )
            q1, q2 = ops.split(
                graph,
                q,
                num_outputs=2,
                axis=-1,
                splits=[self.rotary_dim // 2, self.rotary_dim // 2],
                name="rope_split_q",
            )
            k1, k2 = ops.split(
                graph,
                k,
                num_outputs=2,
                axis=-1,
                splits=[self.rotary_dim // 2, self.rotary_dim // 2],
                name="rope_split_k",
            )
        with graph.nameScope("apply_rotary"):
            q1 = self.apply_rotary_pos_emb(
                graph, q1, rotary_pos_emb, self.num_attention_heads
            )
            k1 = self.apply_rotary_pos_emb(
                graph, k1, rotary_pos_emb, self.multi_query_group_num
            )
            q = ops.concat(graph, q1, q2, axis=-1)
            k = ops.concat(graph, k1, k2, axis=-1)
        return q, k

    def forward_qkv(self, graph, x, step):
        q = self.query(graph, x)
        mixed_kv = self.key_value(graph, x)
        k, v = ops.split(
            graph,
            mixed_kv,
            num_outputs=2,
            axis=-1,
            splits=[
                self.multi_query_group_num * self.hidden_size_per_attention_head,
                self.multi_query_group_num * self.hidden_size_per_attention_head,
            ],
        )
        q = ops.reshape(
            graph,
            q,
            shape=[
                self.batch_size,
                self.sequence_length,
                self.num_attention_heads,
                self.hidden_size_per_attention_head,
            ],
        )
        k = ops.reshape(
            graph,
            k,
            shape=[
                self.batch_size,
                self.sequence_length,
                self.multi_query_group_num,
                self.hidden_size_per_attention_head,
            ],
        )
        v = ops.reshape(
            graph,
            v,
            shape=[
                self.batch_size,
                self.sequence_length,
                self.multi_query_group_num,
                self.hidden_size_per_attention_head,
            ],
        )

        q, k = self.rotary_embedding(graph, q, k, position_ids=step)

        # q = [B, N, L, h]
        q = ops.transpose(graph, q, perm=[0, 2, 1, 3])
        kv = ops.concat(graph, k, v, 0)
        kv = ops.reshape(
            graph,
            kv,
            [
                2,
                self.batch_size,
                self.sequence_length,
                self.multi_query_group_num,
                self.hidden_size_per_attention_head,
            ],
        )
        # kv = [2, B, n, L, h]
        kv = ops.transpose(graph, kv, perm=[0, 1, 3, 2, 4])

        with graph.nameScope("attn_past_update"):
            layer_past = ops.kv_cache(
                graph, step, kv, self.cache_max_length, 3, self.sequence_length
            )
            k, v = ops.split(
                graph, layer_past, 2, axis=0, splits=[1, 1], name="split_past"
            )
            # k, v = [B, n, L, h]
            k = ops.squeeze(graph, k, [0])
            v = ops.squeeze(graph, v, [0])
            # [B, n, L, h] -> [B, N, L, h]
            k = ops.unsqueeze(graph, k, [2])
            k = ops.expand(
                graph,
                k,
                [1, 1, self.num_attention_heads // self.multi_query_group_num, 1, 1],
            )
            k = ops.reshape(
                graph,
                k,
                shape=[
                    self.batch_size,
                    self.num_attention_heads,
                    self.cache_max_length,
                    self.hidden_size_per_attention_head,
                ],
            )
            v = ops.unsqueeze(graph, v, [2])
            v = ops.expand(
                graph,
                v,
                [1, 1, self.num_attention_heads // self.multi_query_group_num, 1, 1],
            )
            v = ops.reshape(
                graph,
                v,
                shape=[
                    self.batch_size,
                    self.num_attention_heads,
                    self.cache_max_length,
                    self.hidden_size_per_attention_head,
                ],
            )
            # k = [B, N, h, L]
            k = ops.transpose(graph, k, [0, 1, 3, 2])
        return q, k, v

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        attention_scores = ops.matmul(graph, q, k)
        norm_factor = ops.constant(
            graph, np.array([1.0 / self.norm_factor]).astype(self.np_float_type)
        )
        attention_scores = ops.mul(graph, attention_scores, norm_factor)
        attention_scores = ops.add(graph, attention_scores, attention_mask)

        softmax_fn = self.softmax_fn_map.get(softmax_type, None)
        if not softmax_fn:
            raise ValueError(
                f"Invalid softmax_fn {softmax_type}, options: {self.softmax_fn_map.keys()}"
            )
        attention_probs = softmax_fn(
            graph, attention_scores, -1, stable_mode=self.sequence_length != 1
        )
        attention_probs = ops.cast(graph, attention_probs, self.popart_float_type)
        return attention_probs

    def forward_output(self, graph, score, v):
        context_layer = ops.matmul(graph, score, v)
        context_layer = ops.transpose(graph, context_layer, [0, 2, 1, 3])
        context_layer = ops.reshape(
            graph, context_layer, [self.batch_size, self.sequence_length, -1]
        )
        output = self.dense(graph, context_layer)
        return output

    def __call__(
        self, graph, x, step, attention_mask, sequence_length, softmax_type="ce"
    ):
        with graph.nameScope(self.context):
            self.sequence_length = sequence_length
            q, k, v = self.forward_qkv(graph, x, step)
            score = self.forward_attention(graph, q, k, attention_mask, softmax_type)
            output = self.forward_output(graph, score, v)
        return output


class TPMultiQueryAttention(BaseMultiQueryAttention):
    def collect_bind_layer_weights(self):
        qkv_tp_settings = {
            "strategy_name": "multi_query_qkv",}
        proj_tp_setting = {
            "strategy_name": "end",
        }
        self.query = Linear(
            self.context,
            "query",
            self.hidden_size,
            self.hidden_size,
            use_bias=self.add_qkv_bias,
            **qkv_tp_settings,
        )
        self.key_value = BaseLinear(
            self.context,
            "key_value",
            self.hidden_size,
            2 * self.hidden_size_per_attention_head * self.multi_query_group_num,
            use_bias=self.add_qkv_bias,
        )
        self.dense = Linear(
            self.context,
            "dense",
            self.hidden_size,
            self.hidden_size,
            use_bias=False,
            **proj_tp_setting,
        )

        self.num_attention_heads = self.num_attention_heads // self.num_replicas


class MultiQueryAttention(TPMultiQueryAttention, BaseMultiQueryAttention):
    layer_class_map = {
        "tp": TPMultiQueryAttention,
        "shard": BaseMultiQueryAttention,
    }

    def __init__(
        self,
        context,
        name,
        hidden_size,
        multi_query_group_num,
        num_head,
        cache_max_length,
        layer_number,
        add_qkv_bias=True,
    ):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(
                f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}"
            )
        self.logger.debug(f"initializing model type: {self.layer_class.__name__}")
        super().__init__(
            context,
            name,
            hidden_size,
            multi_query_group_num,
            num_head,
            cache_max_length,
            layer_number,
            add_qkv_bias,
        )

    def __call__(
        self, graph, x, step, attention_mask, sequence_length, softmax_type="ce"
    ):
        return self.layer_class.__call__(
            self, graph, x, step, attention_mask, sequence_length, softmax_type
        )

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)

    def forward_attention(self, graph, q, k, attention_mask, softmax_type):
        return self.layer_class.forward_attention(
            self, graph, q, k, attention_mask, softmax_type
        )

    def forward_qkv(self, graph, x, step):
        return self.layer_class.forward_qkv(self, graph, x, step)

    def forward_output(self, graph, score, v):
        return self.layer_class.forward_output(self, graph, score, v)
