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
from poptransformer.layers import BaseLayer, Linear


class BaseLMHead(BaseLayer):

    def __init__(self,
                 context,
                 name,
                 vocab_size,
                 topk,
                 embedding_size,
                 embedding_weights,
                 num_embedding_partitions=1,
                 token_offsets=None,
                 embedding_pad_mask=None):
        super().__init__(context, name)
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.topk = topk
        self.embedding_weights = embedding_weights
        self.num_embedding_partitions = num_embedding_partitions
        self.embedding_split_size = vocab_size // num_embedding_partitions + 2
        self.lm_split_count = 6 if self.precision == "int4" else 2
        self.lm_split_size = self.embedding_split_size // self.lm_split_count
        self.logits_pad_mask = np.zeros(
            (1, self.embedding_split_size), dtype=self.np_float_type
        )
        self.logits_pad_mask[:, 0] = -1000.0
        self.logits_pad_mask[:, -1] = -1000.0
        self.token_offsets = token_offsets
        self.embedding_pad_mask = embedding_pad_mask
        # for TP
        self.virtual_id = None
        self.use_tied_embedding = False
        self.tie_weight = embedding_weights
        self.collect_bind_layer_weights()

    def set_virtual_id(self, virtual_id):
        pass

    def collect_bind_layer_weights(self):
        pass

    def __call__(self, graph, hidden_states, sequence_length):
        next_token_prob_list = []
        next_token_list = []
        for i in range(self.num_embedding_partitions):
            with graph.virtualGraph(i):
                transposed_token_weight_ = ops.transpose(
                    graph, self.embedding_weights[i], [1, 0]
                )
                # (B, H) * (H, V/n + 2) -> (B, V/n + 2)
                res = []
                for j in range(self.lm_split_count):
                    y_ = ops.static_slice(
                        graph,
                        transposed_token_weight_,
                        [self.lm_split_size * j],
                        [self.lm_split_size * (j+1)],
                        [1]
                    )
                    with self.option_scope(amp=0.2):
                        o_ = ops.matmul(graph, hidden_states, y_)
                    res.append(o_)
                logits_ = ops.concat_sequence(graph, res, axis=1)
                logits_ = ops.add(
                    graph,
                    logits_,
                    ops.constant(graph, self.logits_pad_mask,
                                 f"logits_pad_mask_{i}"),
                )
                if i == self.num_embedding_partitions - 1:
                    logits_ = ops.add(
                        graph,
                        logits_,
                        ops.constant(
                            graph,
                            self.embedding_pad_mask,
                            f"embedding_pad_mask_{i}",
                        ),
                    )

                # (B, V/n + 2) -> (B, k)
                next_token_prob_, next_token_ = ops.topk(
                    graph, logits_, axis=1, k=self.topk)
                next_token_ = ops.add(
                    graph,
                    next_token_,
                    ops.constant(
                        graph,
                        np.array([self.token_offsets[i]], dtype=np.int32),
                        f"token_offset_{i}",
                    ),
                )
                next_token_prob_list.append(next_token_prob_)
                next_token_list.append(next_token_)

        with graph.virtualGraph(self.num_embedding_partitions - 1):
            next_token_probs = ops.concat_sequence(
                graph, next_token_prob_list, 1)
            next_tokens = ops.concat_sequence(graph, next_token_list, 1)
            idx = ops.argmax(graph, next_token_probs, 1)
            idx = ops.squeeze(graph, idx, [0])
            next_token = ops.dynamic_slice(graph, next_tokens, idx, [1], [1])
        return next_token


class TPLMHead(BaseLMHead):
    def tie_embedding(self, weight_id):
        self.head.weight_id = weight_id
        self.use_tied_embedding = True
        self.logger.info(f'setting weight id to {weight_id}')

    def set_virtual_id(self, virtual_id):
        self.virtual_id = virtual_id
        self.logger.info(f'setting virtual id to {virtual_id}')

    def collect_bind_layer_weights(self):
        if not self.tie_weight:
            lm_tp_settings = {
                'strategy_name': 'start',
            }
        else:
            lm_tp_settings = {
                'strategy_name': 'identity',
            }
        self.head = Linear(self.context, None, self.embedding_size,
                           self.vocab_size, False, **lm_tp_settings)
        if self.tie_weight:
            self.tie_embedding(self.tie_weight)

    def __call__(self, graph, logits, sequence_length):
        with graph.virtualGraph(self.virtual_id):
            vs_setting = {'vs_type': 'consecutive', 'group_size': 1}
            vocab_per_ipu = math.ceil(self.vocab_size / self.num_replicas)
            index_offset_np = np.expand_dims(
                np.arange(self.num_replicas, dtype=np.int32), [1, 2]) * vocab_per_ipu
            index_offset = self.add_initialized_input_tensor(
                index_offset_np, 'index_offset', **vs_setting)
            if self.use_tied_embedding:
                self.head.weight_id = ops.transpose(
                    graph, self.head.weight_id, [1, 0])
                # Avoid to transpose twice in 2 stage mode.
                self.use_tied_embedding = False
            logits = self.head(graph, logits)
            logits = ops.unsqueeze(graph, logits, [1])  # Align with sharding
            pad_idx = ops.equal(graph, ops.constant(graph, np.array(
                0.0).astype(self.np_float_type), 'zero'), logits)
            logits = ops.where(
                graph, pad_idx, ops.constant(graph, np.array(-10000.0).astype(self.np_float_type), '-10000'), logits)
            next_token_prob, next_token_ = ops.topk(
                graph, logits, axis=2, k=self.topk)
            next_token = ops.add(graph, next_token_, index_offset)
            next_token_prob = ops.replicated_allgather(graph, next_token_prob)
            next_token_topk = ops.replicated_allgather(graph, next_token)
            next_token_prob_shape = [sequence_length * self.topk * self.num_replicas, self.batch_size]
            next_token_prob = ops.transpose(
                graph,
                ops.reshape(graph, next_token_prob, next_token_prob_shape),
                [1, 0]
            )
            next_token_topk_shape = [
                sequence_length * self.topk * self.num_replicas, self.batch_size]
            next_token_topk = ops.transpose(
                graph,
                ops.reshape(graph, next_token_topk, next_token_topk_shape),
                [1, 0]
            )
            next_token_prob = ops.reshape(
                graph, next_token_prob, [self.batch_size, sequence_length, self.topk * self.num_replicas])
            next_token_topk = ops.reshape(
                graph, next_token_topk, [self.batch_size, sequence_length, self.topk * self.num_replicas])

            next_token_idx = ops.argmax(graph, next_token_prob, axis=2)  # [B,1,1]
            next_token_idx = ops.squeeze(graph, next_token_idx, [1, 2])  # (B,)
            next_token_topk = ops.squeeze(graph, next_token_topk, [1])  # [B,topk * num_replica]
            next_token = ops.grouped_gather(
                graph, next_token_topk, next_token_idx, axis=1, group_size=self.batch_size)  # (B,)
            next_token = ops.reshape(graph, next_token, [self.batch_size, -1])
            return next_token


class LMHead(TPLMHead, BaseLMHead):
    layer_class_map = {"tp": TPLMHead, "shard": BaseLMHead}

    def __init__(
        self,
        context,
        name,
        vocab_size,
        topk,
        embedding_size,
        embedding_weights,
        num_embedding_partitions=1,
        token_offsets=None,
        embedding_pad_mask=None,
    ):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(
                f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}"
            )
        self.logger.debug(
            f"initializing model type: {self.layer_class.__name__}")
        super().__init__(
            context, name, vocab_size, topk, embedding_size, embedding_weights,
            num_embedding_partitions, token_offsets, embedding_pad_mask)

    def __call__(self, graph, hidden_states, sequence_length):
        return self.layer_class.__call__(self, graph, hidden_states, sequence_length)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
