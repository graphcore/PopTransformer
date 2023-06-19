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
from poptransformer import ops
from .base_layer import BaseLayer


class BaseLMHead(BaseLayer):

    def __init__(self, context, name, topk):
        super().__init__(context, name)
        self.topk = topk

    def collect_bind_layer_weights(self):
        pass

    def __call__(self, graph, logits, embedding_weight, index_offset, sequence_length):
        transposed_wte_weight = ops.transpose(graph, embedding_weight, [1, 0])
        logits = ops.matmul(graph, logits, transposed_wte_weight)
        if self.topk != 1:
            next_token = ops.random_sample(graph, logits, k=self.topk, dim=2)
        else:
            _, next_token = ops.topk(graph, logits, axis=2, k=1)
        next_token = ops.squeeze(graph, next_token, [2])
        return next_token


class TPLMHead(BaseLMHead):

    def __call__(self, graph, hidden_states, embedding_weight, index_offset, sequence_length):
        transposed_wte_weight = ops.transpose(graph, embedding_weight, [1, 0])
        logits = ops.matmul(graph, hidden_states, transposed_wte_weight)

        pad_idx = ops.equal(graph, ops.constant(graph, np.array(0.0).astype(np.float16), 'zero'), logits)
        logits = ops.where(
            graph, pad_idx, ops.constant(graph, np.array(-10000.0).astype(np.float16), '-10000'), logits)
        next_token_prob, next_token_ = ops.topk(graph, logits, axis=2, k=self.topk)
        next_token = ops.add(graph, next_token_, index_offset)
        next_token_prob = ops.replicated_allgather(graph, next_token_prob)
        next_token_topk = ops.replicated_allgather(graph, next_token)
        next_token_prob_shape = [sequence_length * self.topk * self.num_replicas, self.batch_size]
        next_token_prob = ops.transpose(
            graph,
            ops.reshape(graph, next_token_prob, next_token_prob_shape),
            [1, 0]
        )
        next_token_topk_shape = [sequence_length * self.topk * self.num_replicas, self.batch_size]
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
    layer_class_map = {
        'tp': TPLMHead,
        'shard': BaseLMHead}

    def __init__(self, context, name, topk):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, topk)

    def __call__(self, graph, logits, embedding_weight, index_offset, sequence_length):
        return self.layer_class.__call__(self, graph, logits, embedding_weight, index_offset, sequence_length)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
