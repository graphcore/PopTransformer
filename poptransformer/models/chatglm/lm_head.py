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
from poptransformer.layers import BaseLayer


class BaseLMHead(BaseLayer):
    def __init__(
        self,
        context,
        name,
        vocab_size,
        num_embedding_partitions,
        token_offsets,
        embedding_pad_mask,
    ):
        super().__init__(context, name)
        self.num_embedding_partitions = num_embedding_partitions
        self.embedding_split_size = vocab_size // num_embedding_partitions + 2
        self.lm_split_count = 2
        self.lm_split_size = self.embedding_split_size // self.lm_split_count
        self.logits_pad_mask = np.zeros(
            (1, self.embedding_split_size), dtype=np.float16
        )
        self.logits_pad_mask[:, 0] = -1000.0
        self.logits_pad_mask[:, -1] = -1000.0
        self.token_offsets = token_offsets
        self.embedding_pad_mask = embedding_pad_mask

    def collect_bind_layer_weights(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class ShardLMHead(BaseLMHead):
    def __call__(self, graph, hidden_states, embedding_weights):
        next_token_prob_list = []
        next_token_list = []
        for i in range(self.num_embedding_partitions):
            with graph.virtualGraph(i):
                transposed_token_weight_ = ops.transpose(
                    graph, embedding_weights[i], [1, 0]
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
                    o_ = ops.matmul(graph, hidden_states, y_)
                    res.append(o_)
                logits_ = ops.concat_sequence(graph, res, axis=1)
                logits_ = ops.add(
                    graph,
                    logits_,
                    ops.constant(graph, self.logits_pad_mask, f"logits_pad_mask_{i}"),
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
                next_token_prob_, next_token_ = ops.topk(graph, logits_, axis=1, k=1)
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
            next_token_probs = ops.concat_sequence(graph, next_token_prob_list, 1)
            next_tokens = ops.concat_sequence(graph, next_token_list, 1)
            idx = ops.argmax(graph, next_token_probs, 1)
            idx = ops.squeeze(graph, idx, [0])
            next_token = ops.dynamic_slice(graph, next_tokens, idx, [1], [1])

        return next_token


class TPLMHead(BaseLMHead):
    pass


class LMHead(TPLMHead, ShardLMHead):
    layer_class_map = {"tp": TPLMHead, "shard": ShardLMHead}

    def __init__(
        self,
        context,
        name,
        vocab_size,
        num_embedding_partitions,
        token_offsets,
        embedding_pad_mask,
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
            vocab_size,
            num_embedding_partitions,
            token_offsets,
            embedding_pad_mask,
        )

    def __call__(self, graph, hidden_states, embedding_weights):
        return self.layer_class.__call__(self, graph, hidden_states, embedding_weights)

    def collect_bind_layer_weights(self):
        pass
