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
from poptransformer.layers import BaseLayer, Embedding


def split_embedding_weight(weight, vocab_size, count, pad_num=0):
    """
    1. For example vocab_size=150528, count=4, the weight will be splited into 4 parts,
    Each part has 37634 tokens = 37632 valid + 2 pad:

        Emb1: [0(PAD), ..., 37631, 37632, 37633(PAD)]
        Emb2: [0(PAD), ..., 37631, 37632, 37633(PAD)]
        Emb3: [0(PAD), ..., 37631, 37632, 37633(PAD)]
        Emb4: [0(PAD), ..., 37631, 37632, 37633(PAD)]

    2. Besides, input ids should be cliped to avoid out of bounds:

                    IPU-0             IPU-1              IPU-2                IPU-3
                |--37632--|      |---37632---|      |---37632---|       |----37632----|
        input:  [0, ..., 37631, 37632, ..., 75263, 75264, ..., 112895, 112896, ..., 150527]

        input1: clip(input, -1, 37632) -> sub(-1)
        input2: clip(input, 37631, 75264) -> sub(37631)
        input3: clip(input, 75263, 112896) -> sub(75263)
        input4: clip(input, 112895, 150528) -> sub(112895)

        final_input = input1 + input2 + input3 + input4
    """
    split = []  # [tensor, clip_min, clip_max, offset]
    assert vocab_size % count == 0
    num_valid_token = vocab_size // count

    weight_list = np.split(weight, count, 0)
    pad = np.zeros((1, weight.shape[1]), dtype=weight.dtype)
    for i in range(count):
        weight_ = weight_list[i]
        # Each partial embedding has 2 pad tokens.
        weight_ = np.concatenate([pad, weight_], axis=0)
        weight_ = np.concatenate([weight_, pad], axis=0)

        clip_min = i * num_valid_token - 1
        clip_max = (i + 1) * num_valid_token
        offset = clip_min
        split.append([weight_, clip_min, clip_max, offset])

    pad_mask = np.zeros((1, weight_.shape[0]), dtype=weight_.dtype)
    if pad_num > 0:
        pad_mask[:, -(pad_num + 1) :] = -1000.0
    return split, pad_mask


class BaseChatGLMEmbedding(BaseLayer):
    def __init__(self, context, name, vocab_size, embd_size, num_embedding_partitions=1):
        super().__init__(context, name)
        self.vocab_size = vocab_size
        self.embd_size = embd_size

        self.num_embedding_partitions = num_embedding_partitions
        self.embedding_split_size = vocab_size // num_embedding_partitions + 2
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        raise NotImplementedError

    def __call__(self, input_ids, sequence_length):
        raise NotImplementedError


class ShardChatGLMEmbedding(BaseChatGLMEmbedding):
    def collect_bind_layer_weights(self):
        weight_key = ".".join([self.context, "word_embeddings", "weight"])
        weight_np = self.get_param_from_state_dict(weight_key, None)
        pad_num = self.vocab_size - weight_np.shape[0]
        if pad_num > 0:
            pad_weight = np.zeros((pad_num, weight_np.shape[1]), dtype=weight_np.dtype)
            weight_np = np.concatenate([weight_np, pad_weight], axis=0)
        self.weights, self.embedding_pad_mask = split_embedding_weight(
            weight_np, self.vocab_size, self.num_embedding_partitions, pad_num
        )
        self.weight_ids = []
        self.token_offsets = []
        self.clips = []
        for i in range(self.num_embedding_partitions):
            weight_, clip_min, clip_max, offset = self.weights[i]
            weight_id = self.add_initialized_input_tensor(weight_, f"token_weight_{i}")
            self.weight_ids.append(weight_id)
            self.token_offsets.append(offset)
            self.clips.append([clip_min, clip_max])

    def __call__(self, graph, input_ids, sequence_length):
        inputs_embeds_list = []
        with graph.nameScope(self.context):
            for i in range(self.num_embedding_partitions):
                with graph.virtualGraph(i):
                    # input_ids = ops.add(graph, input_ids, ops.constant(graph, np.array([0], dtype=np.int32)))
                    input_ids_ = ops.clip(
                        graph, input_ids, self.clips[i][0], self.clips[i][1]
                    )
                    input_ids_ = ops.sub(
                        graph,
                        input_ids_,
                        ops.constant(
                            graph,
                            np.array([self.token_offsets[i]], dtype=np.int32),
                            f"token_offset_{i}",
                        ),
                    )
                    inputs_embeds_ = ops.gather(graph, self.weight_ids[i], input_ids_)
                    inputs_embeds_list.append(inputs_embeds_)
            with graph.virtualGraph(self.num_embedding_partitions - 1):
                inputs_embeds = ops.sum_op(graph, inputs_embeds_list)
                inputs_embeds = ops.remap_tensor(graph, inputs_embeds)

        return inputs_embeds


class TPChatGLMEmbedding(BaseChatGLMEmbedding):
    def collect_bind_layer_weights(self):
        self.wte = Embedding(self.context, 'word_embeddings', self.vocab_size, self.embd_size)
        # For unify the interface
        self.weight_ids = self.wte.weight_id
        self.token_offsets = [1]
        self.embedding_pad_mask = None

    def __call__(self, graph, input_ids, sequence_length):
        with graph.nameScope(self.context):
            input_embeds = self.wte(graph, input_ids, sequence_length)
            # embeds = ops.remap_tensor(graph, input_embeds)
            embeds = graph.aiGraphcore.replicatedallreduce([input_embeds])
        return embeds


class ChatGLMEmbedding(TPChatGLMEmbedding, ShardChatGLMEmbedding):
    layer_class_map = {
        "tp": TPChatGLMEmbedding,
        "shard": ShardChatGLMEmbedding,
    }

    def __init__(self, context, name, vocab_size, embd_size, num_embedding_partitions):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(
                f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}"
            )
        self.logger.debug(f"initializing model type: {self.layer_class.__name__}")
        super().__init__(context, name, vocab_size, embd_size, num_embedding_partitions)

    def __call__(self, graph, input_ids, sequence_length):
        return self.layer_class.__call__(self, graph, input_ids, sequence_length)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
