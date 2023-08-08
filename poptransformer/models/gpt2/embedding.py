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
from poptransformer.layers import BaseLayer, Embedding


class BaseTransformerEmbedding(BaseLayer):

    def __init__(self, context, name, vocab_size, embd_size, max_position):
        super().__init__(context, name)
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.max_position = max_position
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.wte = Embedding(self.context, 'wte', self.vocab_size, self.embd_size)
        self.wpe = Embedding(self.context, 'wpe', self.max_position, self.embd_size)

    def __call__(self, graph, input_ids, position_ids, sequence_length):
        with graph.nameScope(self.context):
            input_embeds = self.wte(graph, input_ids, sequence_length)
            pos_embeds = self.wpe(graph, position_ids, sequence_length)
            embeds = ops.add(graph, input_embeds, pos_embeds)
        return ops.remap_tensor(graph, embeds)


class TPTransformerEmbedding(BaseTransformerEmbedding):

    def __call__(self, graph, input_ids, position_ids, sequence_length):
        with graph.nameScope(self.context):
            input_embeds = self.wte(graph, input_ids, sequence_length)
            pos_embeds = self.wpe(graph, position_ids, sequence_length)
            embeds = ops.add(graph, input_embeds, pos_embeds)
            embeds = ops.remap_tensor(graph, embeds)
            embeds = graph.aiGraphcore.replicatedallreduce([embeds])
        return embeds


class TransformerEmbedding(TPTransformerEmbedding, BaseTransformerEmbedding):
    layer_class_map = {
        'tp': TPTransformerEmbedding,
        'shard': BaseTransformerEmbedding}

    def __init__(self, context, name, vocab_size, embd_size, max_position):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, vocab_size, embd_size, max_position)

    def __call__(self, graph, input_ids, position_ids, sequence_length):
        return self.layer_class.__call__(self, graph, input_ids, position_ids, sequence_length)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
