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

import torch
import numpy as np
from poptransformer import ops
from poptransformer.layers import LayerNorm, BaseLayer, LMHead
from poptransformer.models.base_model import HFDecBaseModel
from poptransformer.models.rwkv.ffn import RWKVFeedforward
from poptransformer.models.rwkv.attention import RWKVAttention
from poptransformer.models.rwkv.embedding import RWKVEmbedding


class RWKVBlock(BaseLayer):
    def __init__(self, context, name, hidden_size, intermediate_size, attention_hidden_size, eps, layer_id):
        super().__init__(context, name)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_hidden_size = attention_hidden_size
        self.eps = eps
        self.layer_id = layer_id
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        if self.layer_id == 0:
            self.pre_ln = LayerNorm(self.context, 'pre_ln', self.hidden_size, self.eps)
        self.ln1 = LayerNorm(self.context, 'ln1', self.hidden_size, self.eps)
        self.ln2 = LayerNorm(self.context, 'ln2', self.hidden_size, self.eps)
        self.attention = RWKVAttention(
            self.context, 'attention', self.hidden_size, self.attention_hidden_size, self.layer_id)
        self.feed_forward = RWKVFeedforward(
            self.context, 'feed_forward', self.hidden_size, self.intermediate_size, self.layer_id)

    def __call__(self, graph, hidden, layer_state, sequence_length, norm_type='group'):
        if self.layer_id == 0:
            hidden = self.pre_ln(graph, hidden, sequence_length, norm_type)
        temp = self.ln1(graph, hidden, sequence_length, norm_type)
        attention, layer_state = self.attention(graph, temp, layer_state)
        hidden = ops.add(graph, hidden, attention)
        temp = self.ln2(graph, hidden, sequence_length, norm_type)
        feed_forward, layer_state = self.feed_forward(graph, temp, layer_state)
        hidden = ops.add(graph, hidden, feed_forward)
        return hidden, layer_state


class RWKVModel(BaseLayer):
    def __init__(
        self,
        context,
        name,
        hidden_size,
        intermediate_size,
        attention_hidden_size,
        eps,
        vocab_size,
        num_hidden_layers,
        rescale_every,
        layer_per_ipu
    ):
        super().__init__(context, name)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_hidden_size = attention_hidden_size
        self.eps = eps
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.rescale_every = rescale_every
        self.layer_per_ipu = layer_per_ipu
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.embeddings = RWKVEmbedding(self.context, 'embeddings', self.vocab_size, self.hidden_size)
        self.blocks = [
            RWKVBlock(
                context=self.context,
                name=f'blocks.{i}',
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                attention_hidden_size=self.attention_hidden_size,
                eps=self.eps,
                layer_id=i
            ) for i in range(self.num_hidden_layers)
        ]
        self.ln_out = LayerNorm(self.context, 'ln_out', self.hidden_size, self.eps)

    def __call__(self, graph, input_ids, states, sequence_length, norm_type='ce', **kwargs):
        hidden = self.embeddings(graph, input_ids, sequence_length)
        new_states = []
        outline_blocks = kwargs.get('outline_blocks', None)

        end_points = np.cumsum(self.layer_per_ipu)
        for index, block in enumerate(self.blocks):
            stage_offset = sum(index >= end_points)
            if outline_blocks is None:
                outline_attr = None
            elif outline_blocks == 'single_block':
                outline_attr = {'block': f'sub_{index}'}
            elif outline_blocks == 'multi_block':
                outline_attr = {'block': f'sub_{stage_offset}'}
            else:
                raise ValueError(f'invalid value {outline_blocks} for outline_blocks')
            layer_state = states[5 * index: 5 * index + 5]
            with self.device_scope(graph, stage_offset, None, outline_attr):
                hidden, layer_state = block(graph, hidden, layer_state, sequence_length, norm_type)
                if (index + 1) % self.rescale_every == 0:
                    hidden = ops.div(graph, hidden, ops.constant(graph, np.array(2, dtype=self.np_float_type), '2'))
                new_states.append(layer_state)
        hidden = self.ln_out(graph, hidden, sequence_length, norm_type=norm_type)
        return hidden, new_states


class RWKVDecodeModel(HFDecBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_length = kwargs.get('max_length')
        self.layer_per_ipu = kwargs.get('layer_per_ipu')
        self.outline_blocks = kwargs.get('outline_blocks', True)
        self.topk = kwargs.get('topk')
        self.cell = RWKVModel(
            context=None,
            name='rwkv',
            hidden_size=self.hf_config.hidden_size,
            intermediate_size=self.hf_config.intermediate_size,
            attention_hidden_size=self.hf_config.attention_hidden_size,
            eps=self.hf_config.layer_norm_epsilon,
            vocab_size=self.hf_config.vocab_size,
            num_hidden_layers=self.hf_config.num_hidden_layers,
            rescale_every=self.hf_config.rescale_every,
            layer_per_ipu=self.layer_per_ipu
        )
        self.head = LMHead(
            context=None,
            name='head',
            topk=self.topk,
            vocab_size=self.hf_config.vocab_size,
            embedding_size=self.hf_config.hidden_size
        )
        self.head.set_virtual_id(len(self.layer_per_ipu) - 1)

    def process_hf_model_state_dict(self):
        with torch.no_grad():
            for block_id, block in enumerate(self.hf_model.rwkv.blocks):
                if self.hf_config.rescale_every > 0:
                    block.attention.output.weight.div_(2 ** int(block_id // self.hf_config.rescale_every))
                    block.feed_forward.value.weight.div_(2 ** int(block_id // self.hf_config.rescale_every))

    def build_model_graph_inputs(self, graph, inputs):
        with graph.nameScope('step'):
            inputs['step'] = ops.constant(self.graph, np.array(0).astype(np.int32), 'init_step')
        with graph.nameScope('stop_mask'):
            inputs['stop_mask'] = ops.constant(
                self.graph, np.zeros((self.batch_size,)).astype(np.int32), 'stop_mask')
        with graph.nameScope('states'):
            for i in range(self.hf_config.num_hidden_layers):
                for j in range(2):
                    inputs[f'state_{i}_{j}'] = ops.constant(
                        self.graph,
                        np.zeros(
                        (self.batch_size, 1, self.hf_config.hidden_size), dtype=self.np_float_type),
                        f'layer{i}_{j}'
                    )
                for j in range(2, 4):
                    inputs[f'state_{i}_{j}'] = ops.constant(
                        self.graph,
                        np.zeros(
                            (self.batch_size, 1, self.hf_config.hidden_size // self.num_replicas), dtype=np.float32),
                        f'layer{i}_{j}'
                    )
                inputs[f'state_{i}_{j+1}'] = ops.constant(
                    self.graph,
                    np.zeros(
                        (self.batch_size, 1, self.hf_config.hidden_size // self.num_replicas), dtype=np.float32) - 1e30,
                    f'layer{i}_{j+1}'
                )
        del inputs['position_ids']
        del inputs['attention_mask']
        return inputs

    def build_model_graph(self, model_graph, model_graph_inputs, sequence_length=1):
        input_ids_container = model_graph_inputs['input_ids_container']
        step = model_graph_inputs['step']
        states = [i for i in model_graph_inputs.values() if 'states' in i]
        with self.device_scope(model_graph, 0, 0):
            input_ids = ops.dynamic_slice(model_graph, input_ids_container, step, axes=[1], sizes=[sequence_length])
            logits, states = self.cell(
                graph=model_graph,
                input_ids=input_ids,
                states=states,
                sequence_length=sequence_length,
                norm_type='ce',
                outline_blocks=self.outline_blocks
            )

        next_ids = self.head(
            model_graph,
            logits,
            sequence_length=sequence_length
        )
        model_outputs =  {'next_ids': next_ids, 'stage_offset': 1}
        for index_i, i in enumerate(states):
            for index_j, j in enumerate(i):
                model_outputs[f'states/layer_{index_i}_{index_j}'] = j
        return model_outputs

    def build_post_model_graph(self, model_graph, model_graph_inputs, model_outputs):
        # continue build model graph for post inference step
        stage_offset = model_outputs['stage_offset']
        next_ids = model_outputs['next_ids']
        step = model_graph_inputs['step']
        input_ids_container = model_graph_inputs['input_ids_container']
        stop_mask = model_graph_inputs['stop_mask']

        with self.device_scope(model_graph, 0, pipeline_stage_id=stage_offset):
            next_iput_ids_container, next_step, _, id_to_update= self.step_containers(
                model_graph, input_ids_container, step, None, next_ids
            )
            next_stop_mask, keep_going_cond = self.step_loop_cond(model_graph, id_to_update, stop_mask)
        self.add_output_tensor(
            model_graph,
            [keep_going_cond, next_iput_ids_container, next_step, next_stop_mask] +
            [v for i, v in model_outputs.items() if 'states' in i]
        )

    def build_input_dict(self, **kwargs):
        input_string_list = list(kwargs.get('input_string_list', []))
        batch_size = self.batch_size * self.batch_per_step
        if len(input_string_list) >= batch_size:
            input_string_list = input_string_list[:batch_size]
            self.logger.info('num input strings is larger than batch size, truncating')
        else:
            input_string_list.extend([''] * (batch_size - len(input_string_list)))
            self.logger.info('num input string is smaller than batch size, adding fake inputs')
        inputs = self.hf_tokenizer(
            input_string_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='np',
            return_attention_mask=False,
            add_special_tokens=False
        )
        return {'input_ids': inputs['input_ids']}

    def build_output_dict(self, anchor_arrays):
        output, decode_step = anchor_arrays['Loop:0'], anchor_arrays['Loop:1']
        if len(output.shape) == 3:
            output, decode_step = output[0], decode_step[0]
        output = self.hf_tokenizer.batch_decode(output, skip_special_tokens=True)
        output = {str(i): v for i, v in enumerate(output)}
        return {'output': output, 'decode_step': decode_step}
