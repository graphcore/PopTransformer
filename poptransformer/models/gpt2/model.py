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
from poptransformer.layers import MLP, LayerNorm, BaseLayer, LMHead
from poptransformer.models import HFDecBaseModel
from poptransformer.models.gpt2.embedding import TransformerEmbedding
from poptransformer.models.gpt2.attention import Attention


class TransformerBlock(BaseLayer):
    def __init__(self, context, name, input_size, eps, n_head, max_length):
        super().__init__(context, name)
        self.input_size = input_size
        self.eps = eps
        self.n_head = n_head
        self.max_length = max_length
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.layer_norm1 = LayerNorm(self.context, 'ln_1', self.input_size, self.eps)
        self.attention = Attention(
            self.context, 'attn', self.input_size, self.n_head, self.max_length)
        self.layer_norm2 = LayerNorm(self.context, 'ln_2', self.input_size, self.eps)
        self.mlp = MLP(self.context, 'mlp', self.input_size, self.input_size * 4, 'gelu')

    def __call__(self, graph, x, sequence_length, step, attention_mask, norm_type='ce', softmax_type='ce', **kwargs):
        with graph.nameScope(self.context):
            temp_x = self.layer_norm1(graph, x, sequence_length, norm_type)
            temp_x = self.attention(graph, temp_x, step, attention_mask, sequence_length, softmax_type)
            x = ops.add(graph, x, temp_x)
            temp_x = self.layer_norm2(graph, x, sequence_length, norm_type)
            temp_x = self.mlp(graph, temp_x)
            x = ops.add(graph, x, temp_x)
        return x


class Transformer(BaseLayer):
    def __init__(self, context, name, vocab_size, embd_size, eps,
                 n_head, max_length, n_layer, layer_per_ipu, max_position):
        super().__init__(context, name)
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.eps = eps
        self.n_head = n_head
        self.max_length = max_length
        self.n_layer = n_layer
        self.layer_per_ipu = layer_per_ipu
        self.max_position = max_position
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):

        self.embedding = TransformerEmbedding(
            self.context,
            None,
            self.vocab_size,
            self.embd_size,
            self.max_position
        )
        self.blocks = [
            TransformerBlock(
                self.context,
                'h.'+str(i),
                self.embd_size,
                self.eps,
                self.n_head,
                self.max_length,
            )
            for i in range(self.n_layer)
        ]
        self.layer_norm = LayerNorm(self.context, 'ln_f', self.embd_size, self.eps)

    def __call__(self, graph, input_ids, position_ids, step, attention_mask, sequence_length, **kwargs):
        # TODO type hints  sequence length is int here
        norm_type = kwargs.get('norm_type', 'ce')
        softmax_type = kwargs.get('softmax_type', 'ce')
        return_last = kwargs.get('return_last', False)
        outline_blocks = kwargs.get('outline_blocks', 'single_block')
        if outline_blocks:
            self.logger.info('outlining transformer blocks')
            self.logger.info('please make sure disable outlining in session options is set to False')
        with graph.virtualGraph(0):
            hidden_states = self.embedding(graph, input_ids, position_ids, sequence_length)

        end_points = np.cumsum(self.layer_per_ipu)

        for i in range(self.n_layer):
            stage_offset = sum(i >= end_points)
            with graph.virtualGraph(stage_offset):
                if outline_blocks == 'single_block':
                    with graph.outlineAttributes({'block': 'sub_' + str(i)}):
                        hidden_states = self.blocks[i](
                            graph, hidden_states, sequence_length, step, attention_mask, norm_type, softmax_type)
                elif outline_blocks == 'multi_block':
                    with graph.outlineAttributes({'block': 'sub_' + str(stage_offset)}):
                        hidden_states = self.blocks[i](
                            graph, hidden_states, sequence_length, step, attention_mask, norm_type, softmax_type)
                else:
                    hidden_states = self.blocks[i](
                        graph, hidden_states, sequence_length, step, attention_mask, norm_type, softmax_type)
                self.logger.info(f'block {i} placed on IPU {stage_offset}')

        with graph.virtualGraph(stage_offset):
            if return_last:
                hidden_states = ops.static_slice(graph, hidden_states, [0], [1], [1])
            last_sequence_length = 1 if return_last else sequence_length
            hidden_states = self.layer_norm(graph, hidden_states, last_sequence_length, norm_type)
        return hidden_states


class GPT2DecModel(HFDecBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO support topk > 1 for TP
        self.topk = kwargs.get('topk', 1)
        if self.model_type == 'tp':
            assert self.topk == 1
        self.outline_blocks = kwargs.get('outline_blocks', True)
        self.transformer = Transformer(
            None,
            'transformer',
            self.hf_config.vocab_size,
            self.hf_config.n_embd,
            self.hf_config.layer_norm_epsilon,
            self.hf_config.n_head,
            self.max_length,
            self.hf_config.n_layer,
            self.layer_per_ipu,
            self.hf_config.n_positions,
        )
        self.lm_head = LMHead(None, 'lm_head', self.topk)

    def process_hf_model_state_dict(self):
        with torch.no_grad():
            n_embd = self.hf_config.n_embd
            head_size = n_embd // self.hf_config.n_head
            scale_value = float(1 / head_size ** 0.5)
            for block in self.hf_model.transformer.h:
                block.attn.c_attn.weight[:, :n_embd] = block.attn.c_attn.weight[:, :n_embd] * scale_value
                block.attn.c_attn.bias[: n_embd] = block.attn.c_attn.bias[:n_embd] * scale_value
        self.logger.info('prescale applied')

    def build_model_graph(self, input_ids_container, step, attention_mask, stop_mask, sequence_length=1):
        model_graph = self.graph.createSubgraphBuilder()
        model_graph.setGraphName('model_graph')
        # add inputs for loop op
        self.add_input_tensor(model_graph, 'BOOL', [], 'cond_place_holder')
        self.add_input_tensor(model_graph, 'INT32', [], 'max_loop_place_holder')
        # add inputs for looping graph
        self.add_input_tensor_from_parent_graph(
            model_graph, [input_ids_container, step, attention_mask, stop_mask])

        with model_graph.virtualGraph(0):
            input_ids = ops.dynamic_slice(model_graph, input_ids_container, step, axes=[1], sizes=[sequence_length])
            temp_attention_mask = ops.unsqueeze(model_graph, attention_mask, [1, 2])
            if sequence_length != 1:
                position_ids_value = np.array([np.arange(self.max_length)] * self.batch_size)
                position_ids_container = ops.constant(
                    model_graph, position_ids_value.astype(np.int32), 'position_ids')
                position_ids = ops.dynamic_slice(model_graph, position_ids_container, step, [1], [sequence_length])
            else:
                position_ids = ops.unsqueeze(model_graph, step, [0])

        logits = self.transformer(
            model_graph,
            input_ids,
            position_ids,
            step,
            temp_attention_mask,
            sequence_length=sequence_length,
            outline_blocks=self.outline_blocks
        )
        with model_graph.virtualGraph(0):
            next_ids = self.lm_head(
                model_graph,
                logits,
                self.transformer.embedding.wte.weight_id,
                self.transformer.embedding.wte.index_offset,
                sequence_length=sequence_length
            )
        with model_graph.virtualGraph(0):
            next_iput_ids_container, next_step, next_attention_mask, id_to_update= self.step_containers(
                model_graph, input_ids_container, step, attention_mask, next_ids
            )
            next_stop_mask, keep_going_cond = self.step_loop_cond(model_graph, id_to_update, stop_mask)

        self.add_output_tensor(
            model_graph,
            [keep_going_cond, next_iput_ids_container, next_step, next_attention_mask, next_stop_mask])
        return model_graph

    def build_input_dict(self, **kwargs):
        input_string_list = list(kwargs.get('input_string_list', []))
        if len(input_string_list) >= self.batch_size:
            input_string_list = input_string_list[:self.batch_size]
            self.logger.info('num input strings is larger than batch size, truncating')
        else:
            input_string_list.extend([''] * (self.batch_size - len(input_string_list)))
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
