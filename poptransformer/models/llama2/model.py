# Copyright (c) 2023 Graphcore Ltd.
# This is a re-implementation of Llama 2 by Graphcore Ltd
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import numpy as np

from transformers import AutoTokenizer
from transformers import AutoConfig

from poptransformer import ops
from poptransformer.layers import BaseLayer
from poptransformer.models.llama2.embedding import TransformerEmbedding
from poptransformer.models.llama2.mlp import MLP
from poptransformer.layers.rms_layer_norm import RMSLayerNorm
from poptransformer.models.llama2.attention import Attention
from poptransformer.layers.lm_head import LMHead
from poptransformer.models import HFDecBaseModel


class TransformerBlock(BaseLayer):
    def __init__(self, context, name, input_size, eps, n_head,intermediate_size, max_length, fp8_cache=False):
        super().__init__(context, name)
        self.input_size = input_size
        self.eps = eps
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.max_length = max_length
        self.fp8_cache = fp8_cache
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.layer_norm1 = RMSLayerNorm(self.context, 'input_layernorm', self.input_size, self.eps)
        self.attention = Attention(
            self.context, 'self_attn', self.input_size, self.n_head, self.max_length, self.fp8_cache)
        self.layer_norm2 = RMSLayerNorm(self.context, 'post_attention_layernorm', self.input_size, self.eps)
        self.mlp = MLP(self.context, 'mlp', self.input_size, self.intermediate_size, 'swish')

    def __call__(self, graph, x, sequence_length, step, attention_mask, norm_type='ce', softmax_type='ce', **kwargs):
        with graph.nameScope(self.context):
            temp_x = self.layer_norm1(graph, x)
            temp_x = self.attention(graph, temp_x, step, attention_mask, sequence_length, softmax_type)
            x = ops.add(graph, x, temp_x)
            temp_x = self.layer_norm2(graph, x)
            temp_x = self.mlp(graph, temp_x)
            x = ops.add(graph, x, temp_x)
        return x


class Transformer(BaseLayer):
    def __init__(self, context, name, vocab_size, embd_size, eps,
                 n_head, intermediate_size, max_length, n_layer, layer_per_ipu, fp8_cache=False):
        super().__init__(context, name)
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.eps = eps
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.max_length = max_length
        self.n_layer = n_layer
        self.layer_per_ipu = layer_per_ipu
        self.fp8_cache = fp8_cache
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):

        self.embedding = TransformerEmbedding(
            self.context,
            "embed_tokens",
            self.vocab_size,
            self.embd_size,
        )
        self.blocks = [
            TransformerBlock(
                self.context,
                'layers.'+str(i),
                self.embd_size,
                self.eps,
                self.n_head,
                self.intermediate_size,
                self.max_length,
                self.fp8_cache
            )
            for i in range(self.n_layer)
        ]
        self.layer_norm = RMSLayerNorm(self.context, 'norm', self.embd_size, self.eps)

    def __call__(self, graph, input_ids, position_ids, step, attention_mask, sequence_length, **kwargs):
        norm_type = kwargs.get('norm_type', 'ce')
        softmax_type = kwargs.get('softmax_type', 'ce')
        outline_blocks = kwargs.get('outline_blocks', 'single_block')
        if outline_blocks:
            self.logger.info('outlining transformer blocks')
            self.logger.info('please make sure disable outlining in session options is set to False')

        with self.device_scope(graph,0,0):
            hidden_states = self.embedding(graph, input_ids, sequence_length)
        end_points = np.cumsum(self.layer_per_ipu)

        for i in range(self.n_layer):
            stage_offset = sum(i >= end_points)
            with self.device_scope(graph, stage_offset, stage_offset):
                hidden_states = self.blocks[i](
                    graph, hidden_states, sequence_length, step, attention_mask, norm_type, softmax_type)
                self.logger.info(f'block {i} placed on IPU {stage_offset}')

        with self.device_scope(graph,stage_offset,stage_offset):
            hidden_states = self.layer_norm(graph, hidden_states)

        return hidden_states


class LLAMA2DecModel(HFDecBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fp8_cache = kwargs.get('fp8_cache', False)
        self.topk = kwargs.get('topk', 1)
        self.outline_blocks = kwargs.get('outline_blocks', False)
        self.transformer = Transformer(
            None,
            'model',
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            self.hf_config.rms_norm_eps,
            self.hf_config.num_attention_heads,
            self.hf_config.intermediate_size,
            self.max_length,
            self.hf_config.num_hidden_layers,
            self.layer_per_ipu,
            self.fp8_cache,
        )
        self.lm_head = LMHead(None, 'lm_head', self.topk, self.hf_config.vocab_size, self.hf_config.hidden_size)

    def prepare_state_dict(self):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name,
            cache_dir=self.hf_cache_dir,
            pad_token='[PAD]'
        )
        #TODO find out why need this setting of padding_side
        self.hf_tokenizer.padding_side = "right"

        self.logger.info(f'initialized tokenizer by model_name: {self.hf_model_name}')
        if not self.override_hfconfig_from_json:
            model_class = self.hf_model_class_name_map.get(self.hf_model_class_name, None)
            assert model_class, f"Invalid hf_model_class_name: {self.hf_model_class_name}"
            assert self.hf_model_name, f"Invalid hf_model_name: {self.hf_model_name}"
            self.logger.info(f'initializing hf model class: {model_class.__name__}')
            self.hf_model = model_class.from_pretrained(self.hf_model_name, cache_dir=self.hf_cache_dir)
            self.logger.info(f'loading pretrained hf model: {self.hf_model_name}')
            self.hf_config = self.hf_model.config
            if self.precision != 'fp32':
                self.hf_model.half()
                self.logger.info(f'casting model to {self.precision}')
            self.state_dict = self.hf_model.state_dict()
            self.register_state_dict()
        else:
            self.logger.info('using overrided config, no state dict loaded')
            self.hf_config = AutoConfig.from_pretrained(self.override_hfconfig_from_json)
            self.state_dict = {}

    def build_model_graph(self, model_graph, model_graph_inputs, sequence_length=1):
        input_ids_container = model_graph_inputs['input_ids_container']
        attention_mask = model_graph_inputs['attention_mask']
        step = model_graph_inputs['step']

        with self.device_scope(model_graph,0,0):
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

        with self.device_scope(model_graph,len(self.layer_per_ipu)-1,len(self.layer_per_ipu)-1):
            self.lm_head.set_virtual_id(len(self.layer_per_ipu)-1)
            next_ids = self.lm_head(
                model_graph,
                logits,
                sequence_length=sequence_length
            )

        model_outputs =  {'next_ids': next_ids, 'stage_offset': len(self.layer_per_ipu)}
        return model_outputs

    def build_input_dict(self, **kwargs):
        input_string_list = list(kwargs.get('input_string_list', []))
        batch_size = self.batch_size * self.batch_per_step
        if len(input_string_list) >= batch_size:
            input_string_list = input_string_list[:batch_size]
            self.logger.info('num input strings is larger than batch size, truncating')
        else:
            input_string_list.extend([input_string_list[0]] * (batch_size - len(input_string_list)))
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
            # TODO try deal with this automatically
        if len(output.shape) == 4:
            output, decode_step = output[1][0], decode_step[0][0]

        output = self.hf_tokenizer.batch_decode(output, skip_special_tokens=True)
        output = {str(i): v for i, v in enumerate(output)}
        return {'output': output, 'decode_step': decode_step}
