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

from transformers import AutoModel, AutoConfig, AutoTokenizer

from poptransformer import ops
from poptransformer.layers import BaseLayer, LMHead
from poptransformer.layers.rms_layer_norm import RMSLayerNorm
from poptransformer.models.chatglm2.emebdding import TransformerEmbedding
from poptransformer.models.chatglm2.attention import MultiQueryAttention
from poptransformer.models.chatglm2.mlp import MLP
from poptransformer.models import HFDecBaseModel
from poptransformer.utils import REGISTRY


def rearrange_tp_weights(weight, num_heads, num_replicas, axis=0):
    hidden_size = weight.shape[axis]
    offset = hidden_size // 2
    head_size = hidden_size // num_heads
    group_size = num_heads // num_replicas // 2
    if axis == 1:
        weight = weight.T
    weights = []
    # Divide weight into groups for each replica and rearange them.
    for i in range(0, num_heads // 2, group_size):
        weight_1 = weight[
            i * head_size : (i + group_size) * head_size
        ]
        weight_2 = weight[
            offset + i * head_size : offset + (i + group_size) * head_size
        ]
        weights.extend([weight_1, weight_2])
    weight_new = torch.cat(weights, axis=0)
    if axis == 1:
        weight_new = weight_new.T
    return weight_new


class ChatGLM2Block(BaseLayer):
    def __init__(
        self,
        context,
        name,
        hidden_size,
        layernorm_epsilon,
        multi_query_group_num,
        num_attention_heads,
        ffn_hidden_size,
        max_length,
        layer_number,
    ):
        super().__init__(context, name)
        self.hidden_size = hidden_size
        self.multi_query_group_num = multi_query_group_num
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.layernorm_epsilon = layernorm_epsilon
        self.max_length = max_length
        self.layer_number = layer_number
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.layer_norm1 = RMSLayerNorm(
            self.context, "input_layernorm", self.hidden_size, self.layernorm_epsilon
        )
        self.attention = MultiQueryAttention(
            self.context,
            "self_attention",
            self.hidden_size,
            self.multi_query_group_num,
            self.num_attention_heads,
            self.max_length,
            self.layer_number,
        )
        self.layer_norm2 = RMSLayerNorm(
            self.context,
            "post_attention_layernorm",
            self.hidden_size,
            self.layernorm_epsilon,
        )
        self.mlp = MLP(self.context, "mlp", self.hidden_size, self.ffn_hidden_size)

    def __call__(
        self,
        graph,
        x,
        sequence_length,
        step,
        attention_mask,
        softmax_type="ce",
        **kwargs,
    ):
        matmul_kwargs = {"amp": 0.4, "partialtype": "half"}
        with graph.nameScope(self.context):
            temp_x = self.layer_norm1(graph, x)
            with self.option_scope(**matmul_kwargs):
                temp_x = self.attention(
                    graph, temp_x, step, attention_mask, sequence_length, softmax_type
                )
            x = ops.add(graph, x, temp_x)
            temp_x = self.layer_norm2(graph, x)
            with self.option_scope(**matmul_kwargs):
                temp_x = self.mlp(graph, temp_x)
            x = ops.add(graph, x, temp_x)
        return x


class Transformer(BaseLayer):
    def __init__(
        self,
        context,
        name,
        padded_vocab_size,
        num_embedding_partitions,
        hidden_size,
        layernorm_epsilon,
        multi_query_group_num,
        num_attention_heads,
        ffn_hidden_size,
        max_length,
        num_layers,
        layer_per_ipu,
    ):
        super().__init__(context, name)
        self.padded_vocab_size = padded_vocab_size
        self.num_embedding_partitions = num_embedding_partitions
        self.hidden_size = hidden_size
        self.layernorm_epsilon = layernorm_epsilon
        self.multi_query_group_num = multi_query_group_num
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.layer_per_ipu = layer_per_ipu
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.embedding = TransformerEmbedding(
            self.context,
            "embedding.word_embeddings",
            self.padded_vocab_size,
            self.hidden_size,
        )
        self.blocks = [
            ChatGLM2Block(
                self.context,
                "encoder.layers." + str(i),
                self.hidden_size,
                self.layernorm_epsilon,
                self.multi_query_group_num,
                self.num_attention_heads,
                self.ffn_hidden_size,
                self.max_length,
                layer_number=i + 1,
            )
            for i in range(self.num_layers)
        ]
        self.layer_norm = RMSLayerNorm(
            self.context,
            "encoder.final_layernorm",
            self.hidden_size,
            self.layernorm_epsilon,
        )

    def __call__(
        self,
        graph,
        input_ids,
        position_ids,
        step,
        attention_mask,
        sequence_length,
        **kwargs,
    ):
        softmax_type = kwargs.get("softmax_type", "ce")
        with self.device_scope(graph, 0, 0):
            hidden_states = self.embedding(graph, input_ids, sequence_length)
        end_points = np.cumsum(self.layer_per_ipu)
        for i in range(self.num_layers):
            stage_offset = sum(i >= end_points)
            with self.device_scope(graph, stage_offset, stage_offset):
                hidden_states = self.blocks[i](
                    graph,
                    hidden_states,
                    sequence_length,
                    step,
                    attention_mask,
                    softmax_type,
                )
                self.logger.info(f"block {i} placed on IPU {stage_offset}")

        with self.device_scope(graph, stage_offset, stage_offset):
            hidden_states = self.layer_norm(
                graph, hidden_states
            )

        return hidden_states


class ChatGLM2DecModel(HFDecBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outline_blocks = kwargs.get("outline_blocks", False)
        self.num_embedding_partitions = kwargs.get("num_embedding_partitions", 1)
        self.transformer = Transformer(
            None,
            "transformer",
            self.hf_config.padded_vocab_size,
            self.num_embedding_partitions,
            self.hf_config.hidden_size,
            self.hf_config.layernorm_epsilon,
            self.hf_config.multi_query_group_num,
            self.hf_config.num_attention_heads,
            self.hf_config.ffn_hidden_size,
            self.max_length,
            self.hf_config.num_layers,
            self.layer_per_ipu,
        )
        self.lm_head = LMHead(
            None,
            "transformer.output_layer",
            1,
            self.hf_config.padded_vocab_size,
            self.hf_config.hidden_size,
        )

    def prepare_state_dict(self):
        hf_args = {
            "pretrained_model_name_or_path": self.hf_model_name,
            "trust_remote_code": True,
            "cache_dir": self.hf_cache_dir,
        }
        self.hf_tokenizer = AutoTokenizer.from_pretrained(**hf_args)
        self.logger.info(f"initialized tokenizer by model_name: {self.hf_model_name}")
        self.hf_config = AutoConfig.from_pretrained(**hf_args)
        # self.hf_config.num_layers = 4
        self.logger.info(f"loading pretrained hf model: {self.hf_model_name}")
        self.hf_model = AutoModel.from_pretrained(**hf_args).eval()
        if self.precision != "fp32":
            self.hf_model.half()
            self.logger.info(f"casting model to {self.precision}")
        self.state_dict = self.hf_model.state_dict()
        tensor_names = list(self.state_dict.keys())
        for k in tensor_names:
            # Split qkv weight into q & kv for the need of TP mode
            if "query_key_value" in k:
                weight_np = self.state_dict.pop(k)
                weight_1, weight_2 = np.split(
                    weight_np, [self.hf_config.hidden_size], axis=0
                )
                self.state_dict[k.replace("query_key_value", "query")] = weight_1
                self.state_dict[k.replace("query_key_value", "key_value")] = weight_2
            if "dense_h_to_4h" in k:
                weight_np = self.state_dict.pop(k)
                weight_1, weight_2 = weight_np.chunk(2, dim=0)
                self.state_dict[
                    k.replace("dense_h_to_4h", "dense_h_to_4h_1")
                ] = weight_1
                self.state_dict[
                    k.replace("dense_h_to_4h", "dense_h_to_4h_2")
                ] = weight_2
        REGISTRY.register("query_size", self.hf_config.hidden_size)
        REGISTRY.register(
            "key_value_size",
            self.hf_config.multi_query_group_num * self.hf_config.kv_channels,
        )
        # Rearange weights for tp mode
        if self.model_type == "tp":
            for k in self.state_dict.keys():
                if "query" in k or ("dense.weight" in k and "scale" not in k):
                    self.state_dict[k] = rearrange_tp_weights(
                        self.state_dict[k],
                        self.hf_config.num_attention_heads,
                        num_replicas=self.num_replicas,
                        axis=0 if "query" in k else 1,
                    )
        self.register_state_dict()

    def build_model_graph(self, model_graph, model_graph_inputs, sequence_length=1):
        input_ids_container = model_graph_inputs["input_ids_container"]
        attention_mask = model_graph_inputs["attention_mask"]
        step = model_graph_inputs["step"]

        with self.device_scope(model_graph, 0, 0):
            input_ids = ops.dynamic_slice(
                model_graph,
                input_ids_container,
                step,
                axes=[1],
                sizes=[sequence_length],
            )
            temp_attention_mask = ops.unsqueeze(model_graph, attention_mask, [1, 2])
            if sequence_length != 1:
                position_ids_value = np.array(
                    [np.arange(self.max_length)] * self.batch_size
                )
                position_ids_container = ops.constant(
                    model_graph, position_ids_value.astype(np.int32), "position_ids"
                )
                position_ids = ops.dynamic_slice(
                    model_graph, position_ids_container, step, [1], [sequence_length]
                )
            else:
                position_ids = ops.unsqueeze(model_graph, step, [0])

        logits = self.transformer(
            model_graph,
            input_ids,
            position_ids,
            step,
            temp_attention_mask,
            sequence_length=sequence_length,
            outline_blocks=self.outline_blocks,
        )

        with self.device_scope(
            model_graph, len(self.layer_per_ipu) - 1, len(self.layer_per_ipu) - 1
        ):
            self.lm_head.set_virtual_id(len(self.layer_per_ipu) - 1)
            next_ids = self.lm_head(
                model_graph, logits, sequence_length=sequence_length
            )

        model_outputs = {
            "next_ids": next_ids,
            "stage_offset": self.transformer.num_layers + 1,
        }
        return model_outputs

    def build_input_dict(self, **kwargs):
        query = kwargs.get("input_string", "晚上睡不着应该怎么办")
        prompt = self.hf_tokenizer.build_prompt(query, history=[])
        inputs = self.hf_tokenizer([prompt], return_tensors="np")
        input_ids = inputs["input_ids"]
        input_ids_padding = np.zeros((1, self.max_length), dtype=np.int32)
        input_ids_padding[:, : input_ids.shape[1]] = input_ids

        return {"input_ids": input_ids_padding}

    def build_output_dict(self, anchor_arrays):
        output, decode_step = anchor_arrays["Loop:0"], anchor_arrays["Loop:1"]
        if len(output.shape) == 3:
            output, decode_step = output[0], decode_step[0]
        output = self.hf_tokenizer.batch_decode(output, skip_special_tokens=True)
        output = {str(i): v for i, v in enumerate(output)}
        return {"output": output, "decode_step": decode_step}
