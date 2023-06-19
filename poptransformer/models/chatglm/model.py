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

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel

from poptransformer import ops
from poptransformer.layers import LayerNorm, BaseLayer, MLP
from poptransformer.models import HFDec2stageBaseModel
from poptransformer.models.chatglm.embedding import ChatGLMEmbedding
from poptransformer.models.chatglm.attention import RotaryAttention
from poptransformer.models.chatglm.lm_head import LMHead


class ChatGLMBlock(BaseLayer):
    def __init__(
        self,
        context,
        name,
        input_size,
        eps,
        num_attention_heads,
        num_layers,
        max_length,
        layer_index,
    ):
        super().__init__(context, name)
        self.input_size = input_size
        self.eps = eps
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length
        self.num_layers = num_layers
        self.layer_index = layer_index
        self.rotary_dim = input_size // (num_attention_heads * 2)
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.layer_norm1 = LayerNorm(
            self.context, "input_layernorm", self.input_size, self.eps
        )
        self.attention = RotaryAttention(
            self.context,
            "attention",
            self.input_size,
            self.num_attention_heads,
            self.max_length,
            self.layer_index,
            self.rotary_dim,
        )
        self.layer_norm2 = LayerNorm(
            self.context, "post_attention_layernorm", self.input_size, self.eps
        )
        self.mlp = MLP(
            self.context, "mlp", self.input_size, self.input_size * 4, "gelu"
        )

    def __call__(
        self,
        graph,
        x,
        layer_past,
        position_ids,
        block_position_ids,
        sequence_length,
        step,
        attention_mask,
        norm_type="ce",
        softmax_type="ce",
        **kwargs,
    ):
        matmul_kwargs = {
            "amp": 0.2 if sequence_length > 32 else 0.6,
            "partialtype": "half" if sequence_length > 1 else "float"
        }
        with graph.nameScope(self.context):
            attention_input = self.layer_norm1(graph, x, sequence_length, norm_type)
            attention_output, layer_present = self.attention(
                graph,
                attention_input,
                layer_past,
                position_ids,
                block_position_ids,
                step,
                attention_mask,
                sequence_length,
                softmax_type,
                **matmul_kwargs
            )
            alpha = ops.constant(
                graph,
                np.array([(2.0 * self.num_layers) ** 0.5], dtype=np.float16),
            )
            temp_x = ops.add(
                graph, ops.mul(graph, attention_input, alpha), attention_output
            )
            mlp_input = self.layer_norm2(graph, temp_x, sequence_length, norm_type)
            mlp_output = self.mlp(graph, mlp_input, **matmul_kwargs)
            output = ops.add(graph, ops.mul(graph, mlp_input, alpha), mlp_output)
        return output, layer_present


class Transformer(BaseLayer):
    def __init__(
        self,
        context,
        name,
        vocab_size,
        hidden_size,
        eps,
        num_attention_heads,
        max_length,
        num_layers,
        layer_per_ipu,
        max_position,
        num_embedding_partitions=4,
    ):
        super().__init__(context, name)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_embedding_partitions = num_embedding_partitions
        self.eps = eps
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length
        self.num_layers = num_layers
        self.layer_per_ipu = layer_per_ipu
        self.max_position = max_position
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.embedding = ChatGLMEmbedding(
            self.context,
            None,
            self.vocab_size,
            self.hidden_size,
            self.num_embedding_partitions,
        )
        self.blocks = [
            ChatGLMBlock(
                self.context,
                "layers." + str(layer_index),
                self.hidden_size,
                self.eps,
                self.num_attention_heads,
                self.num_layers,
                self.max_length,
                layer_index,
            )
            for layer_index in range(self.num_layers)
        ]
        self.layer_norm = LayerNorm(
            self.context, "final_layernorm", self.hidden_size, self.eps
        )

    def __call__(
        self,
        graph,
        input_ids,
        layer_past_list,
        position_ids,
        block_position_ids,
        step,
        attention_mask,
        sequence_length,
        **kwargs,
    ):
        norm_type = kwargs.get("norm_type", "ce")
        softmax_type = kwargs.get("softmax_type", "ce")
        return_last = kwargs.get("return_last", True)
        outline_blocks = kwargs.get("outline_blocks", "single_block")
        if outline_blocks:
            self.logger.info("outlining transformer blocks")
            self.logger.info(
                "please make sure disable outlining in session options is set to False"
            )
        hidden_states = self.embedding(graph, input_ids)
        end_points = np.cumsum(self.layer_per_ipu)
        layer_present_list = []
        for i in range(self.num_layers):
            stage_offset = sum(i >= end_points)
            with graph.virtualGraph(stage_offset):
                if outline_blocks == "single_block":
                    with graph.outlineAttributes({"block": "sub_" + str(i)}):
                        hidden_states, present = self.blocks[i](
                            graph,
                            hidden_states,
                            layer_past_list[i],
                            position_ids,
                            block_position_ids,
                            sequence_length,
                            step,
                            attention_mask,
                            norm_type,
                            softmax_type,
                        )
                else:
                    hidden_states, present = self.blocks[i](
                        graph,
                        hidden_states,
                        layer_past_list[i],
                        position_ids,
                        block_position_ids,
                        sequence_length,
                        step,
                        attention_mask,
                        norm_type,
                        softmax_type,
                    )
                self.logger.info(f"block {i} placed on IPU {stage_offset}")
                layer_present_list.append(present)

        with graph.virtualGraph(stage_offset):
            hidden_states = self.layer_norm(
                graph, hidden_states, sequence_length, norm_type
            )
            if return_last:
                hidden_states = ops.static_slice(
                    graph, hidden_states, [sequence_length - 1], [sequence_length], [1]
                )
                hidden_states = ops.squeeze(graph, hidden_states, [1])
        return hidden_states, layer_present_list


class ChatGLMDecModel(HFDec2stageBaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outline_blocks = kwargs.get("outline_blocks", True)
        self.num_embedding_partitions = kwargs.get("num_embedding_partitions", 4)
        self.transformer = Transformer(
            None,
            "transformer",
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            self.hf_config.layernorm_epsilon,
            self.hf_config.num_attention_heads,
            self.max_length,
            self.hf_config.num_layers,
            self.layer_per_ipu,
            self.hf_config.max_sequence_length,
            self.num_embedding_partitions,
        )
        self.lm_head = LMHead(
            None,
            "lm_head",
            self.hf_config.vocab_size,
            self.transformer.embedding.num_embedding_partitions,
            self.transformer.embedding.token_offsets,
            self.transformer.embedding.embedding_pad_mask,
        )

    def prepare_state_dict(self):
        hf_args = {
            "pretrained_model_name_or_path": self.hf_model_name,
            "trust_remote_code": True,
            "revision": "v1.1.0",
            "cache_dir": self.hf_cache_dir,
        }
        self.hf_tokenizer = AutoTokenizer.from_pretrained(**hf_args)
        self.logger.info(f"initialized tokenizer by model_name: {self.hf_model_name}")
        self.hf_config = AutoConfig.from_pretrained(**hf_args)
        # self.hf_config.num_layers = 4
        self.logger.info(f"loading pretrained hf model: {self.hf_model_name}")
        self.hf_model = AutoModel.from_pretrained(**hf_args).half().eval()
        self.state_dict = self.hf_model.state_dict()
        tensor_names = list(self.state_dict.keys())
        for k in tensor_names:
            v = self.state_dict[k]
            if ".attention" in k or "mlp" in k:
                if "weight" in k:
                    self.state_dict[k] = v.permute(1, 0)
            if "dense_h_to_4h" in k:
                self.state_dict[
                    k.replace("dense_h_to_4h", "c_fc")
                ] = self.state_dict.pop(k)
            if "dense_4h_to_h" in k:
                self.state_dict[
                    k.replace("dense_4h_to_h", "c_proj")
                ] = self.state_dict.pop(k)
        self.register_state_dict()

    def build_model_graph_1st(
        self,
        input_ids_container,
        step,
        attention_mask,
        position_ids,
        init_past_list,
        sequence_length=1,
    ):
        model_graph = self.graph

        with model_graph.virtualGraph(0):
            input_ids = ops.static_slice(
                model_graph,
                input_ids_container,
                starts=[0],
                ends=[sequence_length],
                axes=[1],
            )
            # stage 1: block_position_ids = [0, 0, ..., 0]
            block_position_ids = ops.constant(
                model_graph,
                np.zeros((self.batch_size, sequence_length)).astype(np.int32),
                "block_position_ids",
            )
        hidden_states, layer_present_list = self.transformer(
            model_graph,
            input_ids,
            init_past_list,
            position_ids,
            block_position_ids,
            step,
            attention_mask,
            sequence_length=sequence_length,
            outline_blocks=self.outline_blocks,
        )
        with model_graph.virtualGraph(0):
            next_ids = self.lm_head(
                model_graph, hidden_states, self.transformer.embedding.weight_ids
            )

        with model_graph.virtualGraph(0):
            next_input_ids = ops.constant(
                model_graph,
                np.ones((self.batch_size, 1), dtype=np.int32)
                * self.hf_tokenizer.bos_token_id,
                "decode_start_id",
            )
            # next_ids is useless for stage 2, use sub and add to avoid being pruned
            next_input_ids = ops.add(model_graph, next_input_ids, next_ids)
            next_input_ids = ops.sub(model_graph, next_input_ids, next_ids)
            step_add_value = ops.constant(
                model_graph, np.array(self.input_length).astype(np.int32), "1"
            )
            next_step = ops.add(model_graph, step, step_add_value)

            next_iput_ids_container = ops.dynamic_update(
                model_graph,
                input_ids_container,
                next_step,
                next_input_ids,
                axes=[1],
                sizes=[1],
            )

        return (
            next_iput_ids_container,
            next_step,
            position_ids,
            layer_present_list,
        )

    def build_model_graph_2nd(
        self,
        input_ids_container,
        step,
        attention_mask,
        stop_mask,
        position_ids,
        layer_past_list,
        sequence_length=1,
    ):
        model_graph = self.graph.createSubgraphBuilder()
        model_graph.setGraphName("model_graph")
        # add inputs for loop op
        self.add_input_tensor(model_graph, "BOOL", [], "cond_place_holder")
        self.add_input_tensor(model_graph, "INT32", [], "max_loop_place_holder")
        # add inputs for graph
        input_ids_container = model_graph.addUntypedInputTensor(input_ids_container)
        step = model_graph.addUntypedInputTensor(step)
        attention_mask = model_graph.addUntypedInputTensor(attention_mask)
        stop_mask = model_graph.addUntypedInputTensor(stop_mask)
        position_ids = model_graph.addUntypedInputTensor(position_ids)
        layer_past_list = [model_graph.addUntypedInputTensor(i) for i in layer_past_list]

        with model_graph.virtualGraph(0):
            input_ids = ops.dynamic_slice(
                model_graph,
                input_ids_container,
                step,
                axes=[1],
                sizes=[sequence_length],
            )
            # stage 2: block_position_ids = [step-input_length+1]
            constant_pos = ops.constant(
                model_graph,
                np.array(self.input_length - 1).astype(np.int32),
                "block_pos_constant",
            )
            block_position_ids = ops.sub(model_graph, step, constant_pos)

        hidden_states, layer_present_list = self.transformer(
            model_graph,
            input_ids,
            layer_past_list,
            position_ids,
            block_position_ids,
            step,
            attention_mask,
            sequence_length=sequence_length,
            outline_blocks=self.outline_blocks,
        )
        with model_graph.virtualGraph(0):
            next_ids = self.lm_head(
                model_graph, hidden_states, self.transformer.embedding.weight_ids
            )

        with model_graph.virtualGraph(0):
            next_input_ids = next_ids
            step_add_value = ops.constant(
                model_graph, np.array(1).astype(np.int32), "1"
            )
            next_step = ops.add(model_graph, step, step_add_value)

            next_iput_ids_container = ops.dynamic_update(
                model_graph,
                input_ids_container,
                next_step,
                next_input_ids,
                axes=[1],
                sizes=[1],
            )

            attention_mask_updater = ops.constant(
                model_graph,
                np.zeros((self.batch_size, 1, 1, 1), dtype=self.np_float_type),
                "attention_mask_updater",
            )
            # next_attention_mask = [0] + attention_mask_container[:-1]
            attention_mask = ops.static_slice(
                model_graph, attention_mask, starts=[0], ends=[-1], axes=[3]
            )
            next_attention_mask = ops.concat(
                model_graph, attention_mask_updater, attention_mask, axis=3
            )

            next_stop_mask, keep_going_cond = self.step_loop_cond(
                model_graph, next_input_ids, stop_mask
            )

        output_list = [
            keep_going_cond,
            next_iput_ids_container,
            next_step,
            next_attention_mask,
            next_stop_mask,
            position_ids,
        ]

        self.add_output_tensor(
            model_graph,
            output_list + layer_present_list,
        )
        return model_graph

    def build_input_dict(self, **kwargs):
        query = kwargs.get("input_string", "晚上睡不着怎么办")
        use_history = kwargs.get("use_history", False)
        global_round_count = 0
        if use_history:
            prompt = f"[Round {global_round_count}]\n问：{query}\n答："
            round_count = 0
            for i in range(len(history) - 1, -1, -1):
                old_query, response = history[i]
                history_text = f"[Round {i}]\n问：{old_query}\n答：{response}\n"
                prompt_length = self.hf_tokenizer(
                    history_text + prompt, return_tensors="pt"
                )["input_ids"].shape[1]
                if prompt_length <= self.input_length + 1:  # don't count '130001'
                    prompt = history_text + prompt
                    round_count += 1
                else:
                    break
            history = history[-round_count:]
        else:
            prompt = query

        inputs = self.hf_tokenizer(
            prompt, return_tensors="pt", max_length=self.input_length
        )

        input_ids = inputs["input_ids"][..., :-1]
        input_ids_container = torch.zeros(1, self.max_length).int()
        input_ids_container[:, : input_ids.size(1)] = input_ids[0]
        input_ids_container = input_ids_container.repeat(self.batch_size, 1)

        position_ids = inputs["position_ids"][:, 0, :-1]
        last_position_id = inputs["position_ids"][:, 0, -1]
        position_ids_container = torch.zeros(1, self.max_length).int()
        position_ids_container[:, : position_ids.size(1)] = position_ids[0]
        position_ids_container[:, position_ids.size(1) :] = last_position_id
        position_ids_container = position_ids_container.repeat(self.batch_size, 1)

        attention_mask_container = torch.zeros(1, self.max_length).int()
        # where 1 is masked while 0 is not
        attention_mask_container[:, : input_ids.size(1)] = 1
        attention_mask_container = attention_mask_container.repeat(self.batch_size, 1)

        return {
            "input_ids": input_ids_container.numpy(),
            "position_ids": position_ids_container.numpy(),
            "attention_mask": attention_mask_container.numpy(),
        }

    def build_output_dict(self, anchor_arrays):
        def process_response(response):
            response = response.strip()
            response = response.replace("[[训练时间]]", "2023年")
            punkts = [
                [",", "，"],
                ["!", "！"],
                [":", "："],
                [";", "；"],
                ["\?", "？"],
            ]
            for item in punkts:
                response = re.sub(
                    r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response
                )
                response = re.sub(
                    r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response
                )
            return response

        output, decode_step = (
            anchor_arrays["Loop:0"][0].tolist(),
            anchor_arrays["Loop:1"],
        )

        output = output[output.index(self.hf_config.bos_token_id) :]
        if self.hf_config.eos_token_id in output:
            output = output[: output.index(self.hf_config.eos_token_id)]
        output = self.hf_tokenizer.decode(output)
        output = process_response(output)

        return {"output": output, "decode_step": decode_step}
