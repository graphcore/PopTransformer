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

from abc import abstractmethod
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM
import numpy as np
import popart
from poptransformer.utils import REGISTRY
from poptransformer import ops


class BaseModel:

    graph = REGISTRY.get('main_graph')
    logger = REGISTRY.get('logger')

    def __init__(self, **kwargs):
        self.logger.info(f'Initializing model class: {self.__class__.__name__}')
        self.batch_size = kwargs.get('batch_size', 1)
        self.model_type = kwargs.get('model_type', 'shard')
        self.num_replicas = kwargs.get('num_replicas', 1)
        self.precision = kwargs.get('precision', 'fp16')
        self.np_float_type = np.float32 if self.precision == 'fp32' else np.float16
        self.popart_float_type = 'FLOAT' if self.precision == 'fp32' else 'FLOAT16'
        self.anchor_return_type = kwargs.get('anchor_return_type', 'ALL')
        self.layer_per_ipu = kwargs.get('layer_per_ipu', [])
        REGISTRY.register('batch_size', self.batch_size)
        REGISTRY.register('precision', self.precision)
        REGISTRY.register('np_float_type', self.np_float_type)
        REGISTRY.register('popart_float_type', self.popart_float_type)
        REGISTRY.register('model_type', self.model_type)
        REGISTRY.register('num_replicas', self.num_replicas)

    @property
    def stage_num(self):
        return max(len(self.layer_per_ipu), 1)

    @abstractmethod
    def prepare_state_dict(self):
        # build self.state_dict
        # then it will be registed to REGISTER for layers usage
        pass

    @abstractmethod
    def build_graph(self):
        # build model's graph
        pass

    @abstractmethod
    def build_input_dict(self, **kwargs):
        # process input, build dict,
        # will be wrapped with stepio and feed to graph later
        pass

    @abstractmethod
    def build_output_dict(self, anchor_arrays):
        # process outputs in session.anchor_arrays,
        # return a dict for layer usage
        pass

    def register_state_dict(self):
        # register the state dict to REGISTER, will be used in layer
        assert self.state_dict
        REGISTRY.register('state_dict', self.state_dict)

    @property
    def initializers(self):
        # all weights' id added into the graph
        return [tensor_id for tensor_id in self.graph.getInputTensorIds()
                if self.graph.isInitializer(tensor_id)]

    @property
    def model_output(self):
        output_tensor_ids = self.graph.getOutputTensorIds()
        anchor_return_type = popart.AnchorReturnType(self.anchor_return_type)
        return {key: anchor_return_type for key in output_tensor_ids}

    @property
    def model_proto(self):
        return self.graph.getModelProto()

    def add_input_tensor(self, graph, popart_dtype, shape, name):
        if self.model_type in ['shard']:
            return graph.addInputTensor(
                popart.TensorInfo(popart_dtype, shape), debugContext=name)
        if self.model_type == 'tp':
            input_setting = popart.InputSettings(
                popart.ReplicatedStreamMode.Broadcast)
            return graph.addInputTensor(
                popart.TensorInfo(popart_dtype, shape), input_setting, debugContext=name)
        raise ValueError(f"Invalid model_type: {self.model_type}")

    def add_input_tensor_from_parent_graph(self, graph, tensor_id):
        if isinstance(tensor_id, list):
            for i in tensor_id:
                graph.addInputTensorFromParentGraph(i)
        elif isinstance(tensor_id, str):
            graph.addInputTensorFromParentGraph(tensor_id)
        else:
            raise ValueError(f"Invalid tensor_id: {tensor_id}")

    def add_output_tensor(self, graph, output):
        if isinstance(output, list):
            for i in output:
                graph.addOutputTensor(i)
        elif isinstance(output, str):
            graph.addOutputTensor(output)
        else:
            raise ValueError(f"Invalid output: {output}")


class HFBaseModel(BaseModel):
    """
    hugggingface base model,
    with the process fn for loading hf model, config and tokenizer
    """
    hf_model_class_name_map = {
        'auto_model': AutoModel,
        'auto_model_for_causallm': AutoModelForCausalLM}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hf_model_name = kwargs.get('hf_model_name', None)
        self.hf_model_class_name = kwargs.get('hf_model_class_name', None)
        self.override_hfconfig_from_json = kwargs.get('override_hfconfig_from_json', None)
        self.hf_cache_dir = kwargs.get('hf_cache_dir', './temp/')
        self.prepare_state_dict()

    def process_hf_model_state_dict(self):
        self.logger.info('no prescale process on hf state dict')

    def prepare_state_dict(self):
        assert self.hf_model_name, f"Invalid hf_model_name: {self.hf_model_name}"
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name,
            cache_dir=self.hf_cache_dir,
            pad_token='[PAD]'
        )
        self.logger.info(f'initialized tokenizer by model_name: {self.hf_model_name}')
        if not self.override_hfconfig_from_json:
            model_class = self.hf_model_class_name_map.get(self.hf_model_class_name, None)
            assert model_class, f"Invalid hf_model_class_name: {self.hf_model_class_name}"
            self.logger.info(f'initializing hf model class: {model_class.__name__}')
            self.hf_model = model_class.from_pretrained(self.hf_model_name, cache_dir=self.hf_cache_dir)
            self.logger.info(f'loading pretrained hf model: {self.hf_model_name}')
            self.hf_config = self.hf_model.config
            self.process_hf_model_state_dict()
            if self.precision == 'fp16':
                self.hf_model.half()
                self.logger.info(f'casting model to {self.precision}')
            self.state_dict = self.hf_model.state_dict()
        else:
            self.logger.info('using overrided config, no state dict loaded')
            self.hf_config = AutoConfig.from_pretrained(self.override_hfconfig_from_json)
            self.state_dict = {}    # No states if using your own model configurations.
        self.register_state_dict()


class HFDecBaseModel(HFBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.early_stop = kwargs.get('early_stop', False)
        self.max_length = kwargs.get('max_length', 128)
        max_loop = kwargs.get('max_loop', None)
        self.max_loop = max_loop if max_loop else self.max_length

    @abstractmethod
    def build_model_graph(
        self,
        input_ids_container,
        step,
        attention_mask,
        stop_mask,
        sequence_length,
    ):
        # return model graph
        pass

    def build_attention_mask(self, graph):
        attention_mask_value = [[0] + [-10000 for i in range(self.max_length - 1)]] * self.batch_size
        attention_mask_value = np.array(attention_mask_value).astype(self.np_float_type)
        attention_mask = ops.constant(
            graph, attention_mask_value, 'attention_mask')
        return attention_mask

    def build_graph(self):
        input_ids_container = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'input_ids'
        )
        attention_mask = self.build_attention_mask(self.graph)
        step = ops.constant(self.graph, np.array(0).astype(np.int32), 'init_step')
        stop_mask = ops.constant(
            self.graph, np.zeros((self.batch_size,)).astype(np.int32), 'stop_mask'
        )
        model_graph = self.build_model_graph(
            input_ids_container, step, attention_mask, stop_mask, sequence_length=1)

        loop_graph_input_list = [input_ids_container, step, attention_mask, stop_mask]

        with self.graph.virtualGraph(0):
            output_id, step = ops.loop(
                self.graph,
                self.max_loop,
                loop_graph_input_list,
                model_graph
            )[:2]
        self.add_output_tensor(self.graph, [output_id, step])

    def step_containers(self, graph, input_ids_container, step, attention_mask, next_ids):
        # TODO: Unify the order of inputs/outputs
        with graph.nameScope('step_containers'):
            step_add_value = ops.constant(graph, np.array(1).astype(np.int32), '1')
            next_step = ops.add(graph, step, step_add_value)
            attention_mask_add = ops.constant(
                graph, np.zeros((self.batch_size, 1), dtype=self.np_float_type), 'attention_mask_add')
            next_attention_mask = ops.dynamic_update(
                graph, attention_mask, next_step, attention_mask_add, axes=[1], sizes=[1])
            input_ids_slice = ops.dynamic_slice(
                graph, input_ids_container, next_step, axes=[1], sizes=[1])
            pad_id = ops.constant(graph, np.array(self.hf_tokenizer.pad_token_id), 'pad_id')
            id_update_cond = ops.equal(graph, input_ids_slice, pad_id)
            id_to_update = ops.where(graph, id_update_cond, next_ids, input_ids_slice)
            next_iput_ids_container = ops.dynamic_update(
                graph, input_ids_container, next_step, id_to_update, axes=[1], sizes=[1])
        return next_iput_ids_container, next_step, next_attention_mask, id_to_update

    def step_loop_cond(self, graph, id_to_update, stop_mask):
        with graph.nameScope('step_loop_cond'):
            temp_id_to_update = ops.squeeze(graph, id_to_update, [1])
            eos_id = ops.constant(graph, np.array(self.hf_tokenizer.eos_token_id), 'eos_id')
            current_stop_mask = ops.equal(graph, temp_id_to_update, eos_id)
            current_stop_mask = ops.cast(graph, current_stop_mask, 'INT32')
            next_stop_mask = ops.bitwiseor(graph, stop_mask, current_stop_mask)
            if self.early_stop:
                mask = ops.reduce_sum(graph, stop_mask, axes=[0], keepdims=False)
                batch_size_constant = ops.constant(
                    graph, np.array(self.batch_size).astype(np.int32), 'batch_size')
                keep_going_cond = ops.less(graph, mask, batch_size_constant)
            else:
                keep_going_cond = ops.constant(graph, np.array(True).astype(np.bool_), 'cond')
        return next_stop_mask, keep_going_cond


class HFDec2stageBaseModel(HFDecBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_length = kwargs.get('input_length', 32)
        max_loop = kwargs.get('max_loop', None)
        self.max_loop = max_loop if max_loop else self.max_length - self.input_length

    @abstractmethod
    def build_model_graph_1st(
        self,
        input_ids_container,
        step,
        attention_mask,
        position_ids,
        init_past_list,
        sequence_length,
    ):
        # return model graph
        pass

    @abstractmethod
    def build_model_graph_2nd(
        self,
        input_ids_container,
        step,
        attention_mask,
        stop_mask,
        position_ids,
        layer_past_list,
        sequence_length,
    ):
        # return model graph
        pass

    @abstractmethod
    def build_positions_ids(self, position_ids_container):
        pass

    def build_graph(self):
        input_ids_container = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'input_ids')
        position_ids_container = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'position_ids')
        attention_mask = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'attention_mask')
        step = ops.constant(self.graph, np.array(0).astype(np.int32), 'init_step')
        stop_mask = ops.constant(
            self.graph, np.zeros((self.batch_size,)).astype(np.int32), 'stop_mask')
        cache_shape = (
            2,
            self.batch_size,
            self.hf_config.num_attention_heads,
            self.max_length,
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
        init_past_list = [
            ops.constant(
                self.graph,
                np.zeros(cache_shape).astype(np.float16),
                f"init_past_{str(i)}",
            )
            for i in range(self.hf_config.num_layers)
        ]
        with self.graph.virtualGraph(0):
            position_ids_stage_1 = ops.static_slice(
                self.graph, position_ids_container, [0], [self.input_length], [1]
            )
            attention_mask = ops.sub(
                self.graph,
                attention_mask,
                ops.constant(self.graph, np.ones(1, dtype=np.int32), '1'),
            )
            attention_mask = ops.cast(self.graph, attention_mask, self.popart_float_type)
            attention_mask = ops.mul(
                self.graph,
                attention_mask,
                ops.constant(self.graph, np.ones(1, dtype=self.np_float_type)*10000, '10000')
            )
            attention_mask = ops.unsqueeze(self.graph, attention_mask, [1, 2])
            attention_mask_stage_1 = ops.static_slice(
                self.graph,
                attention_mask,
                starts=[0],
                ends=[self.input_length],
                axes=[3],
            )
        input_ids_container, step, _, layer_present_list = self.build_model_graph_1st(
            input_ids_container,
            step,
            attention_mask_stage_1,
            position_ids_stage_1,
            init_past_list,
            self.input_length
        )
        with self.graph.virtualGraph(0):
            position_ids_stage_2 = ops.static_slice(
                self.graph,
                position_ids_container,
                [self.input_length - 1],
                [self.input_length],
                [1],
            )
            attention_mask_updater = ops.constant(
                self.graph,
                np.zeros((self.batch_size, 1, 1, 1), dtype=self.np_float_type),
                "attention_mask_updater",
            )
            # next_attention_mask = [0] + attention_mask_container[:-1]
            attention_mask = ops.static_slice(
                self.graph, attention_mask, starts=[0], ends=[-1], axes=[3]
            )
            attention_mask_stage_2 = ops.concat(
                self.graph, attention_mask_updater, attention_mask, axis=3
            )

        stage2_graph = self.build_model_graph_2nd(
            input_ids_container,
            step,
            attention_mask_stage_2,
            stop_mask,
            position_ids_stage_2,
            layer_present_list,
            1
        )

        with self.graph.virtualGraph(0):
            output_id, step = ops.loop(
                self.graph,
                self.max_loop,
                [input_ids_container,
                 step, attention_mask_stage_2,
                 stop_mask,
                 position_ids_stage_2] + layer_present_list,
                stage2_graph
            )[:2]

        self.add_output_tensor(self.graph, [output_id, step])
