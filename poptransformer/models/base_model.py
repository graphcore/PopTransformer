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
from collections import OrderedDict
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM
import numpy as np
import popart
from poptransformer.utils import REGISTRY, DeviceScope, OptionsScope
from poptransformer import ops


class BaseModel:

    graph = REGISTRY.get('main_graph')
    logger = REGISTRY.get('logger')
    tensor_type = REGISTRY.get('tensor_type')

    def __init__(self, **kwargs):
        self.logger.info(f'Initializing model class: {self.__class__.__name__}')
        self.anchor_return_type = kwargs.get('anchor_return_type', 'ALL')
        self.layer_per_ipu = kwargs.get('layer_per_ipu', [])

    @property
    def enable_pipeline(self):
        return REGISTRY.get('enable_pipeline')

    @property
    def batch_size(self):
        return REGISTRY.get('batch_size')

    @property
    def batch_per_step(self):
        return REGISTRY.get('batch_per_step')

    @property
    def model_type(self):
        return REGISTRY.get('model_type')

    @property
    def num_replicas(self):
        return REGISTRY.get('num_replicas')

    @property
    def stage_num(self):
        return max(len(self.layer_per_ipu), 1)

    @property
    def popart_float_type(self):
        return REGISTRY.get('tensor_type').popart_float_type

    @property
    def np_float_type(self):
        return REGISTRY.get('tensor_type').np_float_type

    @property
    def precision(self):
        return REGISTRY.get('tensor_type').precision

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

    def device_scope(self, graph, virtual_graph_id=None, pipeline_stage_id=None, outline_attr=None):
        return DeviceScope(graph, virtual_graph_id, pipeline_stage_id, self.enable_pipeline, outline_attr)

    def option_scope(self, amp=None, partialtype=None, serial_factor=None, serial_mode=None):
        return OptionsScope(amp, partialtype, serial_factor, serial_mode)

    def register_state_dict(self):
        # register the state dict to REGISTER, will be used in layer
        assert self.state_dict
        REGISTRY.update('state_dict', self.state_dict)

    @property
    def initializers(self):
        # all weights' id added into the graph
        return [tensor_id for tensor_id in self.graph.getInputTensorIds()
                if self.graph.isInitializer(tensor_id)]

    @property
    def model_output(self):
        output_tensor_ids = self.get_output_tensor_ids(self.graph)
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
        if isinstance(tensor_id, str):
            graph.addInputTensorFromParentGraph(tensor_id)
        elif isinstance(tensor_id, list):
            for i in tensor_id:
                self.add_input_tensor_from_parent_graph(graph, i)
        else:
            raise ValueError(f"Invalid tensor_id: {tensor_id}")

    def add_untyped_input_tensor(self, graph, tensor_id):
        if isinstance(tensor_id, list):
            all_tensor_ids = []
            for i in tensor_id:
                all_tensor_ids.append(graph.addUntypedInputTensor(i))
            return all_tensor_ids
        if isinstance(tensor_id, str):
            return graph.addUntypedInputTensor(tensor_id)
        raise ValueError(f"Invalid tensor_id: {tensor_id}")

    def add_output_tensor(self, graph, output):
        if isinstance(output, list):
            for i in output:
                graph.addOutputTensor(i)
        elif isinstance(output, str):
            graph.addOutputTensor(output)
        else:
            raise ValueError(f"Invalid output: {output}")

    def get_input_tensor_ids(self, graph):
        return [tensor_id for tensor_id in graph.getInputTensorIds()
                if not graph.isInitializer(tensor_id)]

    def get_output_tensor_ids(self, graph):
        return graph.getOutputTensorIds()

    def create_sub_graph(self, graph, name, sub_graph_inputs, is_loop_graph=False):
        sub_graph = graph.createSubgraphBuilder()
        sub_graph.setGraphName(name)
        if is_loop_graph:
            self.add_input_tensor(sub_graph, 'INT32', [], 'max_loop')
            self.add_input_tensor(sub_graph, 'BOOL', [], 'cond')
        self.add_input_tensor_from_parent_graph(sub_graph, sub_graph_inputs)
        return sub_graph


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
            if self.precision != 'fp32':
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
    def build_model_graph(self, model_graph, model_graph_inputs, sequence_length=1):
        # TODO: Add params docstring.
        # return model graph
        pass

    def build_default_inputs(self):
        # build input tensor here, no need to feed to step io if not used.
        default_inputs = OrderedDict()
        default_inputs['input_ids_container'] = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'input_ids')
        default_inputs['step'] = self.add_input_tensor(
            self.graph, 'INT32', [], 'step')
        default_inputs['attention_mask'] = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'attention_mask')
        default_inputs['position_ids'] = self.add_input_tensor(
            self.graph, 'INT32', [self.batch_size, self.max_length], 'position_ids')
        default_inputs['stop_mask'] = self.add_input_tensor(
            self.graph, 'INT32', [], 'stop_mask')
        return default_inputs

    def build_model_graph_inputs(self, graph, inputs):
        with graph.nameScope('mask'):
            attention_mask_value = [[0] + [-10000 for i in range(self.max_length - 1)]] * self.batch_size
            attention_mask_value = np.array(attention_mask_value).astype(self.np_float_type)
            attention_mask = ops.constant(
                graph, attention_mask_value, 'attention_mask')
            inputs['attention_mask'] = attention_mask
        with graph.nameScope('step'):
            inputs['step'] = ops.constant(self.graph, np.array(0).astype(np.int32), 'init_step')
        with graph.nameScope('stop_mask'):
            inputs['stop_mask'] = ops.constant(
                self.graph, np.zeros((self.batch_size,)).astype(np.int32), 'stop_mask')
        del inputs['position_ids']
        return inputs

    def build_graph(self):
        default_inputs = self.build_default_inputs()
        with self.graph.virtualGraph(0):
            model_graph_inputs = self.build_model_graph_inputs(self.graph, default_inputs)
        model_graph = self.create_sub_graph(
            self.graph,
            'model_graph',
            list(model_graph_inputs.values()),
            True
        )
        model_outputs = self.build_model_graph(model_graph, model_graph_inputs, sequence_length=1)
        self.build_post_model_graph(model_graph, model_graph_inputs, model_outputs)
        with self.device_scope(self.graph, 0):
            outputs = ops.loop(
                self.graph,
                self.max_loop,
                list(model_graph_inputs.values()),
                model_graph
            )[:2]
        self.add_output_tensor(self.graph, outputs)


    def build_post_model_graph(self, model_graph, model_graph_inputs, model_outputs):
        # continue build model graph for post inference step
        stage_offset = model_outputs['stage_offset']
        next_ids = model_outputs['next_ids']
        attention_mask = model_graph_inputs['attention_mask']
        step = model_graph_inputs['step']
        input_ids_container = model_graph_inputs['input_ids_container']
        stop_mask = model_graph_inputs['stop_mask']

        with self.device_scope(model_graph, 0, pipeline_stage_id=stage_offset):
            next_iput_ids_container, next_step, next_attention_mask, id_to_update= self.step_containers(
                model_graph, input_ids_container, step, attention_mask, next_ids
            )
            next_stop_mask, keep_going_cond = self.step_loop_cond(model_graph, id_to_update, stop_mask)

        self.add_output_tensor(
            model_graph,
            [keep_going_cond, next_iput_ids_container, next_step, next_attention_mask, next_stop_mask])
        # build model graph for post inference step

    def step_containers(self, graph, input_ids_container, step, attention_mask, next_ids):
        with graph.nameScope('step_containers'):
            if self.hf_tokenizer.pad_token_id:
                pad_token_id = self.hf_tokenizer.pad_token_id
            else:
                pad_token_id = self.hf_config.pad_token_id
            step_add_value = ops.constant(graph, np.array(1).astype(np.int32), '1')
            next_step = ops.add(graph, step, step_add_value)
            if attention_mask is not None:
                attention_mask_add = ops.constant(
                    graph, np.zeros((self.batch_size, 1), dtype=self.np_float_type), 'attention_mask_add')
                next_attention_mask = ops.dynamic_update(
                    graph, attention_mask, next_step, attention_mask_add, axes=[1], sizes=[1])
            else:
                next_attention_mask = None
            input_ids_slice = ops.dynamic_slice(
                graph, input_ids_container, next_step, axes=[1], sizes=[1])
            pad_id = ops.constant(graph, np.array(pad_token_id), 'pad_id')
            id_update_cond = ops.equal(graph, input_ids_slice, pad_id)
            id_to_update = ops.where(graph, id_update_cond, next_ids, input_ids_slice)
            next_iput_ids_container = ops.dynamic_update(
                graph, input_ids_container, next_step, id_to_update, axes=[1], sizes=[1])
        return next_iput_ids_container, next_step, next_attention_mask, id_to_update

    def step_loop_cond(self, graph, id_to_update, stop_mask):
        with graph.nameScope('step_loop_cond'):
            if self.hf_tokenizer.eos_token_id:
                eos_token_id = self.hf_tokenizer.eos_token_id
            else:
                eos_token_id = self.hf_config.eos_token_id
            temp_id_to_update = ops.squeeze(graph, id_to_update, [1])
            eos_id = ops.constant(graph, np.array(eos_token_id), 'eos_id')
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
            self.hf_config.num_attention_heads // self.num_replicas,
            self.max_length,
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
        init_past_list = [
            ops.constant(
                self.graph,
                np.zeros(cache_shape).astype(self.np_float_type),
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
        with self.device_scope(self.graph, 0, 0):
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
