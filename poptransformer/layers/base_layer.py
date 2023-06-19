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
import numpy as np
from popart import VariableRetrievalMode, VariableSettings, CommGroupType, CommGroup
from poptransformer.utils import REGISTRY


class BaseLayer:
    # class attributes which can be used by child classes
    logger = REGISTRY.get('logger')
    vs_type_map = {
        'consecutive': CommGroupType.Consecutive,
        'all': CommGroupType.All,
        'orthogonal': CommGroupType.Orthogonal}
    retrieval_mode_map = {
        'one_per_group': VariableRetrievalMode.OnePerGroup,
        'all_replicas': VariableRetrievalMode.AllReplicas}

    def __init__(self, context, name):
        self.context = '.'.join([context, name]) if (context and name) else context or name
        self.name = name

    @property
    def model_type(self):
        return REGISTRY.get('model_type')

    @property
    def batch_size(self):
        return REGISTRY.get('batch_size')

    @property
    def num_replicas(self):
        return REGISTRY.get('num_replicas')

    @property
    def main_graph(self):
        return REGISTRY.get('main_graph')

    @property
    def state_dict(self):
        return REGISTRY.get('state_dict')

    @property
    def popart_float_type(self):
        return REGISTRY.get('popart_float_type')

    @property
    def np_float_type(self):
        return REGISTRY.get('np_float_type')

    @property
    def precision(self):
        return REGISTRY.get('precision')

    @abstractmethod
    def __call__(self, graph, *args):
        # build the graph with weights / layers built from the collect_bind_layer_weights
        pass

    @abstractmethod
    def collect_bind_layer_weights(self):
        # build/process the weight / layers
        # should be exectute in the __init__ fn, we may have to find a way to exectute it automatically
        pass

    def build_variable_setting(self, vs_type='consecutive', group_size=1, retrieval_mode='one_per_group'):
        vs_type = self.vs_type_map.get(vs_type, None)
        retrieval_mode = self.retrieval_mode_map.get(retrieval_mode, None)
        assert vs_type, f"Invalid vs_type: {vs_type}"
        assert retrieval_mode, f"Invalid retrieval_mode: {retrieval_mode}"
        variableSettings = VariableSettings(
            CommGroup(vs_type, replicaGroupSize=group_size),
            retrieval_mode)
        return variableSettings

    def add_initialized_input_tensor(self, weight_np, weight_key, **vs_setting_kwargs):
        if vs_setting_kwargs:
            variable_setting = self.build_variable_setting(**vs_setting_kwargs)
            weight_id = self.main_graph.addInitializedInputTensor(weight_np, variable_setting, weight_key)
        else:
            weight_id = self.main_graph.addInitializedInputTensor(weight_np, debugContext=weight_key)
        return weight_id

    def get_param_from_state_dict(self, weight_key, shape_list):
        weight_np = self.state_dict.get(weight_key, None)
        if weight_np is None:
            self.logger.debug(f'{weight_key} not found, using random tensor')
            assert shape_list, f'  no shape provided for random initializing the tensor {weight_key}'
            weight_np = np.random.randn(*shape_list).astype(dtype=self.np_float_type)
        else:
            weight_np = weight_np.numpy()
            if shape_list:
                assert sum(
                    (abs(a - b) for a, b in zip(weight_np.shape, shape_list))
                    ) == 0, f'{weight_key} shape {weight_np.shape} not matched with provided {shape_list}'
            else:
                self.logger.warning(f'shape_list not provided, skip shape check for {weight_key}')
            self.logger.debug(f'  using param {weight_key} from state_dict')

        return weight_np

    def get_add_param_from_state_dict(self, weight_key, shape_list, **vs_setting_kwargs):
        weight_np = self.get_param_from_state_dict(weight_key, shape_list)
        weight_id = self.add_initialized_input_tensor(weight_np, weight_key, **vs_setting_kwargs)
        return weight_id
