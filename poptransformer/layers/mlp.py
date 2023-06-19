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
from poptransformer.utils import shard, repeat
from .base_layer import BaseLayer
from .linear import Linear


class BaseMLP(BaseLayer):
    act_fn_map = {'gelu': ops.gelu}
    def __init__(self, context, name, input_size, hidden_size, act_fn):
        super().__init__(context, name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_fn = self.act_fn_map.get(act_fn, None)
        if not self.act_fn:
            raise ValueError(f"Invalid act_fn {act_fn}, options: {self.act_fn_map.keys()}")
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.c_fc = Linear(self.context, 'c_fc', self.input_size, self.hidden_size)
        self.c_proj = Linear(self.context, 'c_proj', self.hidden_size, self.input_size)

    def __call__(self, graph, x):
        with graph.nameScope(self.context):
            x = ops.reshape(graph, x, [-1, self.input_size])
            x = self.c_fc(graph, x, **self.matmul_kwargs)
            x = self.act_fn(graph, x)
            x = self.c_proj(graph, x, **self.matmul_kwargs)
            x = ops.reshape(graph, x, [self.batch_size, -1, self.input_size])
            return x


class TPMLP(BaseMLP):

    def collect_bind_layer_weights(self):
        vs_setting = {'vs_type': 'consecutive', 'group_size': 1}
        c_fc_weight_key = '.'.join([self.context, 'c_fc.weight'])
        c_fc_weight_np = self.get_param_from_state_dict(c_fc_weight_key, [self.input_size, self.hidden_size])
        c_fc_bias_key = '.'.join([self.context, 'c_fc.bias'])
        c_fc_bias_np = self.get_param_from_state_dict(c_fc_bias_key, [self.hidden_size])
        c_fc_weight_np = shard(c_fc_weight_np, self.num_replicas, axis=-1)
        c_fc_bias_np = shard(c_fc_bias_np, self.num_replicas, axis=-1)
        self.c_fc_weight_id = self.add_initialized_input_tensor(c_fc_weight_np, c_fc_weight_key, **vs_setting)
        self.c_fc_bias_id = self.add_initialized_input_tensor(c_fc_bias_np, c_fc_bias_key, **vs_setting)

        c_proj_weight_key = '.'.join([self.context, 'c_proj.weight'])
        c_proj_weight_np = self.get_param_from_state_dict(c_proj_weight_key, [self.hidden_size, self.input_size])
        c_proj_bias_key = '.'.join([self.context, 'c_proj.bias'])
        c_proj_bias_np = self.get_param_from_state_dict(c_proj_bias_key, [self.input_size])
        c_proj_weight_np = shard(c_proj_weight_np, self.num_replicas, axis=0)
        c_proj_bias_np = repeat(c_proj_bias_np, self.num_replicas, axis=0)
        self.c_proj_weight_id = self.add_initialized_input_tensor(c_proj_weight_np, c_proj_weight_key, **vs_setting)
        self.c_proj_bias_id = self.add_initialized_input_tensor(c_proj_bias_np, c_proj_bias_key, **vs_setting)

    def __call__(self, graph, x):
        with graph.nameScope(self.context):
            x = ops.reshape(graph, x, [-1, self.input_size])
            x = ops.matmul(graph, x, self.c_fc_weight_id)
            x = ops.add(graph, x, self.c_fc_bias_id)
            x = self.act_fn(graph, x)
            x = ops.matmul(graph, x, self.c_proj_weight_id)
            x = graph.aiGraphcore.replicatedallreduce([x])
            x = ops.add(graph, x, self.c_proj_bias_id)
            x = ops.reshape(graph, x, [self.batch_size, -1, self.input_size])
            return x


class MLP(TPMLP, BaseMLP):
    layer_class_map = {
        'tp': TPMLP,
        'shard': BaseMLP}

    def __init__(self, context, name, input_size, hidden_size, act_fn):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, input_size, hidden_size, act_fn)

    def __call__(self, graph, x, **matmul_kwargs):
        self.matmul_kwargs = matmul_kwargs
        return self.layer_class.__call__(self, graph, x)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
