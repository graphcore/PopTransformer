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

import numpy as np
from poptransformer import ops
from poptransformer.layers.layer_norm import BaseLayerNorm


class BaseRMSLayerNorm(BaseLayerNorm):

    def collect_bind_layer_weights(self):
        weight_key = '.'.join([self.context, 'weight'])
        weight_np = self.get_param_from_state_dict(weight_key, [self.input_size])
        self.weight_id = self.add_initialized_input_tensor(weight_np, weight_key)

    def __call__(self, graph, x):
        variance_epsilon = ops.constant(graph, np.array(self.eps).astype(np.float32), 'variance_epsilon')
        variance = ops.cast(graph, x, 'FLOAT')
        variance = ops.mul(graph, variance, variance)
        variance = ops.reducemean(graph, variance)
        variance = ops.add(graph, variance, variance_epsilon)
        variance = ops.sqrt(graph, variance)
        variance = ops.reciprocal(graph, variance)
        variance = ops.cast(graph, variance, self.popart_float_type)
        x = ops.mul(graph, x, variance)
        return ops.mul(graph, x, self.weight_id)


class TPRMSLayerNorm(BaseRMSLayerNorm):
    pass


class RMSLayerNorm(TPRMSLayerNorm, BaseRMSLayerNorm):

    layer_class_map = {
        'tp': TPRMSLayerNorm,
        'shard': BaseRMSLayerNorm}

    def __init__(self, context, name, input_size, eps):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}, options: {self.layer_class_map.keys()}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, input_size, eps)

    def __call__(self, graph, x):
        return self.layer_class.__call__(self, graph, x)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
    