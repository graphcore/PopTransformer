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

from ..registry import REGISTRY
from .precision_strategy import PrecisionStrategyMap
from .tensor_parallel_strategy import TPStragetgyMap


class ParamHandler:

    def __init__(self, host_layer, tp_strategy_name='identity', **vs_setting):
        self.host_layer = host_layer
        self.tp_strategy = TPStragetgyMap[tp_strategy_name]
        self.precision_strategy = PrecisionStrategyMap[self.precision]
        self.vs_setting = vs_setting

    @property
    def num_replicas(self):
        return REGISTRY.get('num_replicas')

    @property
    def precision(self):
        return REGISTRY.get('precision')

    def process_linear_weight(self, weight_np, weight_key):
        weight_fn_tp = self.tp_strategy['weight_fn']
        weight_fn_precision = self.precision_strategy['weight_fn']

        weight_np = weight_fn_tp(weight_np, self.num_replicas, self.tp_strategy['weight_axis'])
        weight_np = weight_fn_precision(
            host_layer=self.host_layer,
            weight_np=weight_np,
            weight_key=weight_key,
            weight_fn_tp=weight_fn_tp,
            num_replicas=self.num_replicas,
            weight_axis=self.tp_strategy['weight_axis'],
            **self.vs_setting
        )
        return weight_np

    def process_linear_bias(self, bias):
        bias_fn_tp = self.tp_strategy['bias_fn']

        bias = bias_fn_tp(bias, self.num_replicas, 0)
        return bias

    def matmul(self, graph, x, weight):
        x, weight = self._prepare_matmul(graph, x, weight)
        y = self._matmul(graph, x, weight)
        y = self._post_process_matmul(graph, y)
        return y

    def _prepare_matmul(self, graph, x, weight):
        prepare_fn_tp = self.tp_strategy['prepare_matmul']
        prepare_fn_precision = self.precision_strategy['prepare_matmul']

        x, weight = prepare_fn_tp(graph, x, weight)
        x, weight = prepare_fn_precision(graph, x, weight)
        return x, weight

    def _post_process_matmul(self, graph, y):
        prepare_fn_tp = self.tp_strategy['post_process_matmul']
        prepare_fn_precision = self.precision_strategy['post_process_matmul']

        y = prepare_fn_tp(graph, y)
        y = prepare_fn_precision(graph, y)
        return y

    def _matmul(self, graph, x, weight):
        matmul_fn = self.precision_strategy['matmul_fn']

        y = matmul_fn(graph, x, weight)
        return y
