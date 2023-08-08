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
from poptransformer.layers import Linear

from poptransformer.layers import BaseLayer


class BaseMLP(BaseLayer):
    def __init__(self, context, name, hidden_size, ffn_hidden_size):
        super().__init__(context, name)
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.act_fn = ops.swish
        self.collect_bind_layer_weights()

    def collect_bind_layer_weights(self):
        self.gate_proj = Linear(
            self.context,
            "dense_h_to_4h_1",
            self.hidden_size,
            self.ffn_hidden_size,
            use_bias=False,
        )
        self.up_proj = Linear(
            self.context,
            "dense_h_to_4h_2",
            self.hidden_size,
            self.ffn_hidden_size,
            use_bias=False,
        )
        self.down_proj = Linear(
            self.context,
            "dense_4h_to_h",
            self.ffn_hidden_size,
            self.hidden_size,
            use_bias=False,
        )

    def __call__(self, graph, x):
        x = ops.reshape(graph, x, [-1, self.hidden_size])
        gate_output = self.gate_proj(graph, x)
        gate_output = self.act_fn(graph, gate_output)
        up_output = self.up_proj(graph, x)
        up_output = ops.mul(graph, up_output, gate_output)
        output = self.down_proj(graph, up_output)
        output = ops.reshape(graph, output, [self.batch_size, -1, self.hidden_size])
        return output


class TPMLP(BaseMLP):
    def collect_bind_layer_weights(self):
        gate_tp_settings = {
            "strategy_name": "start",
        }
        up_proj_tp_settings = {
            "strategy_name": "start",
        }
        down_proj_tp_settings = {
            "strategy_name": "end",
        }
        self.gate_proj = Linear(
            self.context,
            "dense_h_to_4h_1",
            self.hidden_size,
            self.ffn_hidden_size,
            use_bias=False,
            **gate_tp_settings,
        )
        self.up_proj = Linear(
            self.context,
            "dense_h_to_4h_2",
            self.hidden_size,
            self.ffn_hidden_size,
            use_bias=False,
            **up_proj_tp_settings,
        )
        self.down_proj = Linear(
            self.context,
            "dense_4h_to_h",
            self.ffn_hidden_size,
            self.hidden_size,
            use_bias=False,
            **down_proj_tp_settings,
        )


class MLP(TPMLP, BaseMLP):
    layer_class_map = {"tp": TPMLP, "shard": BaseMLP}

    def __init__(self, context, name, hidden_size, ffn_hidden_size):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}")
        self.logger.debug(f"initializing model type: {self.layer_class.__name__}")
        super().__init__(context, name, hidden_size, ffn_hidden_size)

    def __call__(self, graph, x):
        return self.layer_class.__call__(self, graph, x)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
