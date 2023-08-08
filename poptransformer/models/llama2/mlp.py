# Copyright (c) 2023 Graphcore Ltd.
# This is a re-implementation of Llama 2 by Graphcore Ltd
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from poptransformer import ops
from poptransformer.layers import Linear
from poptransformer.layers.mlp import BaseMLP


class ShardMLP(BaseMLP):
    act_fn_map = {
        'gelu': ops.gelu,
        'swish': ops.swish
    }
    def collect_bind_layer_weights(self):
        self.gate_proj = Linear(self.context, 'gate_proj', self.input_size, self.hidden_size, use_bias=False)
        self.up_proj = Linear(self.context, 'up_proj', self.input_size, self.hidden_size, use_bias=False)
        self.down_proj = Linear(self.context, 'down_proj', self.hidden_size, self.input_size, use_bias=False)

    def __call__(self, graph, x):
        x = ops.reshape(graph, x, [-1, self.input_size])
        gate_output = self.gate_proj(graph, x)
        gate_output = self.act_fn(graph, gate_output)
        up_output = self.up_proj(graph, x)
        up_output = ops.mul(graph, up_output, gate_output)
        output = self.down_proj(graph, up_output)
        output = ops.reshape(graph, output, [self.batch_size, -1, self.input_size])
        return output


class TPMLP(ShardMLP):

    def collect_bind_layer_weights(self):
        gate_tp_settings = {
            'strategy_name': 'start',
        }
        up_proj_tp_settings = {
            'strategy_name': 'start',
        }
        down_proj_tp_settings = {
            'strategy_name': 'end',
        }
        self.gate_proj = Linear(
            self.context, 'gate_proj', self.input_size, self.hidden_size, use_bias=False, **gate_tp_settings)
        self.up_proj = Linear(
            self.context, 'up_proj', self.input_size, self.hidden_size, use_bias=False, **up_proj_tp_settings)
        self.down_proj = Linear(
            self.context, 'down_proj', self.hidden_size, self.input_size, use_bias=False, **down_proj_tp_settings)


class MLP(TPMLP, ShardMLP):
    layer_class_map = {
        'tp': TPMLP,
        'shard': ShardMLP}

    def __init__(self, context, name, input_size, hidden_size, act_fn):
        model_type = self.model_type
        self.layer_class = self.layer_class_map.get(model_type, None)
        if not self.layer_class:
            raise ValueError(f"Invalid model_type {model_type}")
        self.logger.debug(f'initializing model type: {self.layer_class.__name__}')
        super().__init__(context, name, input_size, hidden_size, act_fn)

    def __call__(self, graph, x):
        return self.layer_class.__call__(self, graph, x)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)
