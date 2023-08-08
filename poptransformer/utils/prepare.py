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

import logging
import popart
from hydra.utils import instantiate
from .registry import REGISTRY
from .tensor_type import TensorType


def register_global_args(config):
    global_args = instantiate(config.global_args)
    for key, value in global_args.items():
        REGISTRY.register(key, value)
    tensor_type = TensorType(global_args.precision)
    REGISTRY.register('tensor_type', tensor_type)

def register_config_logger(config):
    popart.getLogger().setLevel(config.popart_log_level.upper())
    logger = logging.getLogger('poptransformer')
    logger.setLevel(config.log_level.upper())
    REGISTRY.register('logger', logger)

def prepare_model_session(config):
    register_global_args(config)
    register_config_logger(config)
    model = instantiate(config.model)
    session = instantiate(config.session)
    model.build_graph()
    model.graph.saveInitializersExternally(
        model.initializers,
        'saved_initializer.onnx'
    )
    session.compile(model)
    return session, model
