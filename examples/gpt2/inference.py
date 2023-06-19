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

import time
import logging
import sys
import os
import popart
import hydra
from hydra.utils import instantiate
sys.path.append('../../')
os.environ['HYDRA_FULL_ERROR'] = '1'
from poptransformer.utils import REGISTRY


def register_config_logger(config):
    popart.getLogger().setLevel(config.popart_log_level.upper())
    logger = logging.getLogger('poptransformer')
    logger.setLevel(config.log_level.upper())
    REGISTRY.register('logger', logger)

def prepare(config):
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

def run_inference(session, model, *args, **kwargs):
    input_dict = model.build_input_dict(*args, **kwargs)
    all_time = []
    for _ in range(10):
        session.run(input_dict)

    for _ in range(10):
        start = time.perf_counter()
        session.run(input_dict)
        end = time.perf_counter()
        all_time.append(end-start)

    output = model.build_output_dict(session.anchor_arrays)
    decode_step = output['decode_step'].item()
    latency = sum(all_time) / len(all_time)
    latency_per_token = latency / decode_step
    throughput = int(model.batch_size * model.max_length / latency)
    model.logger.info(output)
    performance = "\nPerformance: \n" + \
                f"batch size: {model.batch_size}, precision: {model.precision}\n" + \
                f"max_length: {model.max_length}, decode_step: {decode_step}\n" + \
                f"Latency per token: {latency_per_token*1000.0:.3f} ms/token \n" + \
                f"Throughput: {throughput} token/s, Total Latency: {latency*1000.0:.3f} ms\n"
    model.logger.info(performance)


@hydra.main(version_base=None, config_path="conf", config_name="sharding")
def main(config):
    session, model = prepare(config)
    run_inference(session, model, **config.inputs)

if __name__ == '__main__':
    main()
    