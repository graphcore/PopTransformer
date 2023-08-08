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

import math
import popart
from poptransformer.utils import REGISTRY


class Session:

    logger = REGISTRY.get('logger')

    def __init__(self, **kwargs):
        self.logger.info(f'initializing {self.__class__.__name__}')
        self.execution_cache_name = kwargs.get('execution_cache_name', None)
        self.disable_outlining = kwargs.get('disable_outlining', False)
        self.unstable_softmax = kwargs.get('unstable_softmax', False)
        self.disable_matmul_multi_stage_reduce = kwargs.get('disable_matmul_multi_stage_reduce', False)
        self.constant_folding_of_multiple_consumers = kwargs.get('constant_folding_of_multiple_consumers', False)
        self.use_loop_candidate_creator = kwargs.get('use_loop_candidate_creator', True)
        self.profile_name = kwargs.get('profile_name', None)
        self.enable_pipeline = REGISTRY.get('enable_pipeline')
        self.batch_per_step = REGISTRY.get('batch_per_step')

    def compile(self, model):
        self.model = model
        self._build_data_flow()
        self._build_device_info()
        self._build_options()
        self._build_inference_session()
        self.logger.info('compiling')
        self.session.prepareDevice()
        self.session.weightsFromHost()
        self._anchor_arrays = self.session.initAnchorArrays()
        self.logger.info('compiled')

    def run(self, input_dict):
        stepio = popart.PyStepIO(input_dict, self.anchor_arrays)
        self.session.run(stepio)

    def _build_data_flow(self):
        self.logger.info(f'building data flow, with batch_per_step: {self.batch_per_step}.')
        self.data_flow = popart.DataFlow(self.batch_per_step, self.model.model_output)

    def _build_device_info(self):
        self.logger.info('building device info')
        ipu_num = pow(2, math.ceil(math.log2(self.model.stage_num * self.model.num_replicas)))
        self.device_info = popart.DeviceManager().acquireAvailableDevice(ipu_num)
        self.logger.info(
            f'acquired {ipu_num} ipu with {self.model.stage_num} stage, {self.model.num_replicas} replica')

    def _build_options(self):
        self.logger.info('building session options')
        self.options = popart.SessionOptions()
        self.options.enableOutlining = not self.disable_outlining
        self.options.enableNonStableSoftmax = self.unstable_softmax
        self.options.enableReplicatedGraphs = self.model.num_replicas != 1
        self.options.replicatedGraphCount = self.model.num_replicas
        self.options.enableConstantFoldingOfMultipleConsumers = self.constant_folding_of_multiple_consumers
        self.options.useLoopCandidateCreator = self.use_loop_candidate_creator
        self.options.enablePipelining = self.enable_pipeline

        if self.disable_matmul_multi_stage_reduce:
            self.options.matmulOptions = {"enableMultiStageReduce": "false"}
        if self.model.stage_num != 1:
            self.options.virtualGraphMode = popart.VirtualGraphMode.Manual
            self.logger.info('setting virtual graph model to manual')
        if self.execution_cache_name:
            self.options.enableEngineCaching = True
            self.options.cachePath = self.execution_cache_name
            self.logger.info(f'saving execution cache in {self.execution_cache_name}')
        if self.profile_name:
            self.options.engineOptions = {
                "autoReport.all": "true",
                "autoReport.directory": f'profiles/{self.profile_name}',
                "autoReport.executionProfileProgramRunCount":"2",
                "debug.allowOutOfMemory": "true"
            }
            self.logger.info(f'saving profiles in profiles/{self.profile_name}')

    def _build_inference_session(self):
        self.logger.info('building inference session')
        self.session = popart.InferenceSession(
            fnModel=self.model.model_proto,
            deviceInfo=self.device_info,
            dataFlow=self.data_flow,
            userOptions=self.options
        )

    @property
    def anchor_arrays(self):
        return self._anchor_arrays
