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


class TensorType:

    all_precision = ['fp32', 'fp16', 'int4', 'fp8', 'fp8_weight']
    np_map = {'fp32': np.float32, 'fp16': np.float16, 'int4': np.float16, 'fp8': np.float16, 'fp8_weight': np.float16}
    popart_map = {'fp32': 'FLOAT', 'fp16': 'FLOAT16', 'int4': 'FLOAT16', 'fp8': 'FLOAT16', 'fp8_weight': 'FLOAT16'}

    def __init__(self, precision):
        assert precision in self.all_precision
        self.precision = precision

    @property
    def np_float_type(self):
        return self.np_map[self.precision]

    @property
    def popart_float_type(self):
        return self.popart_map[self.precision]
