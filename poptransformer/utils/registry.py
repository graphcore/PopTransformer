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

import popart


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, key, value):
        if key in self._registry:
            raise KeyError(f"Duplicate key: {key}")
        self._registry[key] = value

    def update(self, key, value):
        try:
            self._registry[key] = value
        except KeyError:
            raise KeyError(f"key: {key} not found, please register first")

    def get(self, key):
        try:
            return self._registry[key]
        except KeyError:
            raise KeyError(f'key: {key} not found')


REGISTRY = Registry()
REGISTRY.register('main_graph', popart.Builder(opsets={'ai.onnx': 10, 'ai.graphcore': 1}))
# main graph is to be built for certain, we build it here, move to other place if there be more comments
