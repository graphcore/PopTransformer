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

from .registry import REGISTRY


class OptionsScope:

    def __init__(self, amp=None, partialtype=None, serial_factor=None, serial_mode=None):
        self.amp = amp
        self.partialtype = partialtype
        self.serial_factor = serial_factor
        self.serial_mode = serial_mode

        self.default_amp = None
        self.default_partialtype = None
        self.default_serial_factor = None
        self.default_serial_mode = 'none'

    def __enter__(self):
        if self.amp is not None:
            self.default_amp = REGISTRY.get('amp')
            REGISTRY.update('amp', self.amp)
            REGISTRY.get('logger').debug(f'using amp: {self.amp}')
        if self.partialtype is not None:
            self.default_partialtype = REGISTRY.get('partialtype')
            REGISTRY.update('partialtype', self.partialtype)
            REGISTRY.get('logger').debug(f'using partialtype: {self.partialtype}')
        if self.serial_factor is not None:
            self.default_serial_factor = REGISTRY.get('serial_factor')
            self.default_serial_mode = REGISTRY.get('serial_mode')
            REGISTRY.update('serial_factor', self.serial_factor)
            REGISTRY.update('serial_mode', self.serial_mode)
            REGISTRY.get('logger').debug(f'using serialize: {self.serial_factor}, mode: {self.serial_mode}')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.default_amp != self.amp:
            REGISTRY.update('amp', self.default_amp)
            REGISTRY.get('logger').debug(f'exit and using default_amp: {self.default_amp}')
        if self.default_partialtype != self.partialtype:
            REGISTRY.update('partialtype', self.default_partialtype)
            REGISTRY.get('logger').debug(f'exit and using default_partialtype: {self.default_partialtype}')
        if self.default_serial_factor != self.serial_factor:
            REGISTRY.update('serial_factor', self.default_serial_factor)
            REGISTRY.update('serial_mode', self.default_serial_mode)
            REGISTRY.get('logger').debug(f'exit and using default_serial_factor: {self.default_serial_factor}')
            REGISTRY.get('logger').debug(f'exit and using serial_mode: {self.serial_mode}')
        return False
