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

from contextlib import ExitStack


class DeviceScope:
    def __init__(self, graph, virtual_graph_id=None, pipeline_stage_id=None, enable_pipeline=False, outline_attr=None):
        self.graph = graph
        self.virtual_graph_id = virtual_graph_id
        self.pipeline_stage_id = pipeline_stage_id
        self.enable_pipeline = enable_pipeline
        self.outline_attr = outline_attr

    def __enter__(self):
        self.stack = ExitStack()
        if self.virtual_graph_id is not None:
            self.stack.enter_context(self.graph.virtualGraph(self.virtual_graph_id))
        if self.pipeline_stage_id is not None and self.enable_pipeline:
            self.stack.enter_context(self.graph.pipelineStage(self.pipeline_stage_id))
        if self.outline_attr is not None:
            self.stack.enter_context(self.graph.outlineAttributes(self.outline_attr))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()
        return False
