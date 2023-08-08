# PopTransformer

PopTransformer is a framework that allows you to develop and run highly optimized transformer-based models -- for inference only -- with the [Poplar SDK](https://docs.graphcore.ai/projects/sdk-overview/) on [Graphcore IPUs](https://www.graphcore.ai/). PopTransformer includes layers, operators, and models.

## Updates

- [07/31/2023] Added support for LLMs, including ChatGLM2, Llama2 and RWKV.
- [07/31/2023] Added support for inference with FP8 and INT4.
- [07/31/2023] Code was refactored and enhanced to make it easier to implement models.

## Environment setup

To setup the development environment on the C600:

1. (Optional) Create a Python virtual environment.
2. Enable the Poplar SDK (all models are tested with SDK version 3.2.0):
```
source [path-to-sdk]/enable
```
3. Run `make` to compile custom ops.
4. Run `pip install -r requirements.txt` to install Python requirements.

If you are using IPU-PODs or Bow Pods, run the following script to setup a Docker container:
```
bash docker/setup_container.sh
```

## Quick start

The following shows how you can run a simple example from the `examples` directory:

```
cd examples/gpt2
python inference.py --config-name='sharding' log_level='info' model.topk=5
```

## How to build a new model

This section describes how to build a new model that you can run with PopTransformer.

### Background information

It is best if you are familiar with the following before starting to write a new model that uses PopTransformer:

* The the IPU architecture, programming model and tools available is described in the [IPU Programmer's Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/).
* The Poplar graph programming framework is described in the [Poplar and PopLibs User Guide](https://docs.graphcore.ai/projects/poplar-user-guide/).
* The Poplar Advanced Runtime (PopART) for importing and executing models using the ONNX format is described in the [PopART User Guide](https://docs.graphcore.ai/projects/popart-user-guide/).

1. Preparation.

1.1. Create a new directory in the `examples` directory for your model, for example `examples/your_model`. This directory will contain the running script and the configuration. Also, create a sub-directory called `conf` that will contain the configuration files, `examples/your_model/conf`.

1.2 Create an `inference.py` in the model directory. We use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) to initialize classes from a YAML configuration file.

The file tree for the `examples` directory should look like:
```
├── examples
│   ├── chatglm
│   │   ├── conf
│   │   └── inference.py
│   ├── gpt2
│   │    ├── conf
│   │    └── inference.py
│   └── [your model]
│       ├── conf
│       └── inference.py
```

1.3. Create a directory for your model in the `poptransformer/models` directory. This will contain the model implementation.

2. Implement layers

In `inference.py`, implement the following layers:

2.1. Inherit the base layer class, `BaseLayer`.
2.2. Override `collect_bind_layer_weights` with `get_param_from_state_dict` to load weights from a pre-trained model, and bind to the main graph with `add_initialized_input_tensor`.
2.3. Override the `__call__` function.
2.4. Build the tensor parallel layer (`TPCustomLayer`) for tensor parallel execution if needed.
```
class BaseCustomLayer(BaseLayer):
    def __init__(self):
        ...

    def collect_bind_layer_weights(self):
        # load weight and bind to graph here
        weight_np = self.get_param_from_state_dict(...)
        self.weight_id = self.add_initialized_input_tensor(weight_np)

    def __call__(self, graph, x):
        # build the inference process for this layer
        return ops.matmul(graph, x, self.weight_id)


class TPCustomLayer(BaseCustomLayer):
    ...

class CustomLayer(TPCustomLayer, BaseCustomLayer):
    ...
    def __init__(self, *args, **kwargs):
        # choose the parent layer you need by a parameter registered in REGISTRY
        self.layer_class = ...
        super().__init__(self, *args, **kwargs)

    def collect_bind_layer_weights(self):
        return self.layer_class.collect_bind_layer_weights(self)

    def __call__(self, x):
        # bind fn
        return self.layer_class.__call__(self, graph, x)

```


3. Implement model

Next, implement your model in `poptransformer/models/your_model`. See `poptransformer/models/gpt2/model.py` for an example. Refer to the the [PopART User Guide](https://docs.graphcore.ai/projects/popart-user-guide/) for more information on the API.

3.1. Inherit the base model class: `HFDecBaseModel` or `HFDec2stageBaseModel`.
3.2. Override functions if needed.
```
class GPTDecModel(HFDecBaseModel):
    def __init__(self, **kwargs):
        ...
    def build_model_graph(self):
        # build your model graph here
        ...
    def build_input_dict(self, **kwargs):
        # build the processing input fn for your model
        ...
    def build_output_dict(self, anchor_arrays):
        # build the output processing fn for your model
        ...
```

4. Test and run

Write tests for your model, for example to allow you to compare the results using PopTransformer with other frameworks, like PyTorch or TensorFlow.

If you have done all the above and written tests, then you can simply run PopTransformer from the entry file `examples/your_model/inference.py` by:
```
cd examples/your_model
python inference.py
```


## Licenses
The content of this repository is licensed under the [Apache License, Version 2.0](LICENSE) except for the [model code for Llama 2](poptransformer/models/llama2) which is licensed under the [Llama 2 Community License Agreement](poptransformer/models/llama2/LICENSE).

Use of the pre-trained weights is subject to the following licenses:

- GPT2-XL: GPT-2 is licensed under a [Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE)
- ChatGLM: [ChatGLM-6B model license](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)
- RWKV: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) as specified in the [model card for the original weights on Hugging Face](https://huggingface.co/BlinkDL/rwkv-4-raven). PopTransformer requires weights in the Hugging Face format, which are available in the [RWKV Space on Hugging Face](https://huggingface.co/RWKV). According to the [Hugging Face Terms of Service](https://huggingface.co/terms-of-service), the converted weights are also licensed under Apache-2.0.
- ChatGLM2: [ChatGLM2-6B model license](https://github.com/THUDM/ChatGLM2-6B/blob/main/README_EN.md#license)
- Llama 2: [Llama 2 Community License Agreement](https://github.com/facebookresearch/llama/blob/main/LICENSE)
