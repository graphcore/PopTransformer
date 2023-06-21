# PopTransformer
The repository provides a fundamental framework(including layers, operators, models, etc) that allow users to develop and run highly optimized transformer-based models(inference-only) with [Poplar SDK](https://docs.graphcore.ai/en/latest/software.html) on [Graphcore IPU](https://www.graphcore.ai/). Currently, we provide examples like GPT2 and ChatGLM, and more LLM models will be supported in the future.

## Install

Follow the below steps to setup the development environment on C600:
1. (Optional) Create Python virtual env
2. Enable Poplar SDK(all models are tested with SDK `3.2.0`)
```
source [path-to-sdk]/enable.sh
```
3. Run `make` to compile custom ops
4. Run `pip install -r requirements.txt`

If you are using M2000 or BOW-M2000, we provide a script to run in a docker container by:
```
bash docker/setup_container.sh
```

## Quick start
```
cd examples/gpt2
python inference.py --config-name='sharding' log_level='info' model.topk=5
```

## How to build new model
Please read [our documents](https://docs.graphcore.ai/en/latest/index.html) if you're not familiar with Poplar SDK.
1. Preparation.
Prepare a new working folder under `example`, and a sub folder `conf` to place config files. The files tree looks like:
```
├── examples
│   ├── chatglm
│   │   ├── conf
│   │   └── inference.py
│   ├── gpt2
│   │    ├── conf
│   │    └── inference.py
│   └── your model
│       ├── conf
│       └── inference.py
```
Also to create an entry file like `examples/gpt2/inference.py`. We use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) to initialize classes from yaml.

2. Implement layers
- Inherite the base layer class `BaseLayer`.
- Override the `collect_bind_layer_weights` with `get_param_from_state_dict` to load weights from pretrained model, and bind to main graph with `add_initialized_input_tensor`.
- Override `__call__` function.
- Build TP layer for tensor parallel implementation if you need.
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
Example see: `poptransformer/models/gpt2/model.py`
- Inherite the base model class(`HFDecBaseModel`)
- Override the funcitons you need
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
If you have done all the above and the tests, you can simply run from the entry files `examples/your_model/inference.py` by:
```
cd examples/your_model
python inference.py
```


## Benchmark
[GPT2-XL](examples/gpt2/benchmark.md)

[ChatGLM-6B](examples/chatglm/benchmark.md)

## Model Licenses

Use of the pre-trained weights is subject to the following licenses.

* GPT2-XL: GPT-2 is licensed under a [Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE)
* ChatGLM: [ChatGLM-6B model license](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)
