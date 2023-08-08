# PopTransformer ChatGLM Model

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopTransformer | NLP | ChatGLM | N/A | Text generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br>| Min. 16 IPUs (POD16) required | - |


ChatGLM-6B contains 6.2 billion parameters and is an open bilingual language model developed by the Data Mining Research Group at Tsinghua University (THUDM). ChatGLM is based on the General Language Model framework.

For more details about the ChatGLM model, see the [model card](https://huggingface.co/THUDM/chatglm-6b) on Hugging Face.

The PopTransformer ChatGLM model shows how you can run ChatGLM-6B on IPU-PODs or Bow Pods with very low latency.

## Hardware

This model only runs on IPU-PODs or Bow Pods.

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer for these systems.

## Environment setup

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer.

## Running the model

You can run the ChatGLM PopTransformer model with a sharding or tensor-parallel configuration.

- Run the sharding model:
```
python inference.py --config-name=sharding
```

- Run the tensor-parallel model:
```
python inference.py --config-name=tp
```

- Run the sharding model with **INT4** precision:
```
python inference.py --config-name=int4_sharding
```

## License

Refer to the [top-level README](../../README.md#licenses) for licensing details.
