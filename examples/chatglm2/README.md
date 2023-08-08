# PopTransformer ChatGLM2 Model

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopTransformer | NLP | ChatGLM2 | N/A | Text generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br>| Min. 16 IPUs (POD16) required | - |

ChatGLM2-6B is the second generation of ChatGLM-6B, the open bilingual language model developed by the Data Mining Research Group at Tsinghua University (THUDM). For details about the ChatGLM2 model including the new features, see the [model card](https://huggingface.co/THUDM/chatglm2-6b) on Hugging Face.

The PopTransformer ChatGLM2 model shows how you can run ChatGLM2-6B on IPU-PODs or Bow Pods with very low latency.

## Hardware

This model only runs on IPU-PODs or Bow Pods.

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer for these systems.

## Environment setup

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer.

## Running the model

You can run the ChatGLM2 PopTransformer model with a sharding or tensor-parallel configuration.

- Run the sharding model:
```
python inference.py --config-name=sharding
```

- Run the tensor-parallel model with **INT4** precision:
```
python inference.py --config-name=tp
```

## License

Refer to the [top-level README](../../README.md#licenses) for licensing details.
