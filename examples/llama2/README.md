# PopTransformer Llama 2 model

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopTransformer | NLP | Llama 2 | N/A | Text generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br>| Min. 64 IPUs (POD64) required | - |

Llama 2 is the second generation large language model from Meta and Microsoft. For details about Llama 2, see the [model card](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) on Hugging Face.

The PopTransformer Llama 2 model shows how you can run Llama 2 (13B parameters) on IPU-PODs or Bow Pods with very low latency.

## Hardware

This model only runs on IPU-PODs or Bow Pods.

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer for these systems.

## Environment setup

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer.

## Weights

To get the weights, you have to request access to Llama 2 on Hugging Face. See the [model card](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) for how to get access.

# Running the model

You can run the Llama 2 PopTransformer model with a sharding or tensor-parallel configuration.

- Run the sharding model:
```
python inference.py --config-name=sharding
```

- Run the tensor-parallel model:
```
python inference.py --config-name=tp
```


## License

Refer to the [top-level README](../../README.md#licenses) for licensing details.
