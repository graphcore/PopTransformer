# PopTransformer GPT2 model

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopTransformer | NLP | GPT2 | N/A | Text generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br>| Min. 8 IPUs required<br> or Min: 8 C600 cards required| - |

GPT2 is a large language model from OpenAI. GPT2 is capable of performing a large number of NLP tasks beyond text generation. For more details about the GPT2 model, see the [model card](https://huggingface.co/gpt2) on Hugging Face.

GPT2 has a variety of model sizes including GPT2 (124M parameters), GPT2-Medium (355M parameters), GPT2-Large (774M parameters) and GPT2-XL (1.5B parameters). We implemented GPT-XL but it's easy to port other models by changing the value of the `hf_model_name` parameter in the YAML configuration file in the `conf` folder.

The PopTransformer GPT2 model shows how you can run GPT2-XL on C600s, IPU-PODs or Bow Pods with very low latency.

## Hardware

This model runs on IPU-PODs, Bow Pods and C600s.

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer for these systems.

## Environment setup

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer.

## Running the model

You can run the GPT2 PopTransformer model with a sharding or tensor-parallel configuration.

- Run the sharding model:
```
python inference.py --config-name=sharding
```

- Run the tensor-parallel model:

Due to the constraint that the number of attention head need to be divided evenly by the power of 2 when running with tensor parallel. The script below uses GPT2-Medium instead.
```
python inference.py --config-name=tp
```


## License

Refer to the [top-level README](../../README.md#licenses) for licensing details.
