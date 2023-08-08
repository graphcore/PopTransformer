# PopTransformer RWKV model

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PopTransformer | NLP | RWKV | N/A | Text generation | <p style="text-align: center;">❌ <br>| <p style="text-align: center;">✅ <br>| Min. 8 IPUs required<br> or Min: 8 C600 cards required | - |

The RWKV model has a Recurrent Neural Network architecture but can be trained as a transformer. For details about the RWKV model, see the [model card](https://huggingface.co/RWKV/rwkv-raven-1b5) on Hugging Face.

The PopTransformer RWKV model shows how you can run RWKV-1.5B on C600s, IPU-PODs or Bow Pods with very low latency.

## Hardware

This model runs on C600s, IPU-PODs and Bow Pods.

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer for these systems.

## Environment setup

Refer to the [top-level README](../../README.md#environment-setup) for details of how to install PopTransformer.

## Running the model

You can run the RWKV PopTransformer model with a sharding or tensor-parallel configuration.

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
