mkdir benchmark

for bs in 1 8 16 24 32
do 
    python inference.py model.batch_size=$bs model.hf_model_name='gpt2-xl' model.layer_per_ipu=[0,6,7,7,7,7,7,7] model.max_length=264 model.max_loop=264 > benchmark/log_ml264_$bs.log
done

