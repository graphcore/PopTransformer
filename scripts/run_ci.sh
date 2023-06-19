cd examples/gpt2;
python inference.py --config-name='sharding' log_level='info';
export POPLAR_TARGET_OPTIONS='{"ipuLinkTopology":"torus"}'; python inference.py --config-name='tp' log_level='info'; unset POPLAR_TARGET_OPTIONS