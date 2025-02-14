#  accelerate launch --config_file config.yaml --main_process_port=8886 train.py --train_args_file train_args/sft/qlora/qwen-14b-sft-qlora.json
# CUDA_VISIBLE_DEVICES=1 python train.py --train_args_file train_args/sft/qlora/llama2-13b-sft-qlora.json 
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/sft/qlora/qwen-14b-sft-qlora.json
# accelerate launch --config_file config.yaml --main_process_port=8886 train.py --train_args_file train_args/sft/qlora/qwen-14b-sft-qlora.json 