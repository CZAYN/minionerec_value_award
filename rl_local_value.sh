#!/bin/bash

export NCCL_IB_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com

category="Industrial_and_Scientific"

train_file=./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.small.csv
eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

model_path=./output_dir/local_sft_industrial/final_checkpoint
output_dir=/aisphere/data/liuxuanzhen/server_rl_industrial_value_small

echo "category: ${category}"
echo "train_file: ${train_file}"
echo "eval_file: ${eval_file}"
echo "info_file: ${info_file}"
echo "model_path: ${model_path}"
echo "output_dir: ${output_dir}"

accelerate launch \
  --num_processes 1 \
  --main_process_port 29513 \
  rl.py \
  --model_path ${model_path} \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 4 \
  --train_file ${train_file} \
  --eval_file ${eval_file} \
  --info_file ${info_file} \
  --category ${category} \
  --sample_train False \
  --eval_step 0.5 \
  --reward_type value_ranking \
  --num_generations 2 \
  --mask_all_zero False \
  --dynamic_sampling False \
  --sync_ref_model False \
  --beam_search False \
  --test_during_training False \
  --temperature 1.0 \
  --learning_rate 1e-5 \
  --add_gt False \
  --beta 1e-3 \
  --dapo False \
  --output_dir ${output_dir} \
  --wandb_project minionerec_local \
  --wandb_run_name rl_industrial_l20_value_small \
  --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
  --item_value_path ./data/Amazon/index/Industrial_and_Scientific.item_value.json