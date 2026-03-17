#!/bin/bash

# 用法:
# bash evaluate.sh <model_path> <exp_name>
#
# 示例:
# bash evaluate.sh ./output_dir/local_sft_industrial/final_checkpoint sft_only
# bash evaluate.sh /aisphere/data/liuxuanzhen/server_rl_industrial_baseline_small/final_checkpoint rl_baseline
# bash evaluate.sh /aisphere/data/liuxuanzhen/server_rl_industrial_value_small/final_checkpoint rl_value

category="Industrial_and_Scientific"

exp_name="$1"
exp_name_clean="$2"

if [[ -z "$exp_name" || -z "$exp_name_clean" ]]; then
    echo "Usage: bash evaluate.sh <model_path> <exp_name>"
    exit 1
fi

echo "Processing category: $category with model path: $exp_name"
echo "Experiment name: $exp_name_clean"

train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

if [[ ! -f "$test_file" ]]; then
    echo "Error: Test file not found for category $category"
    exit 1
fi

if [[ ! -f "$info_file" ]]; then
    echo "Error: Info file not found for category $category"
    exit 1
fi

temp_dir="./temp/${category}-${exp_name_clean}"
echo "Creating temp directory: $temp_dir"
mkdir -p "$temp_dir"

echo "Splitting test data..."
python ./split.py \
    --input_path "$test_file" \
    --output_path "$temp_dir" \
    --cuda_list "0,1"

if [[ ! -f "$temp_dir/0.csv" ]]; then
    echo "Error: Data splitting failed for category $category"
    exit 1
fi

cudalist="0 1"
echo "Starting parallel evaluation..."
for i in ${cudalist}
do
    if [[ -f "$temp_dir/${i}.csv" ]]; then
        echo "Starting evaluation on GPU $i for category ${category}"
        CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
            --base_model "$exp_name" \
            --info_file "$info_file" \
            --category ${category} \
            --test_data_path "$temp_dir/${i}.csv" \
            --result_json_data "$temp_dir/${i}.json" \
            --batch_size 8 \
            --num_beams 50 \
            --max_new_tokens 256 \
            --length_penalty 0.0 &
    else
        echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
    fi
done

echo "Waiting for all evaluation processes to complete..."
wait

result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
if [[ $result_files -eq 0 ]]; then
    echo "Error: No result files generated for category $category"
    exit 1
fi

output_dir="./results/${exp_name_clean}"
echo "Creating output directory: $output_dir"
mkdir -p "$output_dir"

actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
echo "Merging results from GPUs: $actual_cuda_list"

python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "$output_dir/final_result_${category}.json" \
    --cuda_list "$actual_cuda_list"

if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
    echo "Error: Result merging failed for category $category"
    exit 1
fi

echo "Calculating metrics..."
python ./calc.py \
    --path "$output_dir/final_result_${category}.json" \
    --item_path "$info_file"

echo "Completed processing for category: $category"
echo "Results saved to: $output_dir/final_result_${category}.json"
echo "----------------------------------------"