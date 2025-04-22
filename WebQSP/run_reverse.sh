#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

export DATA_DIR=data/WebQSP/generation/merged
export MODEL_DIR="your model path"
export CUDA_VISIBLE_DEVICES=1
ACTION=${1:-none}
dataset="WebQSP"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/reverse_${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_reverse.sh "${exp_prefix}run_reverse.sh"
    #git rev-parse HEAD > "${exp_prefix}commitid.log"

    nohup python -u WebQSP/run_reverser.py \
        --dataset ${dataset} \
        --model_type Flan-t5-large \
        --model_name_or_path ${MODEL_DIR}/Flan-t5-large \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --train_file ${DATA_DIR}/${dataset}_mask_ptrain.json \
        --predict_file ${DATA_DIR}/${dataset}_mask_pdev.json \
        --learning_rate 3e-5 \
        --evaluation_strategy steps \
        --num_train_epochs 20 \
        --logging_steps 1000 \
        --eval_steps 1000 \
        --save_strategy steps \
        --save_steps 1000 \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --save_total_limit 2 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --eval_beams 10 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 >> "${exp_prefix}train_log.txt" 2>&1 &

elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-dev}
    log="results/reverse/${dataset}_${split}/evaluation_beam_llama"
    mkdir -p ${log}

    nohup python -u WebQSP/run_reverser.py \
        --dataset ${dataset} \
        --model_type llama \
        --model_name_or_path ${model} \
        --eval_beams 15 \
        --do_predict \
        --predict_file results/gen/WebQSP_test/evaluation_beam_llama/beam_test_top_k_predictions.json \
        --overwrite_output_dir \
        --output_dir  ${log} \
        --per_device_eval_batch_size 1 >> "${log}/eval_log.txt" 2>&1 &
else
    echo "train or eval"
fi
