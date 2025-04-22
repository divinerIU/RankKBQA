#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

export DATA_DIR=data/CWQ/generation/merged
export MODEL_DIR="your model path"
export CUDA_VISIBLE_DEVICES=2
ACTION=${1:-none}
  dataset="CWQ"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/gen_${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_gen.sh "${exp_prefix}run_gen.sh"
    #git rev-parse HEAD > "${exp_prefix}commitid.log"

    nohup python -u CWQ/run_generator.py \
        --dataset ${dataset} \
        --model_type t5 \
        --overwrite_output_dir \
        --model_name_or_path ${MODEL_DIR}/t5-base \
        --do_train \
        --do_eval \
        --greater_is_better True \
        --train_file ${DATA_DIR}/${dataset}_mask_train.json \
        --predict_file ${DATA_DIR}/${dataset}_mask_dev.json \
        --learning_rate 3e-5 \
        --evaluation_strategy steps \
        --num_train_epochs 20 \
        --logging_steps 616 \
        --eval_steps 616 \
        --save_strategy steps \
        --save_steps 616 \
        --warmup_ratio 0.1 \
        --load_best_model_at_end \
        --metric_for_best_model ex \
        --save_total_limit 2 \
        --output_dir "${exp_prefix}output" \
        --eval_beams 10 \
        --per_device_train_batch_size 24 \
        --per_device_eval_batch_size 24  >> "${exp_prefix}train_log.txt" 2>&1 &

elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-dev}
    log="results/gen/${dataset}_${split}"
    mkdir ${log}
    nohup python -u GrailQA/run_generator.py \
        --dataset ${dataset} \
        --model_type t5 \
        --model_name_or_path ${model} \
        --tokenizer_name ${MODEL_DIR}/t5-base \
        --eval_beams 10 \
        --do_eval \
        --predict_file ${DATA_DIR}/${dataset}_mask_${split}.json \
        --overwrite_output_dir \
        --output_dir  results/gen/${dataset}_${split} \
        --per_device_eval_batch_size 24  >> "${log}/eval_log.txt" 2>&1 &
#    cp results/gen/${dataset}_${split}/top_k_predictions.json misc/${dataset}_${split}_topk_generations.json
else
    echo "train or eval"
fi
