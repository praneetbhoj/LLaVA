#!/bin/bash

for weight_decay in 0.05 0.1;
do
    for learning_rate in 1e-5 3e-5 5e-5;
    do
        python llava/train/train_mem.py \
            --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
            --model_name_or_path liuhaotian/llava-v1.5-13b \
            --version v1 \
            --data_path ./playground/data/train.json \
            --validation_data_path ./playground/data/evaluation-spatial-reasoning.json ./playground/data/evaluation-task-planner.json \
            --image_folder ./playground/data \
            --vision_tower openai/clip-vit-large-patch14-336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --freeze_mm_mlp_adapter True \
            --freeze_mm_vision_encoder True \
            --freeze_mm_language_model True \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir ./checkpoints/run5 \
            --num_train_epochs 10 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --logging_strategy "steps" \
            --logging_steps 1 \
            --evaluation_strategy "steps" \
            --eval_steps 100 \
            --save_strategy "epoch" \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --warmup_ratio 0.01 \
            --lr_scheduler_type "cosine" \
            --tf32 True \
            --model_max_length 2048 \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb
    done
done