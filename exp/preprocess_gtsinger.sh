#!/bin/bash
# Preprocess GTSinger dataset

python preprocess_gtsinger.py \
    --dataset gtsinger \
    --config_path configs/vibe_style_encoder_gtsinger.yaml \
    --raw_dir dataset/GTSinger \
    --data_dir dataset/gtsinger_en_style \
    --target_stats_dir configs/stats \
    --vocoder bigvgan_v2 \
    --target_sr 24000 \
    --num_processes 4
