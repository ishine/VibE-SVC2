#!/bin/bash
# Preprocess VocalSet dataset

python preprocess_vocalset.py \
    --dataset vocalset \
    --config_path configs/vibe_pitch.yaml \
    --raw_dir dataset/vocalset_raw \
    --data_dir dataset/vocalset_style \
    --target_stats_dir configs/stats \
    --vocoder bigvgan_v2 \
    --target_sr 24000 \
    --num_processes 4
