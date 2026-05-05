#!/bin/bash
# Train Energy Style Encoder (Zero-Shot, GTSinger)

python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/gtsinger_en_style \
    --feature_type energy \
    --zero_shot
