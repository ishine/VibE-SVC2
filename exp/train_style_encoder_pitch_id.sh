#!/bin/bash
# Train Pitch Style Encoder (Target ID, VocalSet)

python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/vocalset_style \
    --feature_type pitch
