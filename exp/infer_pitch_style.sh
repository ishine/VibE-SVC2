#!/bin/bash
# Inference: Pitch Style Conversion (Target ID)

python inference_main.py \
    -m pretrain/vibe_pitch/model_200000.pt \
    -c pretrain/vibe_pitch/config.yaml \
    -f filelists/vocalset_pitch/test.txt \
    -s configs/stats/f0_pitch.yaml \
    -f0p rmvpe \
    -e exp/pitch_style_only \
    --infer_mode pitch \
    --target_pitch_style vibrato \
    --extent_scale 1.0 \
    --rate_scale 1.0
