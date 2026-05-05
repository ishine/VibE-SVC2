#!/bin/bash
# Inference: Pitch & Timbre Style Conversion (Joint)

python inference_main.py \
    -m pretrain/vibe_timbre/model_200000.pt \
    -c pretrain/vibe_timbre/config.yaml \
    -f filelists/vocalset_timbre/test.txt \
    -s configs/stats/f0_timbre.yaml \
    -f0p rmvpe \
    -e exp/joint \
    --infer_mode joint \
    --target_pitch_style vibrato \
    --target_timbre_style breathy \
    --extent_scale 1.0 \
    --rate_scale 1.0
