#!/bin/bash
# Inference: Timbre Style Conversion (Target ID)

python inference_main.py \
    -m pretrain/vibe_timbre/model_200000.pt \
    -c pretrain/vibe_timbre/config.yaml \
    -f filelists/vocalset_timbre/test.txt \
    -s configs/stats/f0_timbre.yaml \
    -f0p rmvpe \
    -e exp/timbre_style_only \
    --infer_mode timbre \
    --target_timbre_style vocal_fry
