# VibE-SVC2

**A Vibrato Controlling Method by Predicting High-frequency F0 contour for Singing Voice Conversion** *(Under review)*

Joon-Seung Choi, Dong-Min Byun, and Seong-Whan Lee

[![demo](https://img.shields.io/badge/GitHub-Demo-green.svg)](https://castlechoi.github.io/VibE-SVC2-demo/)

![](img/model.png)

> **Previous work:** VibE-SVC: Vibrato Extraction with High-frequency F0 Contour for Singing Voice Conversion *(Interspeech 2025)* — [[Paper](https://arxiv.org/abs/2505.20794)] [[Demo](https://castlechoi.github.io/VibE-SVC-demo/)]

---

## 📑 Contents

1. [Setup](#1-setup)
2. [Pre-trained Checkpoints](#2-pre-trained-checkpoints)
3. [Data Preparation & Preprocessing](#3-data-preparation--preprocessing)
4. [Inference](#4-inference)
5. [Training](#5-training)
6. [Acknowledgements](#6-acknowledgements)

---

## 1. Setup

```bash
git clone https://github.com/castlechoi/VibE-SVC2
cd VibE-SVC2
pip install -r requirements.txt
```

---

## 2. Pre-trained Checkpoints

Download from [Google Drive](https://drive.google.com/drive/folders/1HJfU-MVYvlJVV1jkeXln8jJfdHAbC0N0?usp=drive_link) or [Hugging Face](https://huggingface.co/castlechoi/vibesvc2/tree/main) and place each file in the path shown below.

### 2.1 SVC Models

| Model | Path | Dataset | Styles |
|---|---|---|---|
| Pitch Style Conversion | `pretrain/vibe_pitch` | VocalSet | Straight / Vibrato |
| Timbre Style Conversion | `pretrain/vibe_timbre` | VocalSet | Straight / Belt / Breathy / Vocal Fry |
| Pitch Style Conversion (GTSinger) | `logs/vibe_pitch_gtsinger/diffusion` | GTSinger | Control / Vibrato (zero-shot) |

### 2.2 Style Converters

| Cond. Type | Converter | Path | Dataset | Styles |
|---|---|---|---|---|
| ID | Pitch | `pretrain/style_converter/pitch_ID` | VocalSet | Straight / Vibrato |
| ID | Energy | `pretrain/style_converter/energy_ID` | VocalSet | Straight / Vibrato |
| Reference high-freq. F0 | Pitch | `pretrain/style_converter/pitch_ZS` | GTSinger | Straight / Vibrato |
| Reference high-freq. F0 | Energy | `pretrain/style_converter/energy_ZS` | GTSinger | Straight / Vibrato |

### 2.3 Feature Extractors

We use the pre-trained feature extractors available from [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc):

- **Content encoder**: [HuBERT-soft](https://github.com/bshall/hubert/releases/)
- **F0 extractor**: [RMVPE](https://github.com/yxlllc/RMVPE/releases/) 



## 3. Data Preparation & Preprocessing

### 3.1 VocalSet

Prepare [VocalSet](https://zenodo.org/records/1193957) so `dataset/VocalSet/FULL/...` exists, then run:

```bash
python preprocess_vocalset.py \
    --dataset vocalset \
    --config_path configs/vibe_pitch.yaml \
    --raw_dir dataset/vocalset_raw \
    --data_dir dataset/vocalset_style \
    --target_stats_dir configs/stats \
    --vocoder bigvgan_v2 \
    --target_sr 24000 \
    --num_processes 4
```

The flatten step covers all 5 techniques: **straight, vibrato, belt, breathy, vocal_fry**.

`preprocess_flist` is invoked twice internally
- `filelists/vocalset_pitch/{train,val,test}.txt` — straight + vibrato
- `filelists/vocalset_timbre/{train,val,test}.txt` — straight + belt + breathy + vocal_fry

### 3.2 GTSinger

Prepare the **English** subset of [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) so `dataset/GTSinger/English/...` exists, then run:

```bash
python preprocess_gtsinger.py \
    --dataset gtsinger \
    --config_path configs/vibe_style_encoder_gtsinger.yaml \
    --raw_dir dataset/GTSinger \
    --data_dir dataset/gtsinger_en_style \
    --target_stats_dir configs/stats \
    --vocoder bigvgan_v2 \
    --target_sr 24000 \
    --num_processes 4
```



Stage 1 produces two filelists from a single data directory:

- `filelists/gtsinger_en_style/{train,val,test}.txt` — Control + Vibrato (train/val/test)
- `filelists/gtsinger_glissando/test.txt` — **Glissando-only, evaluation-only** (test)

The glissando filelist is generated via `configs/vibe_style_encoder_gtsinger_glissando.yaml`, which sets `data.eval_only: true`.

---

## 4. Inference

```bash
python inference_main.py [options]
```

### 4.1 Common Arguments

| Group | Arg | Description |
|---|---|---|
| Env | `-m`, `--model_path` | SVC checkpoint |
|     | `-c`, `--config_path` | Model config |
|     | `-f`, `--filelist` | Test filelist |
|     | `-e`, `--exp_dir` | Output dir under `results/` |
|     | `-s`, `--stats_path` | Speaker F0 stats yaml |
| Features | `-f0p`, `--f0_predictor` | `rmvpe` / `crepe` / `dio` |
|          | `-ks`, `--k_step` | Diffusion timestep |
| Mode | `--infer_mode` | `pitch` / `timbre` / `joint` |
|      | `--target_pitch_style` | `straight` / `vibrato` / `Control_Group` / `Vibrato_Group` |
|      | `--target_timbre_style` | `straight` / `belt` / `breathy` / `vocal_fry` |
|      | `--pitch_zeroshot` | Enable zero-shot pitch transfer |
| Vibrato control | `--extent_scale` | Vibrato extent scale α (default 1.0) |
|                 | `--extent_scale_type` | `global` / `inc_linear` / `dec_linear` / `sinusoidal` |
|                 | `--rate_scale` | Vibrato rate scale β (default 1.0) |
|                 | `--vocal_fry_enforcement` | Force vocal-fry behaviour |
| Misc | `--multi_infer` | Convert to **all** target styles (output → `{env}/to_{tech}/`) |
|      | `-rcs`, `--recon_spk` | Reconstruct with source speaker |
|      | `-rct`, `--recon_tech` | Reconstruct with source technique |
|      | `--vocoded` | Generate vocoded reference |

### 4.2 Pitch Style Conversion (Target ID)

```bash
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
```

### 4.3 Pitch Style Conversion (Zero-Shot)


**VocalSet → VocalSet** (using the VocalSet pitch model):

```bash
python inference_main.py \
    -m pretrain/vibe_pitch/model_200000.pt \
    -c pretrain/vibe_pitch/config.yaml \
    -f filelists/vocalset_pitch/test.txt \
    -s configs/stats/f0_pitch_gtsinger.yaml \
    -f0p rmvpe \
    -e exp/zeroshot_vocalset \
    --infer_mode pitch \
    --target_pitch_style vibrato \
    --extent_scale 1.0 \
    --rate_scale 1.0 \
    --pitch_zeroshot
```

**GTSinger → GTSinger** (use the GTSinger-trained model):

```bash
python inference_main.py \
    -m logs/vibe_pitch_gtsinger/diffusion/model_200000.pt \
    -c logs/vibe_pitch_gtsinger/diffusion/config.yaml \
    -f filelists/gtsinger_en_style/test.txt \
    -s configs/stats/f0_pitch_gtsinger.yaml \
    -f0p rmvpe \
    -e exp/zeroshot_gtsinger \
    --infer_mode pitch \
    --target_pitch_style Vibrato_Group \
    --extent_scale 1.0 \
    --rate_scale 1.0 \
    --pitch_zeroshot
```

Outputs are written to `results/{exp_dir}/{gt,ref,gen}/`.

### 4.4 Timbre Style Conversion (Target ID)

```bash
python inference_main.py \
    -m pretrain/vibe_timbre/model_200000.pt \
    -c pretrain/vibe_timbre/config.yaml \
    -f filelists/vocalset_timbre/test.txt \
    -s configs/stats/f0_timbre.yaml \
    -f0p rmvpe \
    -e exp/timbre_style_only \
    --infer_mode timbre \
    --target_timbre_style vocal_fry
```

### 4.5 Joint Pitch + Timbre Conversion

Use the timbre model for joint conversion.

```bash
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
```

---

## 5. Training

### 5.1 SVC Models

```bash
# Pitch Style Conversion
python train.py -c configs/vibe_pitch.yaml

# Timbre Style Conversion
python train.py -c configs/vibe_timbre.yaml
```

### 5.2 Style Converters

| Variant | Data | Flag |
|---|---|---|
| `Pitch_ID` | VocalSet | — |
| `Pitch_ZS` | GTSinger | `--zero_shot` |
| `Energy_ID` | VocalSet | — |
| `Energy_ZS` | GTSinger | `--zero_shot` |

```bash
# Pitch_ID
python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/vocalset_style \
    --feature_type pitch

# Pitch_ZS
python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/gtsinger_en_style \
    --feature_type pitch \
    --zero_shot

# Energy_ID
python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/vocalset_style \
    --feature_type energy

# Energy_ZS
python train_style_converter.py \
    -c configs/vibe_style_encoder.yaml \
    --data_dir dataset/gtsinger_en_style \
    --feature_type energy \
    --zero_shot
```

---

## 6. Acknowledgements

- [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
