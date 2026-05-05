import argparse
import os
import shutil
import warnings

import soundfile
import torch
import torch.distributed as dist
from tqdm import tqdm

from inference.infer_zsvibesvcII import Svc
from utils.commons.utils import load_config

warnings.filterwarnings("ignore")


def run(rank, n_gpus, args, audio_list, ref_list):
    # Multiprocessing setting
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    # Load arguments
    exp_dir = args.exp_dir
    f0p = args.f0_predictor

    # model parameters
    model_path = args.model_path
    config_path = args.config_path
    k_step = args.k_step
    pitch_zeroshot = args.pitch_zeroshot

    # task settings
    recon_spk = args.recon_spk
    vocoded = args.vocoded
    infer_mode = args.infer_mode
    recon_tech = args.recon_tech

    # inference paramters
    target_spk = args.target_spk
    target_pitch_style = args.target_pitch_style
    target_timbre_style = args.target_timbre_style

    extent_scale = args.extent_scale
    extent_scale_type = args.extent_scale_type
    rate_scale = args.rate_scale

    extent_scale_energy = args.extent_scale_energy
    rate_scale_energy = args.rate_scale_energy

    vocal_fry_enforcement = args.vocal_fry_enforcement

    # Catch the exceptional task
    if target_spk is not None:
        print("Zero-shot task do not provide speaker ID control.")
        return

    # Load configuration files
    pretrain_path_config = load_config("./configs/paths.yaml")
    spk_stat_path = args.stats_path

    d_config = load_config(config_path)
    spk_stat_config = load_config(spk_stat_path)

    svc_model = Svc(
        model_path,
        config_path,
        pretrain_path_config,
        f0p,
        infer_mode,
        rank,
        eps=float(d_config.data.eps),
        vocal_fry_enforcement=vocal_fry_enforcement,
        spk_stat_config = spk_stat_config
    )

    # Set output directory
    gt_dir = f"./results/{exp_dir}/gt"
    gen_dir = f"./results/{exp_dir}/gen"
    ref_dir = f"./results/{exp_dir}/ref"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    d_config["env"]["expdir"] = f"./results/{exp_dir}"
    # spk_list = d_config["spk"]

    # Allocate sample idx for multi-gpu inference
    prev_rank_idx = 0
    if rank != 0:
        for r in range(rank):
            prev_rank_idx += len(audio_list[r])

    # Main stream
    for idx, src in enumerate(tqdm(audio_list[rank])):
        src = src.strip()
        src_spk = src.split("/")[-2]
        src_tech = src.split("/")[-1].split("#")[2]

        # copy GT & Reference audio from raw path
        gt_audio_path = f"{gt_dir}/{prev_rank_idx + idx}_{src_spk}#{src_tech}#{src.split('/')[-1].split('#')[-1]}"
        if not os.path.exists(gt_audio_path):
            shutil.copyfile(src, gt_audio_path)

        if infer_mode == "pitch" and src_tech == target_pitch_style:
            continue
        elif infer_mode == "timbre" and src_tech == target_timbre_style:
            continue
        elif infer_mode == "joint" and src_tech == target_timbre_style:
            continue

        # Generate vocoded samples
        if vocoded:
            gen_audio = svc_model.vocoded(gt_audio_path)
            gen_audio_path = f"{gen_dir}/{prev_rank_idx + idx}_{src_spk}#{src_tech}#{src.split('/')[-1].split('#')[-1]}"
            soundfile.write(
                gen_audio_path, gen_audio, svc_model.target_sample, format="wav"
            )
            continue

        # Generate converted samples
        for ref_idx, ref_path in enumerate(ref_list):
            # Define reference audio path
            ref_src = ref_list[ref_idx].strip()
            ref_spk = ref_src.split("/")[-2]

            ref_tech = ref_src.split("/")[-1].split("#")[2]
            # Skip reconstruction settings
            if src_tech == ref_tech:
                continue
            
            ref_audio_path = f"{ref_dir}/{prev_rank_idx + idx}_{ref_idx}_{ref_spk}#{ref_tech}#{ref_src.split('/')[-1].split('#')[-1]}"
            if not os.path.exists(ref_audio_path):
                shutil.copyfile(ref_src, ref_audio_path)

            

            # Shifting f0 contour by a scaler factor based on precalculate statistics
            if ref_spk != src_spk:
                if svc_model.n_timbre_style != 1:
                    f0_shift = spk_stat_config[ref_spk]
                else:
                    f0_shift = spk_stat_config[ref_spk] / spk_stat_config[src_spk]
            else:
                f0_shift = None

            # Info : Try **torch.no_grad** when an error occurred by inplace computation
            with torch.inference_mode():
                gen_audio = svc_model.infer(
                    source_spk=src_spk,
                    target_spk=ref_spk,
                    source_tech=src_tech,
                    target_pitch_style=None,
                    target_timbre_style=target_timbre_style,
                    raw_path=gt_audio_path,
                    ref_path=ref_audio_path,
                    k_step=k_step,
                    f0_shift=f0_shift,
                    extent_scale=extent_scale,
                    extent_scale_type=extent_scale_type,
                    rate_scale=rate_scale,
                    extent_scale_energy=extent_scale_energy,
                    rate_scale_energy=rate_scale_energy,
                    vocal_fry_enforcement=vocal_fry_enforcement,
                )

            # Save the generated audio
            gen_audio_path = f"{gen_dir}/{prev_rank_idx + idx}_{ref_idx}_{ref_spk}#{src_tech}#{src.split('/')[-1].split('#')[-1]}"
            soundfile.write(
                gen_audio_path, gen_audio, svc_model.target_sample, format="wav"
            )
            svc_model.clear_empty()
