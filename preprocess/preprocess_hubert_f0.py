import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from utils.commons.encoder_utils import get_speech_encoder
from utils.commons.energy_utils import Volume_Extractor
from utils.commons.f0_utils import get_f0_predictor
from utils.commons.utils import load_config
from vocoder import Vocoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/vocalset_style",
        help="path to source dir",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="You are advised to set the number of processes to the same as the number of CPU cores",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/configs/diffusion.yaml",
        help="set yaml path",
    )
    return parser.parse_args()


def process_one(
    filename, content_encoder, mel_extractor, f0p, config=None, device=None
):
    sampling_rate = config.data.sampling_rate
    hop_length = config.data.hop_size

    # Load waveform
    wav, _ = librosa.load(filename, sr=sampling_rate)
    wav = librosa.util.normalize(wav)
    wav_torch = torch.FloatTensor(wav).unsqueeze(0)

    # Extraction of SSL features
    soft_path = filename + ".soft.pt"
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = content_encoder.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    # Extraction of F0 contour
    f0_path = filename + ".f0.npy"
    if not os.path.exists(f0_path):
        f0_predictor = get_f0_predictor(
            f0p,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            device=device,
            threshold=0.05,
        )
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))

    # Extraction of volume contour
    volume_path = filename + ".vol.npy"
    volume_extractor = Volume_Extractor(hop_length)
    if not os.path.exists(volume_path):
        volume = volume_extractor.extract(wav_torch).cpu().numpy().reshape(-1)
        np.save(volume_path, volume)

    # Extraction of Mel-Spectrogram
    mel_path = filename + ".mel.npy"
    if not os.path.exists(mel_path) and mel_extractor is not None:
        mel_t = mel_extractor.extract(wav_torch.to(device), sampling_rate)
        mel = mel_t.squeeze().cpu().numpy()
        np.save(mel_path, mel)


def process_batch(rank, file_chunk, config_path, device="cpu"):
    # Check GPU availablity
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        raise NotImplementedError("Non-GPU processing is not allowed!")

    # Load configurations
    config = load_config(config_path)

    logger.info(f"Rank {rank} uses device {device}")
    if rank == 0:
        logger.info("Using SpeechEncoder: " + config.data.encoder)
        logger.info("Using extractor: " + config.data.f0_predictor)

    # Load feature extractor
    content_encoder = get_speech_encoder(config.data.encoder, device=device)
    mel_extractor = Vocoder(config.model.vocoder.type, device=device)

    # Run multi-process extraction
    for filename in tqdm(file_chunk, position=rank):
        process_one(
            filename,
            content_encoder,
            mel_extractor,
            config.data.f0_predictor,
            config,
            device,
        )


def main(args):
    num_processes = os.cpu_count() if args.num_processes == 0 else args.num_processes
    filenames = glob(f"{args.data_dir}/*/*.wav", recursive=True)

    # Set multi-process feature extraction
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for rank in range(num_processes):
            start = int(rank * len(filenames) / num_processes)
            end = int((rank + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(
                executor.submit(
                    process_batch,
                    rank,
                    file_chunk,
                    args.config_path,
                    device=args.device,
                )
            )
        for task in tasks:
            task.result()


if __name__ == "__main__":
    args = parse_args()
    main(args)
