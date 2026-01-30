import argparse
import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import librosa
import numpy as np
from rich.progress import track
from scipy.io import wavfile


def partition(sig, sr, max_s=10):
    sig_len = sig.shape[0]
    max_len = sr * max_s
    chunks = []
    i = -1

    while sig_len > max_len:
        i += 1
        chunks.append((sig[max_len * i : max_len * (i + 1)]))
        sig_len -= max_len
    if sig_len > sr * 2:
        chunks.append((sig[max_len * (i + 1) :]))
    return chunks


def load_wav(wav_path):
    return librosa.load(wav_path, sr=None)


def trim_wav(wav, top_db=40):
    return librosa.effects.trim(wav, top_db=top_db)


def normalize_peak(wav, threshold=1.0):
    peak = np.abs(wav).max()
    if peak > threshold:
        wav = 0.98 * wav / peak
    return wav


def resample_wav(wav, sr, target_sr):
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def save_wav_to_path(wav, save_path, sr):
    wavfile.write(save_path, sr, (wav * np.iinfo(np.int16).max).astype(np.int16))


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]

    wav_path = os.path.join(args.raw_dir, speaker, wav_name)
    if os.path.exists(wav_path) and ".wav" in wav_path:
        os.makedirs(os.path.join(args.data_dir, speaker), exist_ok=True)
        wav, sr = load_wav(wav_path)
        wav, _ = trim_wav(wav)
        wav = normalize_peak(wav)
        resampled_wav = resample_wav(wav, sr, args.target_sr)

        if not args.skip_loudnorm:
            resampled_wav /= np.max(np.abs(resampled_wav))
        chunks = partition(resampled_wav, args.target_sr)

        for i in range(len(chunks)):
            save_path2 = os.path.join(
                args.data_dir, speaker, f"{os.path.splitext(wav_name)[0]}_{i}.wav"
            )

            save_wav_to_path(chunks[i], save_path2, args.target_sr)


def main(args):
    print(f"CPU count: {cpu_count()}")
    speakers = os.listdir(args.raw_dir)
    process_count = (
        30 if os.cpu_count() > 60 else (os.cpu_count() - 2 if os.cpu_count() > 4 else 1)
    )
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for speaker in speakers:
            spk_dir = os.path.join(args.raw_dir, speaker)
            print(spk_dir)
            if os.path.isdir(spk_dir):
                futures = [
                    executor.submit(process, (spk_dir, i, args))
                    for i in os.listdir(spk_dir)
                    if i.endswith("wav")
                ]
                for _ in track(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    description="resampling:",
                ):
                    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_sr", type=int, default=44100, help="sampling rate")
    parser.add_argument(
        "--raw_dir", type=str, default="./dataset/raw", help="path to source dir"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset/44k", help="path to target dir"
    )
    parser.add_argument(
        "--skip_loudnorm",
        action="store_true",
        help="Skip loudness matching if you have done it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
