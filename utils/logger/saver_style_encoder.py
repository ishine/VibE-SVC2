import datetime
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import torch

from .base import BaseSaver

# from pitch_style_encoder.utils import log_to_constant_trans_np


EPS = 1e-6


def log_to_constant_trans_np(signal):
    return 2**signal - EPS


class Saver(BaseSaver):
    def log_pitch_contour(
        self,
        name,
        audio_path,
        pred_f0,
        gt_f0,
        gt_low_f0,
        gt_high_f0,
        pred_high_f0,
        audio_config,
        epoch,
    ):
        wav, _ = librosa.load(audio_path, sr=audio_config["sampling_rate"])
        snd = parselmouth.Sound(wav, sampling_frequency=audio_config["sampling_rate"])
        spectrogram = snd.to_spectrogram(
            maximum_frequency=audio_config["sampling_rate"],
            frequency_step=10.0,
            time_step=audio_config["window_size"] / audio_config["sampling_rate"],
        )
        # Type casting to numpy
        if isinstance(gt_f0, torch.Tensor):
            gt_f0 = gt_f0.cpu().numpy()
        if isinstance(gt_low_f0, torch.Tensor):
            gt_low_f0 = gt_low_f0.cpu().numpy()
        if isinstance(gt_high_f0, torch.Tensor):
            gt_high_f0 = gt_high_f0.cpu().numpy()

        if isinstance(pred_f0, torch.Tensor):
            pred_f0 = pred_f0.cpu().numpy()
        if isinstance(pred_high_f0, torch.Tensor):
            pred_high_f0 = pred_high_f0.cpu().numpy()

        # Denormaling to constant scale
        gt_f0 = log_to_constant_trans_np(gt_f0)
        gt_low_f0 = log_to_constant_trans_np(gt_low_f0)
        gt_high_f0 = log_to_constant_trans_np(gt_high_f0)

        pred_f0 = log_to_constant_trans_np(pred_f0)
        pred_high_f0 = log_to_constant_trans_np(pred_high_f0)

        # Load plot
        fig = plt.figure(figsize=(12, 12))
        plot_xs = librosa.times_like(
            pred_f0, sr=audio_config["sampling_rate"], hop_length=256
        )

        plt.subplot(4, 1, 1)
        self.draw_spectrogram(spectrogram)
        plt.twinx()
        self.draw_pitch(gt_f0, gt_low_f0, plot_xs, label=["GT_f0", "GT_low_f0"])
        plt.legend()
        plt.title("Orig F0 & Smooth F0")
        plt.xlim([snd.xmin, snd.xmax])
        plt.ylim(0, 800)

        plt.subplot(4, 1, 2)
        self.draw_detail(
            gt_high_f0, pred_high_f0, plot_xs, label=["GT_high_f0", "Pred_high_f0"]
        )
        plt.legend()
        plt.title("Comparison of high-frequency F0")
        plt.xlim([snd.xmin, snd.xmax])

        plt.subplot(4, 1, 3)
        self.draw_pitch(
            np.zeros_like(pred_f0), pred_f0, plot_xs, label=["GT_f0", "Pred_f0"]
        )
        plt.legend()
        plt.title("Converted F0 contour")
        plt.xlim([snd.xmin, snd.xmax])
        plt.ylim(0, 800)

        plt.subplot(4, 1, 4)
        self.draw_pitch(gt_f0, pred_f0, plot_xs, label=["GT_f0", "Pred_f0"])
        plt.legend()
        plt.title("Comparison of F0 contour")
        plt.xlim([snd.xmin, snd.xmax])
        plt.ylim(0, 800)

        plt.tight_layout()

        self.writer.add_figure(name, fig, epoch)
        plt.close()

    def get_ETA_time(self, max_steps, to_str=True):
        total_time = time.time() - self.init_time
        ETA_time = total_time / self.global_step * (max_steps - self.global_step)
        if self.global_step > max_steps:
            ETA_time = 0
        if to_str:
            ETA_time = str(datetime.timedelta(seconds=ETA_time))[:-5]
        return ETA_time

    def draw_spectrogram(spectrogram, dynamic_range=70):
        X, Y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values + 1e-9)
        plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range)
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.xlabel("time [s]")
        plt.ylabel("frequency [Hz]")

    def draw_pitch(pitch, pitch2, plot_xs, label, plot_type="plot"):
        if plot_type == "plot":
            plt.plot(plot_xs, pitch, label=label[0], color="dodgerblue")
            plt.plot(plot_xs, pitch2, label=label[1], color="red")
        else:
            plt.plot(plot_xs, pitch, "o", markersize=5, color="w")
            plt.plot(
                plot_xs, pitch, "o", markersize=2, label=label[0], color="dodgerblue"
            )
            plt.plot(plot_xs, pitch2, "o", markersize=5, color="w")
            plt.plot(plot_xs, pitch2, "o", markersize=2, label=label[1], color="red")
        plt.ylabel("fundamental frequency [Hz]")

    def draw_detail(pitch, pitch2, plot_xs, label, plot_type="plot"):
        if plot_type == "plot":
            plt.plot(plot_xs, pitch, label=label[0], color="dodgerblue")
            plt.plot(plot_xs, pitch2, label=label[1], color="red")
        else:
            plt.plot(plot_xs, pitch, "o", markersize=5, color="w")
            plt.plot(plot_xs, pitch, "o", markersize=2, label=label[0])
            plt.plot(plot_xs, pitch2, "o", markersize=5, color="w")
            plt.plot(plot_xs, pitch2, "o", markersize=2, label=label[1])
        plt.ylabel("Detail frequency [Hz]")

    def log_energy_contour(
        self,
        name,
        audio_path,
        gt_vol,
        gt_low_vol,
        gt_high_vol,
        pred_vol,
        pred_high_vol,
        audio_config,
        epoch,
    ):
        wav, _ = librosa.load(audio_path, sr=audio_config["sampling_rate"])
        snd = parselmouth.Sound(wav, sampling_frequency=audio_config["sampling_rate"])
        spectrogram = snd.to_spectrogram(
            maximum_frequency=audio_config["sampling_rate"],
            frequency_step=10.0,
            time_step=audio_config["window_size"] / audio_config["sampling_rate"],
        )
        # Type casting to numpy
        if isinstance(gt_vol, torch.Tensor):
            gt_vol = gt_vol.cpu().numpy()
        if isinstance(gt_low_vol, torch.Tensor):
            gt_low_vol = gt_low_vol.cpu().numpy()
        if isinstance(gt_high_vol, torch.Tensor):
            gt_high_vol = gt_high_vol.cpu().numpy()
        if isinstance(pred_high_vol, torch.Tensor):
            pred_high_vol = pred_high_vol.cpu().numpy()
        if isinstance(pred_vol, torch.Tensor):
            pred_vol = pred_vol.cpu().numpy()

        # Load plot
        fig = plt.figure(figsize=(12, 12))
        plot_xs = librosa.times_like(
            pred_vol, sr=audio_config["sampling_rate"], hop_length=256
        )

        plt.subplot(4, 1, 1)
        self.draw_spectrogram(spectrogram)
        plt.twinx()
        self.draw_pitch(gt_vol, gt_low_vol, plot_xs, label=["GT_vol", "GT_low_vol"])
        plt.legend()
        plt.title("Orig F0 & Smooth F0")
        plt.xlim([snd.xmin, snd.xmax])

        plt.subplot(4, 1, 2)
        self.draw_detail(
            gt_high_vol, pred_high_vol, plot_xs, label=["GT_high_vol", "Pred_high_vol"]
        )
        plt.legend()
        plt.title("Comparison of high-frequency F0")
        plt.xlim([snd.xmin, snd.xmax])

        plt.subplot(4, 1, 3)
        self.draw_pitch(
            np.zeros_like(pred_vol), pred_vol, plot_xs, label=["GT_vol", "Pred_vol"]
        )
        plt.legend()
        plt.title("Converted F0 contour")
        plt.xlim([snd.xmin, snd.xmax])

        plt.subplot(4, 1, 4)
        self.draw_pitch(gt_vol, pred_vol, plot_xs, label=["GT_vol", "Pred_vol"])
        plt.legend()
        plt.title("Comparison of F0 contour")
        plt.xlim([snd.xmin, snd.xmax])

        plt.tight_layout()

        self.writer.add_figure(name, fig, epoch)
        plt.close()
