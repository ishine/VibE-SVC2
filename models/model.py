import torch.nn as nn

from .diffusion import GaussianDiffusion
from .wavenet import WaveNet


class Unit2Mel(nn.Module):
    def __init__(
        self,
        in_channel,
        n_spk=1,
        n_timbre_tech=1,
        out_dims=128,
        n_layers=20,
        n_chans=384,
        n_hidden=256,
        timesteps=1000,
        k_step_max=0,
    ):
        super().__init__()
        self.n_spk = n_spk
        self.n_hidden = n_hidden
        self.timesteps = timesteps
        self.n_timbre_tech = n_timbre_tech

        self.unit_embed = nn.Linear(in_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.energy_embed = nn.Linear(1, n_hidden)
        self.spk_embed = nn.Embedding(n_spk, n_hidden)
        if n_timbre_tech > 1:
            self.t_style_embed = nn.Embedding(n_timbre_tech, n_hidden)

        # diffusion
        self.decoder = GaussianDiffusion(
            WaveNet(out_dims, n_layers, n_chans, n_hidden),
            timesteps=self.timesteps,
            k_step=k_step_max,
            out_dims=out_dims,
        )

    def forward(
        self,
        units,
        f0,
        energy,
        spk_id,
        timbre_style_id=None,
        gt_spec=None,
        infer=False,
        infer_speedup=10,
        method="dpm-solver",
        k_step=1000,
        use_tqdm=True,
    ):
        """
        input:
            units   : [batch, n_frames, in_channel]
            f0      : [batch, n_frames, 1]
            energy  : [batch, n_frames, 1]
            spk_id  : [batch, 1]

            timbre_style_id (Optional)  : [batch, 1]
            gt_spec (Optional)          : [batch, n_frames, out_channel]

        return:
            mel     : [batch , n_frames, out_channel]
        """
        cond = (
            self.unit_embed(units)
            + self.f0_embed((1 + f0 / 700).log())
            + self.energy_embed(energy)
            + self.spk_embed(spk_id)
        )
        if self.n_timbre_tech > 1:
            cond += self.t_style_embed(timbre_style_id)

        mel = self.decoder(
            cond,
            gt_spec=gt_spec,
            infer=infer,
            infer_speedup=infer_speedup,
            method=method,
            k_step=k_step,
            use_tqdm=use_tqdm,
        )
        return mel
