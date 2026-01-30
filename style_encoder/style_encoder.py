import torch
import torch.nn as nn

from style_encoder.feed_forward_transformer import MLP, Encoder
from style_encoder.mel_style_encoder import MelStyleEncoder


class TechConverter(nn.Module):
    def __init__(self, transformer_config, n_tech_type, zero_shot=False):
        super(TechConverter, self).__init__()
        self.hidden_dim = transformer_config["encoder_hidden"]
        self.zero_shot = zero_shot

        # input condition embedding
        self.f0_embed = nn.Linear(1, self.hidden_dim)
        self.uv_embed = nn.Linear(1, self.hidden_dim)

        if zero_shot:  # Zero-shot based model
            self.mel_style_encoder = MelStyleEncoder()
            self.mlp = MLP(self.hidden_dim * 2, self.hidden_dim)
        else:  # Target ID based model
            self.tech_embed = nn.Embedding(n_tech_type, self.hidden_dim)
            self.mlp = MLP(self.hidden_dim * 3, self.hidden_dim)

        # style encoder
        self.encoder = Encoder(transformer_config, zero_shot=zero_shot)
        self.proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, low_f0, uv_vector, tech_id=None, high_signal=None):
        """
        Args:
            low_f0 : Log F0 contour [B, T, C]
            uv_vector : Unvoiced flag [B, T, C]
            (Optional) tech_id : Target technique ID [B, C]
            (Optional) high_signal : Reference high F0 contour [B, T, C]

        Returns:
            pred_f0 : Converted F0 contour
            pred_high_f0 : Converted High-frequency F0 contour [B, T, C]
        """
        # Generate embeddings
        f0_out = self.f0_embed(low_f0)
        uv_emb = self.uv_embed(uv_vector)

        # Predict style-converted high-frequency F0 contour
        if self.zero_shot:
            tech_emb = self.mel_style_encoder(high_signal)
            cond = self.mlp(torch.cat([f0_out, uv_emb], 2))
            pred_high_f0 = self.encoder(cond, latent_tech=tech_emb)
        else:
            tech_emb = self.tech_embed(tech_id).expand(-1, low_f0.shape[1], -1)
            cond = self.mlp(torch.cat([f0_out, uv_emb, tech_emb], 2))

            pred_high_f0 = self.encoder(cond, latent_tech=None)

        pred_high_f0 = self.proj(pred_high_f0)
        return low_f0 + pred_high_f0, pred_high_f0
