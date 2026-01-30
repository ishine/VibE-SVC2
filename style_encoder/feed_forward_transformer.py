from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class StyleAdapter(nn.Module):
    """
    Adaspeech2 conditional layer normalization.
    """

    def __init__(self, speaker_dim, adapter_dim, epsilon=1e-5):
        super(StyleAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.W_bias = nn.Linear(self.speaker_dim, self.adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):

        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)
        return y


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1, zero_shot=False):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.dropout = nn.Dropout(dropout)
        self.zero_shot = zero_shot

        if zero_shot:
            self.style_adapter = StyleAdapter(d_in, d_in)
        else:
            self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x, latent_tech):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.zero_shot:
            output = self.style_adapter(output + residual, latent_tech)
        else:
            output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(
        self,
        d_model,
        n_head,
        d_k,
        d_v,
        d_inner,
        kernel_size,
        dropout=0.1,
        zero_shot=False,
    ):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout, zero_shot=zero_shot
        )

    def forward(self, enc_input, latent_tech, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output, latent_tech)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


# Implementation from FastSpeech2


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, config, zero_shot=False):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["encoder_hidden"]
        n_layers = config["encoder_layer"]
        n_head = config["encoder_head"]
        d_k = d_v = config["encoder_hidden"] // config["encoder_head"]
        d_model = config["encoder_hidden"]
        d_inner = config["conv_filter_size"]
        kernel_size = config["conv_kernel_size"]
        dropout = config["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model
        self.zero_shot = zero_shot

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size[i],
                    dropout=dropout,
                    zero_shot=zero_shot,
                )
                for i in range(n_layers)
            ]
        )

    def get_mask_from_lengths(self, lengths, max_len=None, device=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

    def forward(self, src_seq, latent_tech, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # # -- Prepare masks
        mask = self.get_mask_from_lengths(
            torch.Tensor([max_len for _ in range(batch_size)]).to(src_seq.device),
            max_len,
            device=src_seq.device,
        )

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = src_seq + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = src_seq + self.position_enc[:, :max_len, :].expand(
                batch_size, -1, -1
            )

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, latent_tech, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc_0",
                        nn.Linear(self.in_channel, self.out_channel),
                    ),
                    ("mish_0", Mish()),
                    (
                        "fc_0",
                        nn.Linear(self.in_channel, self.out_channel),
                    ),
                    ("mish_1", Mish()),
                ]
            )
        )

    def forward(self, x):
        return self.mlp(x)
