from __future__ import annotations

import torch
from torch import nn


class _TemporalLayer(nn.Module):
    def __init__(self, channels, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(channels * 2, channels))
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, sequence, key_padding_mask=None, return_attention=False):
        attended, weights = self.attention(sequence, sequence, sequence, key_padding_mask=key_padding_mask, need_weights=return_attention)
        output = self.norm1(sequence + attended)
        return self.norm2(output + self.ffn(output)), weights


class ConstructionTemporalFeatureService(nn.Module):
    """Per-spatial-position temporal attention; preserves B,T,C,H,W."""

    def __init__(self, channels, num_heads=4, num_layers=2, dropout=0.1, max_timesteps=8, return_attention_for_audit=False):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.layers = nn.ModuleList([_TemporalLayer(channels, num_heads, dropout) for _ in range(num_layers)])
        self.time_projection = nn.Sequential(nn.Linear(1, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.max_timesteps = int(max_timesteps)
        self.return_attention_for_audit = bool(return_attention_for_audit)

    def _relative_time(self, timestamps, batch, timesteps, device, dtype):
        if timestamps is None:
            values = torch.arange(timesteps, device=device, dtype=dtype)[None].expand(batch, -1)
        else:
            values = timestamps.to(device=device, dtype=dtype)
        values = values - values[:, :1]
        scale = values.abs().amax(1, keepdim=True).clamp_min(1.0)
        return self.time_projection((values / scale)[..., None])

    def forward(self, multi_scale_features, timestamps=None, temporal_valid_mask=None):
        features = multi_scale_features if isinstance(multi_scale_features, (list, tuple)) else [multi_scale_features]
        refined, audits = [], []
        for feature in features:
            if feature.ndim != 5:
                raise ValueError("construction TFR expects [B,T,C,H,W]")
            batch, timesteps, channels, height, width = feature.shape
            if timesteps < 2 or timesteps > self.max_timesteps:
                raise ValueError(f"timesteps must be 2..{self.max_timesteps}")
            valid = temporal_valid_mask if temporal_valid_mask is not None else torch.ones(batch, timesteps, dtype=torch.bool, device=feature.device)
            sequence = feature.permute(0, 3, 4, 1, 2).reshape(batch * height * width, timesteps, channels)
            time_embedding = self._relative_time(timestamps, batch, timesteps, feature.device, feature.dtype)
            sequence = sequence + time_embedding[:, None, None].expand(batch, height, width, timesteps, channels).reshape_as(sequence)
            padding = ~valid[:, None, None].expand(batch, height, width, timesteps).reshape(batch * height * width, timesteps)
            layer_audits = []
            for layer in self.layers:
                sequence, attention = layer(sequence, padding, self.return_attention_for_audit)
                if attention is not None:
                    layer_audits.append(attention.mean(0).detach())
            output = sequence.reshape(batch, height, width, timesteps, channels).permute(0, 3, 4, 1, 2)
            output = output.masked_fill(~valid[:, :, None, None, None], 0)
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("non-finite construction TFR output")
            refined.append(output); audits.append(layer_audits)
        return {"refined_features": refined, "temporal_attention": audits if self.return_attention_for_audit else None, "valid_timesteps": valid}
