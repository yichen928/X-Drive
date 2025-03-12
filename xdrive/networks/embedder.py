import logging
import torch
import torch.nn as nn

import math


class Embedder:
    """
    borrow from
    https://github.com/zju3dv/animatable_nerf/blob/master/lib/networks/embedder.py
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj


class PositionEmbedder3D(nn.Module):
    def __init__(self, embed_dim, num_freqs=8, fov=(-30, 10)):
        super().__init__()
        self.embedder_obj = get_embedder(input_dims=3, num_freqs=num_freqs)
        out_dim = self.embedder_obj.out_dim

        self.pos_encoder = nn.Sequential(
            nn.Linear(out_dim, embed_dim),
            nn.SiLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fov = [deg / 180 * math.pi for deg in fov]

    def forward(self, hidden_states):
        width = hidden_states.shape[2]
        height = hidden_states.shape[1]

        yaw = (torch.arange(width) + 0.5) / width * 2 * math.pi
        pitch = self.fov[1] - (torch.arange(height) + 0.5) / height * (self.fov[1] - self.fov[0])

        yaw = yaw.type_as(hidden_states)
        pitch = pitch.type_as(hidden_states)

        zs = torch.sin(pitch[:, None]).repeat(1, yaw.shape[0])
        ys = torch.cos(pitch[:, None]) * torch.sin(yaw[None])
        xs = torch.cos(pitch[:, None]) * torch.cos(yaw[None])

        pos_embedding = torch.stack([xs, ys, zs], dim=-1)
        pos_embedding = self.embedder_obj(pos_embedding)

        pos_encoding = self.pos_encoder(pos_embedding)

        hidden_states = hidden_states + pos_encoding[None]

        return hidden_states


class PositionEmbedderMLP3D(nn.Module):
    def __init__(self, embed_dim, num_freqs=8, fov=(-30, 10)):
        super().__init__()
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fov = [deg / 180 * math.pi for deg in fov]

    def forward(self, hidden_states):
        width = hidden_states.shape[2]
        height = hidden_states.shape[1]

        yaw = (torch.arange(width) + 0.5) / width * 2 * math.pi
        pitch = self.fov[1] - (torch.arange(height) + 0.5) / height * (self.fov[1] - self.fov[0])

        yaw = yaw.type_as(hidden_states)
        pitch = pitch.type_as(hidden_states)

        zs = torch.sin(pitch[:, None]).repeat(1, yaw.shape[0])
        # ys = torch.cos(pitch[:, None]) * torch.sin(yaw[None])
        # xs = torch.cos(pitch[:, None]) * torch.cos(yaw[None])
        ys = torch.cos(pitch[:, None]) * torch.cos(yaw[None])
        xs = torch.cos(pitch[:, None]) * torch.sin(yaw[None])

        pos_embedding = torch.stack([xs, ys, zs], dim=-1)

        pos_encoding = self.pos_encoder(pos_embedding)

        hidden_states = hidden_states + pos_encoding[None]

        return hidden_states


class PositionEmbedderFourierMLP3D(nn.Module):
    def __init__(self, embed_dim, num_freqs=8, fov=(-30, 10)):
        super().__init__()
        self.fov = [deg / 180 * math.pi for deg in fov]
        self.fourier_embedder = get_embedder(3, num_freqs)
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.fourier_embedder.out_dim, embed_dim),
            nn.SiLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, hidden_states):
        width = hidden_states.shape[2]
        height = hidden_states.shape[1]

        yaw = (torch.arange(width) + 0.5) / width * 2 * math.pi
        pitch = self.fov[1] - (torch.arange(height) + 0.5) / height * (self.fov[1] - self.fov[0])

        yaw = yaw.type_as(hidden_states)
        pitch = pitch.type_as(hidden_states)

        zs = torch.sin(pitch[:, None]).repeat(1, yaw.shape[0])
        ys = torch.cos(pitch[:, None]) * torch.sin(yaw[None])
        xs = torch.cos(pitch[:, None]) * torch.cos(yaw[None])
        
        xyz_pos = torch.stack([xs, ys, zs], dim=-1)
        xyz_pos = (xyz_pos + 1) / 2

        pos_embedding = self.fourier_embedder(xyz_pos)

        pos_encoding = self.pos_encoder(pos_embedding)

        hidden_states = hidden_states + pos_encoding[None]

        return hidden_states