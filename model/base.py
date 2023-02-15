import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip



class BaseSceneDiffusionModel(nn.Module):
    def __init__(self, seg_len, encoder_mode, decoder_mode, modality, clip_version='ViT-B/32', clip_dim=512, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 d_hid=512, mesh_ds_dir="../data/mesh_ds", posa_path=None, latent_dim=256, **kwargs) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        self.clip_version = clip_version
        self.clip_dim = clip_dim
        self.latent_dim = latent_dim

        # Setup modality for the model, e.g., text.
        self.modality = modality
        self._set_up_modality()

        # Setup timestep embedding layer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        
    def forward(self, x, timesteps, y=None):
        bs, vert_cans, dimension, cat = x.shape
        emb = self.embed_timestep(timesteps)
        return 

    def _set_up_modality(self):
        assert self.modality in ['text', 'audio', None]
        if self.modality == 'text':
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.clip_version = self.clip_version
            self.clip_model = self.load_and_freeze_clip(self.clip_version)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


if __name__ == '__main__':
    model = BaseSceneDiffusionModel(100, True, True, 'text')
    x = torch.rand(10, 256, 655, 8)
    model(x)