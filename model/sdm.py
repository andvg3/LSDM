import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from model.diffusion_utils import *

from posa.posa_models import Decoder as POSA_Decoder



class SceneDiffusionModel(nn.Module):
    def __init__(self, seg_len=256, modality='text', clip_version='ViT-B/32', clip_dim=512, dropout=0.1, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 d_hid=256, mesh_ds_dir="data/mesh_ds", posa_path=None, latent_dim=512, cond_mask_prob=1.0, device=0, vert_dims=655, obj_cat=8, 
                 data_rep='rot6d', njoints=251, use_cuda=True, **kwargs) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.clip_version = clip_version
        self.clip_dim = clip_dim
        self.latent_dim = latent_dim
        self.extract_dim = clip_dim + latent_dim
        self.dropout = dropout
        self.cond_mask_prob = cond_mask_prob
        self.data_rep = data_rep
        self.input_feats = vert_dims * obj_cat
        self.device = "cuda:{}".format(device) if use_cuda else "cpu"

        # Setup modality for the model, e.g., text.
        self.modality = modality
        self._set_up_modality()

        # Setup timestep embedding layer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, device=self.device)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder, device=self.device)

        # Setup POSA embedding for human motions
        self.posa = POSA_Decoder(input_feats=self.input_feats, ds_us_dir=mesh_ds_dir, use_semantics=True, channels=f_vert).to(self.device)
        self.linear_extraction = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.input_feats, self.extract_dim),
            nn.GELU(),
        ).to(self.device)

        # Setup combination layers for extracted information
        self.combine_extraction = nn.Sequential(
            nn.Linear(self.extract_dim * 2, self.extract_dim),
            nn.GELU(),
        ).to(self.device)

        # Setup U-net-like input and output process
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.extract_dim).to(self.device)
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.extract_dim, self.seg_len,
                                            vert_dims, obj_cat).to(self.device)
        
    def forward(self, x, vertices, mask, timesteps, y=None, force_mask=False):
        """
        x: noisy signal - torch.Tensor.shape([bs, seq_len, dims, cat]). E.g, 1, 256, 655, 8
        vertices: torch.Tensor.shape([bs, seq_len, dim, 3])
        mask: torch.Tensor.shape([bs, seq_len])
        timesteps: torch.Tensor.shape([bs,])
        y: modality, e.g., text
        """
        bs, seq_len, dims, cat = x.shape

        # Embed features from time
        emb_ts = self.embed_timestep(timesteps)

        # Embed features from modality
        if self.modality == 'text':
            enc_text = self._encode_text(y)
            emb_mod = self.embed_text(self._mask_cond(enc_text, force_mask=force_mask))
            emb_mod = emb_mod.unsqueeze(0)
        
        # Combine features from timestep and modality
        emb = torch.cat((emb_ts, emb_mod), dim=-1)
        emb = emb.repeat(1, seq_len, 1)

        # Embed human motions
        vertices = vertices.squeeze()
        _x = x.squeeze(0).clone().detach()
        _x = _x.view(_x.shape[0], _x.shape[1] * _x.shape[2])
        posa_out = self.posa(_x, vertices)
        out = posa_out.unsqueeze(0)
        out = self.linear_extraction(out)

        # Final embedding features
        emb = torch.cat((out, emb), dim=-1)
        emb = self.combine_extraction(emb)

        # Reconstruct features
        x = self.input_process(x, emb)
        x = self.output_process(x)
        return x

    def _set_up_modality(self):
        assert self.modality in ['text', 'audio', None]
        if self.modality == 'text':
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim).to(self.device)
            self.clip_version = self.clip_version
            self.clip_model = self._load_and_freeze_clip(self.clip_version, device=self.device)
    
    def _mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def _encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = self.device
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model


if __name__ == '__main__':
    model = SceneDiffusionModel(256, 'text')
    x = torch.rand(1, 256, 655, 8).cuda()
    cf = torch.rand(1, 256, 655, 8).cuda()
    t = torch.randint(0, 8, (1,)).cuda()
    vertices = torch.rand(1, 256, 655, 3).cuda()
    mask = torch.randint(0, 1, (1, 256)).cuda()
    print(model(x, cf, vertices, mask, t, ["Hello" for _ in range(1)]).data.shape)