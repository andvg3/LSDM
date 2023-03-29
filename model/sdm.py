import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import clip

from model.diffusion_utils import *
from model.pcd_backbone.pointnet2 import get_backbone

from posa.posa_models import Decoder as POSA_Decoder



class SceneDiffusionModel(nn.Module):
    def __init__(self, seg_len=256, modality='text', clip_version='ViT-B/32', clip_dim=512, dropout=0.1, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 cat_emb=32, mesh_ds_dir="data/mesh_ds", posa_path=None, latent_dim=128, cond_mask_prob=1.0, device=0, vert_dims=655, obj_cat=8, 
                 data_rep='rot6d', njoints=251, use_cuda=True, pcd_points=1024, pcd_dim=128, xyz_dim=3, max_cats=13, translation_params=12,
                 **kwargs) -> None:
        super().__init__()
        self.seg_len = seg_len
        self.pcd_points = pcd_points
        self.clip_version = clip_version
        self.clip_dim = clip_dim
        self.latent_dim = latent_dim
        self.pcd_dim = pcd_dim
        self.pcd_points = pcd_points
        self.xyz_dim = xyz_dim
        self.extract_dim = self.latent_dim
        self.dropout = dropout
        self.cond_mask_prob = cond_mask_prob
        self.data_rep = data_rep
        self.input_feats = vert_dims * obj_cat
        self.n_head = n_head
        self.translation_params = translation_params
        self.device = "cuda:{}".format(device) if use_cuda else "cpu"

        # Setup modality for the model, e.g., text.
        self.modality = modality
        self._set_up_modality()

        # Setup timestep embedding layer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, device=self.device)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder, device=self.device)

        # Setup embedding layer for modality
        self.saved_cat = None
        self.embed_text = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim//2),
            nn.GELU(),
            nn.Linear(self.clip_dim//2, self.latent_dim*2),
            nn.GELU(),
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.GELU(),
        ).to(self.device)

        # Setup embedding layer for categories
        self.embed_cat = nn.Sequential(
            nn.Linear(max_cats, cat_emb),
            nn.GELU(),
        ).to(self.device)

        # Setup inference for categorical
        self.predict_cat = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim//2),
            nn.GELU(),
            nn.Linear(self.latent_dim//2, self.latent_dim//4),
            nn.GELU(),
            nn.Linear(self.latent_dim//4, max_cats),
            nn.GELU(),
            nn.Softmax(dim=2),
        ).to(self.device)
        
        # Setup attention layer
        self.attn_layer = MultiheadAttention(embed_dim=self.latent_dim, num_heads=n_head, kdim=cat_emb, vdim=pcd_points*pcd_dim, batch_first=True).to(self.device)
        
        # Setup translation layer
        self.translation_layer = nn.Sequential(
            nn.Linear(self.latent_dim + cat_emb, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.translation_params),
            nn.GELU(),
        ).to(self.device)
        self.point_wise_trans_layer = nn.Sequential(
            nn.Linear(self.translation_params + self.xyz_dim, self.xyz_dim),
            nn.GELU(),
        ).to(self.device)

        # Setup pointcloud backbone for point cloud extraction
        self.pcd_attention = MultiheadAttention(embed_dim=self.translation_params, num_heads=self.translation_params, kdim=self.xyz_dim, vdim=self.xyz_dim, batch_first=True).to(self.device)
        self.pcd_backbone = get_backbone(self.pcd_dim).to(self.device)
        self.human_backbone = POSA_Decoder(input_feats=xyz_dim, pcd_dim=self.pcd_points).to(self.device)
        # self.pcd_attention = MultiheadAttention(embed_dim=self.latent_dim)

        # Setup combination layers for extracted information
        self.upsampling_layer = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, self.pcd_points),
            nn.GELU(),
        ).to(self.device)

        self.combine_extraction = nn.Sequential(
            # nn.Linear(self.latent_dim*2, self.latent_dim*1.5),
            # nn.GELU(),
            nn.Linear(self.latent_dim*2, self.extract_dim),
            nn.GELU(),
        ).to(self.device)

        # Setup U-net-like input and output process
        self.input_process = InputProcess(self.data_rep, self.xyz_dim, self.extract_dim).to(self.device)
        self.output_process = OutputProcess(self.data_rep, self.xyz_dim, self.extract_dim, self.pcd_points).to(self.device)
        
    def forward(self, x, mask, timesteps, given_objs, given_cats, y=None, force_mask=False):
        """
        x: noisy signal - torch.Tensor.shape([bs, seq_len, dims, cat]). E.g, 1, 256, 655, 8
        vertices: torch.Tensor.shape([bs, seq_len, dim, 3])
        mask: torch.Tensor.shape([bs, seq_len])
        timesteps: torch.Tensor.shape([bs,])
        y: modality, e.g., text
        """
        # Embed features from time
        emb_ts = self.embed_timestep(timesteps)
        emb_ts = emb_ts.permute(1, 0, 2)

        # Embed features from modality
        if self.modality == 'text':
            enc_text = self._encode_text(y)
            # Pass through linear layer of text
            enc_text = self.embed_text(enc_text)
            # enc_text = self.embed_text(self._mask_cond(enc_text, force_mask=force_mask))
            enc_text = enc_text.unsqueeze(1)

        # Predict output categorical
        out_cat = self.predict_cat(enc_text.clone().detach())
        self.saved_cat = out_cat

        # Embed information from categories
        emb_cat = self.embed_cat(given_cats)
        
        # Combine features from timestep and modality
        emb = torch.cat((emb_ts, enc_text), dim=-1)
        emb = emb.permute(0, 2, 1)
        emb = self.upsampling_layer(emb)
        emb = emb.permute(0, 2, 1)

        # Embed point clouds feature
        bs, num_obj, num_points, pcd_dim = given_objs.shape

        # Get human pose features
        hm_in = given_objs[:,0].clone().detach()
        given_objs = given_objs.view(bs * num_obj, num_points, pcd_dim)
        hm_out = self.human_backbone(hm_in)
        pcd_out = self.pcd_backbone(given_objs)
        pcd_out = pcd_out.reshape(bs, num_obj, -1)

        # Pass through attention layer to attain attention matrix
        attn_mask = mask.unsqueeze(1).clone().detach()
        attn_mask = attn_mask.repeat(self.n_head, 1, 1)
        attn_output, attn_output_weights = self.attn_layer(enc_text, emb_cat, pcd_out, attn_mask=attn_mask)
        
        # Pass through translation layer
        enc_text = enc_text.repeat(1, num_obj, 1)
        emb_cat = torch.cat((emb_cat, enc_text), dim=-1)
        translation_output = self.translation_layer(emb_cat).unsqueeze(-2).repeat(1, 1, self.pcd_points, 1)
        translation_output = translation_output.view(-1, self.pcd_points, self.translation_params)

        # Pass through point cloud backbone and retrieve spatial relation
        pcd_out = pcd_out.permute(0, 2, 1)
        pcd_out = pcd_out * attn_output_weights
        pcd_out = pcd_out.reshape(bs, num_obj, num_points, -1)
        pcd_trans = pcd_out.view(-1, self.pcd_points, self.xyz_dim)
        pcd_trans, _ = self.pcd_attention(translation_output, pcd_trans, pcd_trans)
        pcd_trans = pcd_trans.view(bs, num_obj, num_points, -1)
        pcd_out = torch.cat((pcd_out, pcd_trans), dim=-1)
        pcd_out = self.point_wise_trans_layer(pcd_out)
        pcd_out = pcd_out.reshape(num_points, -1, bs, num_obj)
        pcd_out = pcd_out * mask
        pcd_out = pcd_out.reshape(bs, num_obj, num_points, -1)
        pcd_out = pcd_out.sum(dim=1)
        pcd_out = (pcd_out + hm_out)/2
        x += pcd_out

        # Final embedding features
        # emb = torch.cat((emb, pcd_out), dim=-1)
        emb = self.combine_extraction(emb)

        # Reconstruct features
        x = self.input_process(x, emb)
        x = self.output_process(x)
        return out_cat, x

    def _set_up_modality(self):
        assert self.modality in ['text', 'audio', None]
        if self.modality == 'text':
            self.embed_text = nn.Sequential(
                nn.Linear(self.clip_dim, self.clip_dim//2),
                nn.GELU(),
                nn.Linear(self.clip_dim//2, self.latent_dim),
                nn.GELU()
            ).to(self.device)
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