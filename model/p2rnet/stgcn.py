#  Copyright (c) 8.2021. Yinyu Nie
#  License: MIT

import torch
import torch.nn as nn
from model.p2rnet.stgcn_layers import Graph, st_gcn_block
from model.p2rnet.vn_dgcnn_util import get_graph_offset
from model.p2rnet.sub_modules import SingleConv

class STGCN(nn.Module):
    def __init__(self, joint_num=1024, n_seeds=0, num_frames=1, seed_sampling='uniform', origin_joint_id=0, optim_spec=None):
        '''
        Encode poses to propose boxes.
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(STGCN, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Build graph'''
        self.graph = Graph(layout='virtualroom', strategy='spatial', max_hop=5)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        '''Network modules'''
        self.n_seeds = n_seeds
        self.joint_num = joint_num
        self.num_frames = num_frames
        self.origin_joint_id = origin_joint_id
        in_channels = 2
        out_joint_channels = 2
        out_channels = 3072
        temporal_kernel_size = 3
        edge_importance_weighting = True
        spatial_kernel_size = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {}
        kwargs = {}
        self.knn = 20

        self.pos_embed = nn.Sequential(SingleConv(3, 64, kernel_size=1, order='cbr', padding=0, ndim=1),
                                    #    SingleConv(64, 64, kernel_size=1, order='cbr', padding=0, ndim=1),
                                       SingleConv(64, in_channels, kernel_size=1, order='c', padding=0, ndim=1))
        self.sk_feat = nn.Sequential(SingleConv(3, 64, kernel_size=1, order='cbr', padding=0, ndim=1),
                                    #    SingleConv(64, 64, kernel_size=1, order='cbr', padding=0, ndim=1),
                                       SingleConv(64, in_channels, kernel_size=1, order='c', padding=0, ndim=1))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, out_joint_channels, kernel_size, 1, **kwargs),
        ))

        self.conv_joint = nn.Conv1d(in_channels=self.joint_num * out_joint_channels,
                                    out_channels=out_channels, kernel_size=1)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        '''Fixed params'''
        if self.n_seeds >= self.num_frames:
            self.seed_inds = torch.round(torch.linspace(0, self.num_frames - 1, self.n_seeds)).long()
        else:
            self.seed_sampling = seed_sampling

    def forward(self, input_joints: torch.cuda.FloatTensor, end_points=None):
        input_joints = input_joints.unsqueeze(1)
        n_batch, n_frames, n_joints, n_dim = input_joints.size()
        device = input_joints.device

        '''Get seed xyz and seed inds'''
        origin_joints = input_joints[:, :, self.origin_joint_id]
        if self.n_seeds >= origin_joints.size(1):
            seed_inds = self.seed_inds.repeat(n_batch, 1).to(device)
        else:
            if self.seed_sampling == 'random':
                seed_inds = torch.argsort(torch.rand(size=(n_batch, n_frames)), dim=1)[:, :self.n_seeds]
                seed_inds = torch.sort(seed_inds, dim=1)[0].to(device)
            elif self.seed_sampling == 'uniform':
                movement_dist = torch.norm(torch.diff(origin_joints, dim=1), dim=2)
                cum_dist = torch.cumsum(torch.cat([torch.zeros(size=(n_batch, 1)).to(device), movement_dist], dim=1), dim=1)
                step_len = cum_dist[:, -1] / (self.n_seeds - 1)
                target_cum_dist = step_len.unsqueeze(-1) * torch.arange(self.n_seeds, dtype=torch.float).to(device)
                seed_inds = torch.argmin(torch.abs(cum_dist.unsqueeze(-1) - target_cum_dist.unsqueeze(1)), dim=1)
            else:
                raise NotImplementedError

        '''Get seed features'''
        x = input_joints - input_joints[:, :, [self.origin_joint_id]]

        # get pose index
        idx = torch.arange(0, n_frames).unsqueeze(0).unsqueeze(-1).expand(n_batch, n_frames, self.knn)
        idx_add = torch.arange(-self.knn//2, self.knn//2).unsqueeze(0).unsqueeze(0).expand_as(idx)
        idx = idx + idx_add
        idx[idx < 0] = 0
        idx[idx >= n_frames] = n_frames - 1
        idx = idx.to(device)
        # get rel position embedding
        rel_dist_matrix = get_graph_offset(origin_joints.transpose(1,2), idx=idx)
        rel_dist_matrix = rel_dist_matrix.squeeze(3)
        pos_embed = self.pos_embed(rel_dist_matrix.view(n_batch, n_frames * self.knn , -1).transpose(1, 2))
        pos_embed = pos_embed.transpose(1,2).contiguous()
        pos_embed = pos_embed.view(n_batch, n_frames, self.knn, -1)
        pos_embed = pos_embed.mean(dim=2)

        # get seed feature embedding
        seed_features = self.sk_feat(x.view(n_batch, n_frames * n_joints, -1).transpose(1, 2))
        seed_features = seed_features.transpose(1,2).contiguous()
        seed_features = seed_features.view(n_batch, n_frames, n_joints, -1)

        # embed rel pose embedding
        x = seed_features + pos_embed.unsqueeze(2)
        x = x.permute(0, 3, 1, 2).contiguous()

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        joint_dim = x.size(1)
        x = x.transpose(2, 3).contiguous()
        x = x.view(n_batch, joint_dim * n_joints, n_frames)
        x = self.conv_joint(x)

        x = x.view(n_batch, n_joints, -1)
        return x

if __name__ == '__main__':
    model = STGCN().cuda()
    x = torch.rand(6, 1024, 3).cuda()
    x = model(x)
    print(x.data.shape)