import copy

from torch import nn
from .mlp import build_mlps
from einops.layers.torch import Rearrange
import torch
def cal_proximity(traj):#B,P,T,J,K
    traj=traj[:,:,:,0,:].transpose(1,2)#B,T,P,K
    distances=torch.sqrt(torch.sum((traj.unsqueeze(3) - traj.unsqueeze(2)) ** 2, dim=-1))#B,T,P,P
    return distances
def compute_distances_hierarchical_normalization(positions,zero_score=True):#B,P,T,J,K
    positions=positions[:,:,:,0,:]#B,P,T,K
    B, P, T, K = positions.size()
    
    # Step 1(No use): Compute centroid for each frame (across people) and normalize positions
    centroid = positions.mean(dim=1, keepdim=True)
    normalized_positions = positions - centroid

    # Step 2: Compute pairwise Euclidean distances
    positions_exp1 = normalized_positions.unsqueeze(2)  # Shape: [B, P, 1, T, K]
    positions_exp2 = normalized_positions.unsqueeze(1)  # Shape: [B, 1, P, T, K]
    dist = torch.norm(positions_exp1 - positions_exp2, dim=-1)  # Shape: [B, P, P, T]
    distances = dist.permute(0, 3, 1, 2)  # Shape: [B, T, P, P]

    # Step 3: Standardize distances across P, P dimensions
    if zero_score:
        mean_distances = distances.mean(dim=(2, 3), keepdim=True)
        std_distances = distances.std(dim=(2, 3), keepdim=True)
        standardized_distances = (distances - mean_distances) / (std_distances + 1e-8)
    else:
        standardized_distances = distances
    return standardized_distances  # Shape: [B, T, P, P]

# class siMLPe(nn.Module):
#     def __init__(self, config):
#         self.config = copy.deepcopy(config)
#         super(siMLPe, self).__init__()
#         self.arr0 = Rearrange('b p n d -> b p d n')
#         self.arr1 = Rearrange('b p d n -> b p n d')

#         self.motion_mlp = build_mlps(self.config.motion_mlp)

#         self.temporal_fc_in = config.motion_fc_in.temporal_fc
#         self.temporal_fc_out = config.motion_fc_out.temporal_fc
#         if self.temporal_fc_in:
#             self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
#         else:
#             self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)#!!!!!
#         if self.temporal_fc_out:
#             self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
#         else:
#             self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)#!!!!!!!!!!
#             self.motion_fc_out_t=nn.Linear(self.config.motion.h36m_input_length_dct, config.motion_mlp.out_len)#!!!!!!!!!!
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
#         # nn.init.xavier_uniform_(self.motion_fc_out.weight)

#         nn.init.constant_(self.motion_fc_out.bias, 0)

#     def forward(self, motion_input,traj):
#         distances=compute_distances_hierarchical_normalization(traj,zero_score=False)
#         if self.temporal_fc_in:
#             motion_feats = self.arr0(motion_input)
#             motion_feats = self.motion_fc_in(motion_feats)
#         else:
#             motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
#             motion_feats = self.arr0(motion_feats)#B,P,D,T

#         motion_feats = self.motion_mlp(motion_feats,distances)

#         if self.temporal_fc_out:
#             motion_feats = self.motion_fc_out(motion_feats)
#             motion_feats = self.arr1(motion_feats)
#         else:
#             motion_feats=self.motion_fc_out_t(motion_feats)#B,P,D,T
#             motion_feats = self.arr1(motion_feats)#B,P,T,D
#             motion_feats = self.motion_fc_out(motion_feats)#B,P,T,D

#         return motion_feats


#第二种输出方式
class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)#!!!!!
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, 3*self.config.motion.dim)#!!!!!!!!!!
            # self.motion_fc_out_t=nn.Linear(self.config.motion.h36m_input_length_dct, config.motion_mlp.out_len)#!!!!!!!!!!
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        # nn.init.xavier_uniform_(self.motion_fc_out.weight)

        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input,traj):
        distances=compute_distances_hierarchical_normalization(traj,zero_score=False)
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
            motion_feats = self.arr0(motion_feats)#B,P,D,T

        motion_feats = self.motion_mlp(motion_feats,distances)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            # motion_feats=self.motion_fc_out_t(motion_feats)#B,P,D,T
            
            motion_feats = self.arr1(motion_feats)#B,P,T,D
            b,p,t,d=motion_feats.shape
            motion_feats = self.motion_fc_out(motion_feats)#B,P,T,3D
            motion_feats=motion_feats.reshape(b,p,t,3,d).reshape(b,p,3*t,d)#B,P,3T,D

        return motion_feats


# #第三种输出方式
# class siMLPe(nn.Module):
#     def __init__(self, config):
#         self.config = copy.deepcopy(config)
#         super(siMLPe, self).__init__()
#         self.arr0 = Rearrange('b p n d -> b p d n')
#         self.arr1 = Rearrange('b p d n -> b p n d')

#         self.motion_mlp = build_mlps(self.config.motion_mlp)

#         self.temporal_fc_in = config.motion_fc_in.temporal_fc
#         self.temporal_fc_out = config.motion_fc_out.temporal_fc
#         if self.temporal_fc_in:
#             self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
#         else:
#             self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)#!!!!!
#         if self.temporal_fc_out:
#             self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
#         else:
#             self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)#!!!!!!!!!!
#             self.motion_fc_out_t=nn.Linear(self.config.motion.h36m_input_length_dct, config.motion_mlp.out_len)#!!!!!!!!!!
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
#         # nn.init.xavier_uniform_(self.motion_fc_out.weight)

#         nn.init.constant_(self.motion_fc_out.bias, 0)

#     def forward(self, motion_input,traj):
#         distances=compute_distances_hierarchical_normalization(traj,zero_score=True)
#         if self.temporal_fc_in:
#             motion_feats = self.arr0(motion_input)
#             motion_feats = self.motion_fc_in(motion_feats)
#         else:
#             motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
#             motion_feats = self.arr0(motion_feats)#B,P,D,T

#         motion_feats = self.motion_mlp(motion_feats,distances)

#         if self.temporal_fc_out:
#             motion_feats = self.motion_fc_out(motion_feats)
#             motion_feats = self.arr1(motion_feats)
#         else:
#             motion_feats=self.motion_fc_out_t(motion_feats)#B,P,D,3T
#             motion_feats = self.arr1(motion_feats)#B,P,3T,D
#             b,p,t,d=motion_feats.shape
#             motion_feats = self.motion_fc_out(motion_feats)#B,P,3T,D

#         return motion_feats