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

def compute_distances_hierarchical(positions):#B,P,T,J,K
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

    first_distance = distances[:,0,:,:] # Shape: [B, P, P]
    first_distance=torch.sum(first_distance,dim=-1)#B,P
    
    return first_distance
def sort_tensor_by_distance(first_distance, tensor_to_sort):
    """
    first_distance: [B, P] - 每个 Batch 在 P 维度进行排序
    tensor_to_sort: [B, P, ...] - 按照排序索引重新排列 P 维度
    """

    # Step 1: 对 first_distance 在 P 维度进行排序，获取排序索引
    _, sorted_indices = torch.sort(first_distance, dim=-1, descending=False)  # 升序排序，可改为 True 为降序

    # Step 2: 创建与 tensor_to_sort 相同形状的新张量
    sorted_tensor = torch.empty_like(tensor_to_sort)  # 创建空张量以提高性能

    # Step 3: 使用索引对 tensor_to_sort 按 P 维度重新排列
    B = tensor_to_sort.shape[0]
    for b in range(B):  # 按 Batch 进行遍历
        for p_new, p_old in enumerate(sorted_indices[b]):
            sorted_tensor[b, p_new] = tensor_to_sort[b, p_old]  # 将 tensor_to_sort 的第 p_old 行填入第 p_new 行

    return sorted_tensor, sorted_indices

def inverse_sort_tensor(sorted_tensor, sorted_indices):
    """
    将根据 sorted_indices 排序后的 sorted_tensor 再次排回原始顺序。

    参数:
    - sorted_tensor: [B, P, ...] - 已按 P 维度重新排列的张量。
    - sorted_indices: [B, P] - 排序时使用的索引。

    返回:
    - original_tensor: [B, P,...] - 排回原始顺序的张量。
    """

    # 获取张量的维度
    B, P = sorted_tensor.shape[:2]

    # 创建存放恢复后结果的新张量
    original_tensor = torch.empty_like(sorted_tensor)

    # Step 1: 计算逆索引 (inverse_indices)
    # inverse_indices[b, sorted_indices[b, i]] = i
    inverse_indices = torch.empty_like(sorted_indices)
    for b in range(B):
        inverse_indices[b, sorted_indices[b]] = torch.arange(P).to(sorted_indices.device)

    # Step 2: 根据逆索引重排 sorted_tensor
    for b in range(B):
        for p_new, p_original in enumerate(inverse_indices[b]):
            original_tensor[b, p_new] = sorted_tensor[b, p_original]

    return original_tensor

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
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        # nn.init.xavier_uniform_(self.motion_fc_out.weight)

        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input,traj):
        distances=compute_distances_hierarchical_normalization(traj,zero_score=True)
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
            motion_feats = self.arr1(motion_feats)#B,P,T,D
            motion_feats = self.motion_fc_out(motion_feats)#B,P,T,D

        return motion_feats


