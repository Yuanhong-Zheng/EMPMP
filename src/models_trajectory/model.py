import copy

from torch import nn
from .mlp import build_mlps
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)
        self.trajectory_mlp = build_mlps(self.config.motion_mlp)
        
        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            self.trajectory_fc_in = nn.Linear(3, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            self.trajectory_fc_out=nn.Linear(self.config.motion.dim, 3)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)
        
        nn.init.xavier_uniform_(self.trajectory_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.trajectory_fc_out.bias, 0)
    def forward(self, motion_input,trajectory_input):#b,p,t,jk  b,p,t,k

        if self.temporal_fc_in:#!no use
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
            motion_feats = self.arr0(motion_feats)#B,P,D,T
            trajectory_feats=self.trajectory_fc_in(trajectory_input)#B,P,T,D
            trajectory_feats=self.arr0(trajectory_feats)#B,P,D,T

        motion_feats = self.motion_mlp(motion_feats)
        trajectory_feats = self.trajectory_mlp(trajectory_feats)
        
        if self.temporal_fc_out:#!no use
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)#B,P,T,D
            motion_feats = self.motion_fc_out(motion_feats)#B,P,T,D
            
            trajectory_feats = self.arr1(trajectory_feats)#B,P,T,D
            trajectory_feats = self.trajectory_fc_out(trajectory_feats)#B,P,T,3

        return motion_feats,trajectory_feats

