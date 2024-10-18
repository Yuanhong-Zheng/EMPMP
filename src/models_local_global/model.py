import copy

from torch import nn
from .mlp import build_mlps,build_mlps_for_global
from einops.layers.torch import Rearrange

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange('b p n d -> b p d n')
        self.arr1 = Rearrange('b p d n -> b p n d')
        
        self.conv = nn.Conv1d(in_channels=self.config.motion.dim, out_channels=self.config.motion.dim, kernel_size=self.config.motion_mlp.n_p, stride=self.config.motion_mlp.n_p)

        self.motion_mlp = build_mlps(self.config.motion_mlp)
        self.global_motion_mlp = build_mlps_for_global(self.config.motion_mlp)
        
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

    def forward(self, motion_input,global_motion):#global_motion:B,P,T,D
        b,p,t,d=global_motion.shape
        global_motion=global_motion.reshape(b,p*t,d).permute(0,2,1)#B,D,P*T
        global_motion=self.conv(global_motion)#B,D,T
        cond=self.global_motion_mlp(global_motion)#B,D,T
        
        if self.temporal_fc_in:#! no use
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
            motion_feats = self.arr0(motion_feats)#B,P,D,T

        motion_feats = self.motion_mlp(motion_feats,cond)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)#B,P,T,D
            motion_feats = self.motion_fc_out(motion_feats)#B,P,T,D

        return motion_feats

