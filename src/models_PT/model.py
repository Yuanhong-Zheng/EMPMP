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

    def forward(self, motion_input):
        #motion_input:b,p,t,d
        b,p,t,d = motion_input.shape
        if self.temporal_fc_in:#! no use
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)#B,P,T,D
            motion_feats=motion_feats.reshape(b,p*t,d)#b,pt,d
            motion_feats = motion_feats.permute(0,2,1)#b,d,pt
        
        motion_feats = self.motion_mlp(motion_feats)#b,d,pt

        if self.temporal_fc_out:#! no use
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = motion_feats.permute(0,2,1)#b,pt,d
            motion_feats = self.motion_fc_out(motion_feats)#B,PT,D
            motion_feats=motion_feats.reshape(b,p,t,d)#b,p,t,d

        return motion_feats

