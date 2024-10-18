import torch
from torch import nn
from einops.layers.torch import Rearrange

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.Linear(dim, 2 * dim),#d->2d
        )
        self.norm = LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(dim, dim)),
        )

    def forward(self, x, x_global):
        """
        x: B, P,D,T
        x_global: B,D,T
        """
        x=x.permute(0,1,3,2)#B,P,T,D
        x_global=x_global.unsqueeze(1).permute(0,1,3,2)#B,1,T,D
        x_global = self.emb_layers(x_global)#b,1,t,2d
        # scale: B,1, t,d / shift: B,1, t, d
        scale, shift = torch.chunk(x_global, 2, dim=-1)
        x = x* (1 + scale) + shift#B.P,D,T
        x = self.out_layers(x)
        x=x.permute(0,1,3,2)#B,P,D,T
        x=self.norm(x)
        return x
class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        #B,P,D,T
        mean = x.mean(axis=-2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-2, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()
        self.fuse=StylizationBlock(dim)
        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        # nn.init.xavier_uniform_(self.fc0.fc.weight)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x,cond):
        #x:b,p,d,t
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        x=x+self.fuse(x,cond)
        return x
    
class MLPblock_for_global(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()
        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        # nn.init.xavier_uniform_(self.fc0.fc.weight)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        #x:b,p,d,t
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        return x
    
class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,is_global=False):
        super().__init__()
        self.is_global=is_global
        if not is_global :
            self.mlps = nn.ModuleList([
                MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for i in range(num_layers)])
        else:
            self.mlps_global = nn.Sequential(*[
                MLPblock_for_global(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
                for i in range(num_layers)])

    def forward(self, x,cond=None):
        if  self.is_global:
            x = self.mlps_global(x)
        else:
            for block in self.mlps:
                x = block(x,cond)
        return x

def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )
def build_mlps_for_global(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
        is_global=True,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    #if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")


