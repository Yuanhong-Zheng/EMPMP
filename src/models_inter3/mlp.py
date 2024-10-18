import torch
from torch import nn
from einops.layers.torch import Rearrange

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

        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        return x
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, time_dim, num_p,dim):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.Linear(time_dim*num_p, 2 * time_dim),#pt->2t
        )
        self.norm = LN(dim)
        self.out_layers = nn.Sequential(
            zero_module(nn.Linear(time_dim, time_dim)),
        )

    def forward(self, x, x_global):
        """
        x: B, P,D,T
        x_global: B,D,PT
        """
        x_global=x_global.unsqueeze(1)#B,1,D,PT
        x_global = self.emb_layers(x_global)#b,1,d,2t
        # scale: B,1, d, t / shift: B,1, d, t
        scale, shift = torch.chunk(x_global, 2, dim=-1)
        x = x* (1 + scale) + shift#B.P,D,T
        x = self.out_layers(x)
        x=self.norm(x)
        return x
    
class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis,interaction_interval=2,p=3):
        super().__init__()
        self.local_mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])
        self.global_mlps=nn.Sequential(*[
            MLPblock(dim, seq*p, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers//interaction_interval)])
        self.stylization_blocks = nn.ModuleList([
            StylizationBlock(seq, p,dim) for _ in range(num_layers//interaction_interval)
        ])
        self.interaction_interval=interaction_interval
    def forward(self, x,x_global):
        global_step = 0

        # 逐层进行local和global的交互
        for i, local_layer in enumerate(self.local_mlps):
            x = local_layer(x)  # 计算 local MLP

            # 每经过 interaction_interval 层 local_mlp，执行一次 global_mlp 的更新和交互
            if (i + 1) % self.interaction_interval == 0 and global_step < len(self.global_mlps):
                x_global = self.global_mlps[global_step](x_global)  # 动态计算 global MLP
                x = x+self.stylization_blocks[global_step](x, x_global)  # 使用 StylizationBlock 进行交互
                global_step += 1
        
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
        p=args.n_p,
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


