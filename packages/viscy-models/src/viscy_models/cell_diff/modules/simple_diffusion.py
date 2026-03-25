from typing import Optional
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class Upsample(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        factor: int = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_cubic = factor ** 3

        dim_out = default(dim_out, dim)
        assert isinstance(dim_out, int) and dim_out > 0, 'dim_out must be a positive integer'
        conv = nn.Conv3d(dim, dim_out * self.factor_cubic, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            PixelShuffle3d(factor)
        )

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    dim_out = default(dim_out, dim)
    assert isinstance(dim_out, int) and dim_out > 0, 'dim_out must be a positive integer'
    return nn.Sequential(
        Rearrange('b c (d p0) (h p1) (w p2) -> b (c p0 p1 p2) d h w', p0 = factor,  p1 = factor, p2 = factor),
        nn.Conv3d(dim * (factor ** 3), dim_out, 1)
    )

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()        
        self.proj = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            assert isinstance(scale_shift, tuple) and len(scale_shift) == 2, 'scale and shift must be a tuple of len 2'
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)