import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(MlpBlock, self).__init__()        
        self.p = 0.0
        
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(self.p),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(self.p)
        )

    def forward(self, x):
        out = self.net(x)
        
        return out

class MixerBlock(nn.Module):
    """Mixer block layer."""
    def __init__(self, dim, num_patch, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('i j k -> i k j'),
            MlpBlock(num_patch, tokens_mlp_dim),
            Rearrange('i k j -> i j k')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MlpBlock(dim, channels_mlp_dim)
        )

    def forward(self, x):
        tmp = x + self.token_mix(x)
        out = tmp + self.channel_mix(tmp)

        return out

class MlpMixer(nn.Module):
    """Mixer architecture."""
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        assert image_size % patch_size == 0, 'Image size not divisible by patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.mixer_blocks = nn.ModuleList([])
        for i in range(depth):
            blk = MixerBlock(dim, self.num_patch, tokens_mlp_dim, channels_mlp_dim)
            self.mixer_blocks.append(blk)

        self.layer_norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(nn.Linear(dim, num_classes))
    
    
    def forward(self, x):
        x = self.stem(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x