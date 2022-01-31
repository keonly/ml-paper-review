import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(MlpBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class MixerBlock(nn.Module):
    """Mixer block layer."""
    def __init__(self, dim, num_patch, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class MlpMixer(nn.Module):
    """Mixer architecture."""
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        pass
    
    
    def forward(self, x):
        pass