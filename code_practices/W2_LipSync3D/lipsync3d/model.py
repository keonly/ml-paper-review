from cv2 import getOptimalNewCameraMatrix
import torch
import torch.nn as nn
from torch.nn import functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class Lipsync3DMesh(nn.Module):
    def __init__(self):
        super().__init__()

        #TODO
        self.leaky_slope  = 0.1
        self.AudioEncoder = nn.Sequential(
            # Define Network Architecture (Hint: Architecture mentioned in the paper, Change in latent space dimensions are as follows)
            nn.Conv1d(in_channels=2, out_channels=72, kernel_size=(3,1), stride=(2,1), padding=(1,0)),    # 2 x 256 x 24 -> 72 x 128 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=72, out_channels=108, kernel_size=(3,1), stride=(2,1), padding=(1,0)),  # 72 x 128 x 24 -> 108 x 64 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=108, out_channels=162, kernel_size=(3,1), stride=(2,1), padding=(1,0)), # 108 x 64 x 24 -> 162 x 32 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=162, out_channels=243, kernel_size=(3,1), stride=(2,1), padding=(1,0)), # 162 x 32 x 24 -> 243 x 16 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=243, out_channels=256, kernel_size=(3,1), stride=(2,1), padding=(1,0)), # 243 x 16 x 24 -> 256 x 8 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=(3,1), stride=(2,1), padding=(1,0)), # 256 x 8 x 24 -> 256 x 4 x 24
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,2), padding=(0,2)), # 256 x 4 x 24 -> 128 x 4 x 13
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=(1,3), stride=(1,2), padding=(0,2)),  # 128 x 4 x 13 -> 64 x 4 x 8
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=(1,3), stride=(1,2), padding=(0,2)),   # 64 x 4 x 8 -> 32 x 4 x 5
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=(1,3), stride=(1,2), padding=(0,2)),   # 32 x 4 x 5 -> 16 x 4 x 4
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=(1,3), stride=(1,2), padding=(0,2)),    # 16 x 4 x 4 -> 8 x 4 x 3
            nn.LeakyReLU(self.leaky_slope),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=(1,3), stride=(1,2), padding=(0,1)),     # 8 x 4 x 3 -> 4 x 4 x 2
            nn.LeakyReLU(self.leaky_slope),
            View([-1, 32]),
        )

        self.GeometryDecoder = nn.Sequential(
            nn.Linear(32, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 1434)
        )

    def forward(self, spec, latentMode=False):
        # spec : B x 2 x 256 x 24
        # texture : B x 3 x 128 x 128

        latent = self.AudioEncoder(spec)
        if latentMode:
            return latent
        geometry_diff = self.GeometryDecoder(latent)

        return geometry_diff