# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
import os
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation

# Local path to the encoder weights (relative to this file's directory)
_ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "data"))
HEIGHTSCAN_ENCODER_PATH = os.path.join(_ASSETS_DIR, "Policies", "heightscan_encoder", "vae_heightscan3.pth")
    
class VAESampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAESampler, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv = Conv2dNormActivation(input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Convolutional Layers for 2D mean and logvar
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        )
        
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.conv(x)
        x = self.mean_layers(x)
        return x
    
class EncoderFPN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderFPN, self).__init__()
        encoder = regnet_x_400mf(weights=None)
        # Remove classification head from the encoder
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        # Modify the first layer to accept the number of channels in the input image
        encoder[0][0] = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channel)

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in the child class.")
        
# Depth image encoder
class HeightScanEncoder(EncoderFPN):
    def __init__(self, out_channel):
        super(HeightScanEncoder, self).__init__(1, out_channel)
        
    def forward(self, x):
        # check if depth has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        out = OrderedDict()
        x = self.enc(x)
        out['feat1'] = self.enc_1(x)
        out['feat2'] = self.enc_2(out['feat1'])
        out['feat3'] = self.enc_3(out['feat2'])
        
        out = self.fpn(out)
        
        return out['feat1']
    
class VAEDecoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.conv = Conv2dNormActivation(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.decoder = nn.Sequential(
            # Layer 0
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Layer 1
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            # Final Output Layer
            nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1)
        ) # Expand the spatial dimensions by a factor of 2**4=16
    
    def forward(self, z):
        z =  self.conv(z)
        img = self.decoder(z)
        return img
    
class HeightScanDecoder(VAEDecoder):
    def __init__(self, input_dim):
        super(HeightScanDecoder, self).__init__(input_dim, 1)
    
class VAENet(nn.Module):
    def __init__(self, latent_dim):
        super(VAENet, self).__init__()
        self.depth_encoder = HeightScanEncoder(latent_dim)
        
        self.vae_sampler = VAESampler(latent_dim, latent_dim)

        self.depth_decoder = HeightScanDecoder(latent_dim)
        
    def forward(self, x):
        x = self.depth_encoder(x)
        x = self.vae_sampler(x)
        return x
    
    def decode(self, z):
        return self.depth_decoder(z)

    
class HeightScanFeatEncoder(torch.nn.Module):
    def __init__(self, 
        feature_dim,
    ):
        """
        A Simply PyTorch module to add realistic noise to depth images.
        
        Args:
            feature_dim (int): Number of output channels from the encoder.
        """
        super().__init__()

        self.encoder = VAENet(feature_dim)
        
        try:
            self.encoder.load_state_dict(torch.load(HEIGHTSCAN_ENCODER_PATH, weights_only=True), strict=True)
            print('\033[92m' + f'Successfully loaded pre-trained weights from {HEIGHTSCAN_ENCODER_PATH}' + '\033[0m')
        except Exception as e:
            print('\033[91m' + f'Failed to load pre-trained weights: {e}' + '\033[0m')
        
    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        encoded_scan = self.encoder(scan) # (B, 1, H, W) -> (B, C, H//8, W//8)
        return encoded_scan
    
    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(z)
        
    
# Example usage
if __name__ == "__main__":
    import cv2
    import numpy as np
    
    # parameters for the depth noise
    feature_dim = 64

    # Initialize the DepthNoise class
    heightscan_feat_encoder = HeightScanFeatEncoder(feature_dim=feature_dim)
    
    # compile the depth_noise module
    compiled_scan_encoder = torch.jit.script(heightscan_feat_encoder)

    # random height scan image
    scan = np.random.rand(51, 51) * 5
    
    # convert depth to tensor
    scan = torch.tensor(scan).unsqueeze(0).unsqueeze(0).float()
    clipped_scan = torch.clamp(scan, min=-5, max=5)

    # Add noise to the depth image
    noisy_scan = compiled_scan_encoder(clipped_scan)
    
    noisy_scan = noisy_scan.numpy().squeeze().squeeze()
    # check the min and max values of the noisy depth
    print("Min depth: ", noisy_scan.min())
    print("Max depth: ", noisy_scan.max())
