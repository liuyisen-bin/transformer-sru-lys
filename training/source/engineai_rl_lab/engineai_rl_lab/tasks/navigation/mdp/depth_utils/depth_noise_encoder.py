# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Depth noise generation and encoding for navigation tasks."""

import os
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .camera_config import CameraConfig

# Default camera configuration (ZedX Camera for b2w and aow d)
DEFAULT_CAMERA_CONFIG = CameraConfig(
    focal_length=25.0,
    baseline=0.12,
    min_depth=0.25,
    max_depth=10.0,
    depth_encoder_path=os.path.join(
        str(ISAACLAB_ASSETS_DATA_DIR),
        "Policies",
        "RSL-ETHZ/AoW_d/depth_encoder",
        "vae_pretrain_fuse.pth"
    )
)


class VAESampler(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAESampler, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv = Conv2dNormActivation(
            input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Convolutional Layers for 2D mean and logvar
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        )

        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
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
class DepthEncoder(EncoderFPN):
    def __init__(self, out_channel):
        super(DepthEncoder, self).__init__(1, out_channel)

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
        self.conv = Conv2dNormActivation(
            input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

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
        )  # Expand the spatial dimensions by a factor of 2**4=16

    def forward(self, z):
        z = self.conv(z)
        img = self.decoder(z)
        return img


class DepthDecoder(VAEDecoder):
    def __init__(self, input_dim):
        super(DepthDecoder, self).__init__(input_dim, 1)


class VAENet(nn.Module):
    def __init__(self, latent_dim):
        super(VAENet, self).__init__()
        self.depth_encoder = DepthEncoder(latent_dim)

        self.vae_sampler = VAESampler(latent_dim, latent_dim)

        self.depth_decoder = DepthDecoder(latent_dim)

    def forward(self, x):
        x = self.depth_encoder(x)
        x = self.vae_sampler(x)
        return x

    def decode(self, z):
        return self.depth_decoder(z)


class DepthNoise(torch.nn.Module):
    def __init__(
        self,
        focal_length,
        baseline,
        min_depth,
        max_depth,
        filter_size=3,
        inlier_thred_range=(0.01, 0.05),
        prob_range=(0.4, 0.6),
        invalid_disp=1e7
    ):
        """A Simply PyTorch module to add realistic noise to depth images.

        Args:
            focal_length (float): Focal length of the camera (in pixels).
            baseline (float): Baseline distance between stereo cameras (in meters).
            min_depth (float): Minimum depth value after clamping.
            max_depth (float): Maximum depth value after clamping.
            filter_size (int): Kernel size for local mean disparity computation.
            inlier_thred_range (tuple): Threshold range for normalized disparity differences.
            prob_range (tuple): Probability range for matching pixels.
            invalid_disp (float): Invalid disparity
        """
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.invalid_disp = invalid_disp
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = filter_size

        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer('weights', weights.view(1, 1, filter_size, filter_size))
        self.register_buffer('substitutes', substitutes.view(1, 1, filter_size, filter_size))

    def _compute_weights(self, filter_size):
        """Compute weights and substitutes for disparity filtering.

        Args:
            filter_size (int): Kernel size for local mean disparity computation.
        """
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing='ij')
        sqr_radius = x_filter ** 2 + y_filter ** 2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()

        return weights, substitutes

    def filter_disparity(self, disparity):
        """Filter the disparity map using local mean disparity.

        Args:
            disparity (torch.Tensor): Input disparity map tensor of shape (B, C, H, W).
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        prob = (
            torch.rand(B, 1, 1, 1, device=device)
            * (self.prob_range[1] - self.prob_range[0])
            + self.prob_range[0]
        )
        random_mask = (torch.rand(B, 1, H, W, device=device) < prob)

        # Compute mean disparity
        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)

        # Compute differences
        differences = torch.abs(disparity - weighted_disparity)

        # Normalize differences based on current image statistics
        differences_flat = differences.view(B, -1)
        mean_diff = torch.mean(differences_flat, dim=1, keepdim=True)
        std_diff = torch.std(differences_flat, dim=1, keepdim=True) + 1e-6

        normalized_differences_flat = (differences_flat - mean_diff) / std_diff
        normalized_differences = normalized_differences_flat.view_as(differences)

        # Use parameter-based threshold on normalized differences
        threshold = (
            torch.rand(B, 1, 1, 1, device=device)
            * (self.inlier_thred_range[1] - self.inlier_thred_range[0])
            + self.inlier_thred_range[0]
        )
        update_mask = (normalized_differences < threshold) & random_mask

        # Compute output value: round with 1/32 precision
        disparity = torch.round(disparity * 32.0) / 32.0

        # Update output disparity
        output_disparity = torch.where(update_mask, disparity, output_disparity)

        # Apply substitutes to fill neighboring pixels
        filled_values = F.conv2d(update_mask.float() * disparity, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)

        return output_disparity

    def forward(self, depth, add_noise: bool) -> torch.Tensor:
        # correct input shape
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1)  # add channel dimension

        # check dimension (B, 1, H, W)
        assert depth.shape[1] == 1, "Input depth tensor must have shape (B, 1, H, W)."
        assert len(depth.shape) == 4, "Input depth tensor must have shape (B, 1, H, W)."

        if add_noise:
            # Clamp the depth values
            depth = torch.clamp(depth, min=1. / self.invalid_disp)

            # Step 1: Convert depth to disparity
            disparity = self.focal_length * self.baseline / depth

            # Step 2: Filter the disparity map
            filtered_disparity = self.filter_disparity(disparity)

            # Step 3: Recompute depth from disparity
            depth = self.focal_length * self.baseline / filtered_disparity

            # Step 4: Clamp the depth values
            depth[depth < self.min_depth] = 0.0

        # Step 5: Set invalid depth values to 0.0
        depth[depth > self.max_depth] = 0.0

        return depth


class DepthNoiseEncoder(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        camera_config: CameraConfig = None
    ):
        """A Simply PyTorch module to add realistic noise to depth images.

        Args:
            feature_dim (int): Number of output channels from the encoder.
            camera_config (CameraConfig, optional): Camera configuration parameters.
                                                   If None, uses DEFAULT_CAMERA_CONFIG.
        """
        super().__init__()

        # Use provided config or default
        if camera_config is None:
            camera_config = DEFAULT_CAMERA_CONFIG

        self.camera_config = camera_config
        self.depth_noise = DepthNoise(
            focal_length=camera_config.focal_length,
            baseline=camera_config.baseline,
            min_depth=camera_config.min_depth,
            max_depth=camera_config.max_depth
        )

        self.encoder = VAENet(feature_dim)

        try:
            self.encoder.load_state_dict(
                torch.load(camera_config.depth_encoder_path, weights_only=True),
                strict=True
            )
            print(
                '\033[92m'
                + f'Successfully loaded pre-trained weights from {camera_config.depth_encoder_path}'
                + '\033[0m'
            )
        except Exception as e:
            print('\033[91m' + f'Failed to load pre-trained weights: {e}' + '\033[0m')

    def forward(
        self, depth: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_depth = self.depth_noise(depth, add_noise)
        encoded_depth = self.encoder(noisy_depth)  # (B, 1, H, W) -> (B, C, H//8, W//8)
        return encoded_depth, noisy_depth

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(z)
