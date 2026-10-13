"""Lightweight PointNet encoder for fixed-size LiDAR scans."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PointNetEncoder(nn.Module):
    """Encode a batch of XYZ point clouds into one global feature vector."""

    def __init__(self, point_dim: int = 3, feature_dim: int = 128, max_batch_size: int = 256):
        super().__init__()
        self.point_dim = point_dim
        self.feature_dim = feature_dim
        self.max_batch_size = max_batch_size

        self.conv1 = nn.Conv1d(point_dim, 64, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.norm3 = nn.GroupNorm(16, 256)
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.activation = nn.ReLU(inplace=True)
        self._training_stage = "full"

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points with shape ``(..., num_points, point_dim)``."""
        is_sequence = points.dim() == 4
        sequence_length = points.size(0)
        batch_size = points.size(1)
        points = points.reshape(-1, points.shape[-2], self.point_dim)

        if torch.jit.is_scripting():
            features = self._encode_chunk(points)
        else:
            features = self._encode_in_chunks(points)

        if is_sequence:
            return features.reshape(sequence_length, batch_size, self.feature_dim)
        return features

    def _encode_chunk(self, points: torch.Tensor) -> torch.Tensor:
        """Encode one memory-bounded batch of point clouds."""
        valid = torch.any(points != 0.0, dim=-1)

        features = points.transpose(1, 2)
        features = self.activation(self.norm1(self.conv1(features)))
        features = self.activation(self.norm2(self.conv2(features)))
        features = self.activation(self.norm3(self.conv3(features)))

        # Ignore zero-filled rays. Handle an entirely empty scan explicitly.
        features = features.masked_fill(~valid.unsqueeze(1), -1.0e4)
        features = features.max(dim=-1).values
        features = torch.where(valid.any(dim=-1, keepdim=True), features, torch.zeros_like(features))
        return self.projection(features)

    @torch.jit.unused
    def _encode_in_chunks(self, points: torch.Tensor) -> torch.Tensor:
        """Limit activation peaks and checkpoint trainable PointNet chunks."""
        if torch.onnx.is_in_onnx_export() or points.size(0) <= self.max_batch_size:
            return self._encode_chunk(points)

        outputs = []
        trainable = torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.parameters())
        for start in range(0, points.size(0), self.max_batch_size):
            point_chunk = points[start : start + self.max_batch_size]
            if trainable:
                output = checkpoint(self._encode_chunk, point_chunk, use_reentrant=False)
            else:
                output = self._encode_chunk(point_chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def partial_parameters(self) -> list[nn.Parameter]:
        """Parameters trained during the middle curriculum stage."""
        return list(self.conv3.parameters()) + list(self.norm3.parameters()) + list(self.projection.parameters())

    def set_training_stage(self, stage: str) -> None:
        """Apply ``frozen``, ``partial``, or ``full`` trainability."""
        if stage not in {"frozen", "partial", "full"}:
            raise ValueError(f"Unsupported PointNet training stage: {stage}")
        self._training_stage = stage

        for parameter in self.parameters():
            parameter.requires_grad_(stage == "full")

        if stage == "frozen":
            nn.Module.train(self, False)
        elif stage == "partial":
            for parameter in self.partial_parameters():
                parameter.requires_grad_(True)
            nn.Module.train(self, True)
            self.conv1.eval()
            self.norm1.eval()
            self.conv2.eval()
            self.norm2.eval()
        else:
            nn.Module.train(self, True)

    @property
    def training_stage(self) -> str:
        return self._training_stage
