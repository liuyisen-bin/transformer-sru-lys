"""Range-image CNN encoder for ordered LiDAR point clouds."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class RangeImageEncoder(nn.Module):
    """Encode an ordered ``vertical x horizontal`` XYZ scan into one feature."""

    def __init__(
        self,
        point_dim: int = 3,
        image_height: int = 16,
        image_width: int = 41,
        feature_dim: int = 128,
        max_distance: float = 5.0,
        max_batch_size: int = 256,
        transformer_num_layers: int = 0,
        transformer_num_heads: int = 4,
        transformer_feedforward_dim: int = 512,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        if transformer_num_layers < 0:
            raise ValueError("transformer_num_layers must be non-negative.")
        if transformer_num_layers > 0 and feature_dim % transformer_num_heads != 0:
            raise ValueError("feature_dim must be divisible by transformer_num_heads.")
        self.point_dim = point_dim
        self.image_height = image_height
        self.image_width = image_width
        self.num_points = image_height * image_width
        self.feature_dim = feature_dim
        self.max_distance = max_distance
        self.max_batch_size = max_batch_size
        self.transformer_num_layers = transformer_num_layers

        self.block1 = nn.Sequential(
            nn.Conv2d(point_dim + 1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ELU(inplace=True),
        )
        token_height = (image_height + 3) // 4
        token_width = (image_width + 3) // 4
        self.num_spatial_tokens = token_height * token_width
        if transformer_num_layers > 0:
            self.positional_embedding = nn.Parameter(
                torch.zeros(1, self.num_spatial_tokens, feature_dim)
            )
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=transformer_num_heads,
                dim_feedforward=transformer_feedforward_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                transformer_layer,
                num_layers=transformer_num_layers,
            )
            self.transformer_norm = nn.LayerNorm(feature_dim)
        else:
            self.register_parameter("positional_embedding", None)
            self.transformer = nn.Identity()
            self.transformer_norm = nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ELU(inplace=True),
        )
        self._training_stage = "full"

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points with shape ``(..., H*W, point_dim)``."""
        is_sequence = points.dim() == 4
        sequence_length = points.size(0)
        batch_size = points.size(1)
        points = points.reshape(-1, points.shape[-2], self.point_dim)
        if points.size(1) != self.num_points:
            raise ValueError(
                f"Range image expects {self.num_points} points, received {points.size(1)}."
            )

        if torch.jit.is_scripting():
            features = self._encode_chunk(points)
        else:
            features = self._encode_in_chunks(points)

        if is_sequence:
            return features.reshape(sequence_length, batch_size, self.feature_dim)
        return features

    def _encode_chunk(self, points: torch.Tensor) -> torch.Tensor:
        xyz_image = points.reshape(-1, self.image_height, self.image_width, self.point_dim)
        xyz_image = xyz_image.permute(0, 3, 1, 2)
        range_image = torch.linalg.vector_norm(xyz_image, dim=1, keepdim=True)
        image = torch.cat([xyz_image, range_image], dim=1) / self.max_distance
        image = image.clamp(min=-1.0, max=1.0)

        features = self.block1(image)
        features = self.block2(features)
        features = self.block3(features)
        if self.transformer_num_layers > 0:
            tokens = features.flatten(2).transpose(1, 2)
            if tokens.shape[1] != self.num_spatial_tokens:
                raise RuntimeError(
                    f"Unexpected range-image token count {tokens.shape[1]}, "
                    f"expected {self.num_spatial_tokens}."
                )
            tokens = tokens + self.positional_embedding.to(tokens.device, tokens.dtype)
            tokens = self.transformer_norm(self.transformer(tokens))
            pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        else:
            pooled = torch.cat(
                [self.avg_pool(features).flatten(1), self.max_pool(features).flatten(1)],
                dim=-1,
            )
        return self.projection(pooled)

    @torch.jit.unused
    def _encode_in_chunks(self, points: torch.Tensor) -> torch.Tensor:
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
        parameters = list(self.block3.parameters()) + list(self.projection.parameters())
        parameters += list(self.transformer.parameters())
        parameters += list(self.transformer_norm.parameters())
        if self.positional_embedding is not None:
            parameters.append(self.positional_embedding)
        return parameters

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"frozen", "partial", "full"}:
            raise ValueError(f"Unsupported Range Image training stage: {stage}")
        self._training_stage = stage

        for parameter in self.parameters():
            parameter.requires_grad_(stage == "full")

        if stage == "frozen":
            nn.Module.train(self, False)
        elif stage == "partial":
            for parameter in self.partial_parameters():
                parameter.requires_grad_(True)
            nn.Module.train(self, True)
            self.block1.eval()
            self.block2.eval()
        else:
            nn.Module.train(self, True)

    @property
    def training_stage(self) -> str:
        return self._training_stage
