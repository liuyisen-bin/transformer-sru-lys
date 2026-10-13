"""Batched Torch runtime for Unitree's official G1 velocity ONNX policy."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


UNITREE_POLICY_INPUT_DIM = 480
UNITREE_POLICY_OUTPUT_DIM = 29
UNITREE_POLICY_HISTORY_LENGTH = 5

_BUNDLED_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "data"
    / "Policies"
    / "unitree_g1_velocity"
    / "policy.onnx"
)
_LAYER_SPECS = (
    ("actor.0.weight", "actor.0.bias", 480, 512),
    ("actor.2.weight", "actor.2.bias", 512, 256),
    ("actor.4.weight", "actor.4.bias", 256, 128),
    ("actor.6.weight", "actor.6.bias", 128, 29),
)


def get_unitree_velocity_policy_path(*, validate: bool = True) -> Path:
    """Resolve the bundled official policy, with an explicit-path override."""

    path = Path(os.environ.get("UNITREE_G1_POLICY", str(_BUNDLED_POLICY_PATH))).expanduser()
    if validate and not path.is_file():
        raise FileNotFoundError(
            f"Unitree G1 velocity policy not found: {path}. "
            "Restore the bundled policy or set UNITREE_G1_POLICY."
        )
    return path


class UnitreeVelocityPolicyBatch(nn.Module):
    """Load the official ONNX weights into an equivalent frozen batched MLP.

    Unitree's downloaded ONNX has a fixed input shape of ``[1, 480]``.  Its
    graph is a four-layer ELU MLP.  Reading those exact initializers into
    ``nn.Linear`` layers preserves the policy and enables GPU-batched inference
    for parallel Isaac Lab training environments.
    """

    def __init__(self, policy_path: str | os.PathLike[str], device: torch.device | str):
        super().__init__()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError(
                "Unitree velocity policy loading requires the 'onnx' package."
            ) from exc

        self.policy_path = Path(policy_path).expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Unitree velocity policy not found: {self.policy_path}")

        model = onnx.load(str(self.policy_path), load_external_data=True)
        op_types = [node.op_type for node in model.graph.node]
        expected_ops = ["Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
        if op_types != expected_ops:
            raise ValueError(
                "Unsupported Unitree policy graph. Expected the official four-layer "
                f"Gemm/ELU actor, got {op_types}."
            )
        arrays = {
            initializer.name: numpy_helper.to_array(initializer).copy()
            for initializer in model.graph.initializer
        }

        layers: list[nn.Module] = []
        for index, (weight_name, bias_name, input_dim, output_dim) in enumerate(_LAYER_SPECS):
            if weight_name not in arrays or bias_name not in arrays:
                raise ValueError(
                    f"Official Unitree initializer missing: {weight_name} or {bias_name}."
                )
            weight = arrays[weight_name]
            bias = arrays[bias_name]
            if weight.shape != (output_dim, input_dim) or bias.shape != (output_dim,):
                raise ValueError(
                    f"Unexpected {weight_name}/{bias_name} shapes: {weight.shape}, {bias.shape}."
                )
            layer = nn.Linear(input_dim, output_dim)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight))
                layer.bias.copy_(torch.from_numpy(bias))
            layers.append(layer)
            if index != len(_LAYER_SPECS) - 1:
                layers.append(nn.ELU())

        self.actor = nn.Sequential(*layers).to(device=device, dtype=torch.float32)
        self.actor.eval()
        self.actor.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != UNITREE_POLICY_INPUT_DIM:
            raise ValueError(
                f"Unitree observations must be [B, {UNITREE_POLICY_INPUT_DIM}], "
                f"got {tuple(observations.shape)}."
            )
        return self.actor(observations.to(dtype=torch.float32))


__all__ = [
    "UNITREE_POLICY_HISTORY_LENGTH",
    "UNITREE_POLICY_INPUT_DIM",
    "UNITREE_POLICY_OUTPUT_DIM",
    "UnitreeVelocityPolicyBatch",
    "get_unitree_velocity_policy_path",
]
