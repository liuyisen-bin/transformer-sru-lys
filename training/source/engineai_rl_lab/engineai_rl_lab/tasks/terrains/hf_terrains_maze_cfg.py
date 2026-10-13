# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze height field terrains."""

from dataclasses import MISSING
from typing import Any, Optional

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

from . import hf_terrains_maze


@configclass
class HfMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a maze height field terrain.

    This terrain generates a procedural maze with configurable wall structures,
    obstacles, and optional stairs. The maze can be used for navigation tasks
    with various difficulty levels.

    Height Field Data (set during terrain generation):
        - height_field_visual: Heights for Z-lookup (num_terrains, W, H)
        - height_field_valid_mask: Valid goal positions with safety padding
        - height_field_platform_mask: Platform positions for curriculum
        - height_field_spawn_mask: Valid spawn positions with larger padding
    """

    function = hf_terrains_maze.maze_terrain

    # =========================================================================
    # Height Field Storage (populated during terrain generation)
    # =========================================================================

    height_field_visual: torch.Tensor = None
    """Height field for Z-lookup (actual terrain heights)."""

    height_field_valid_mask: torch.Tensor = None
    """Boolean mask of valid goal positions (padded with GOAL_PADDING)."""

    height_field_platform_mask: torch.Tensor = None
    """Boolean mask of platform positions for curriculum learning."""

    height_field_spawn_mask: torch.Tensor = None
    """Boolean mask of valid spawn positions (larger padding for robot body)."""

    # =========================================================================
    # Maze Generation Parameters
    # =========================================================================

    maze: bool = True
    """Flag indicating this is a maze terrain."""

    open_probability: float = None
    """Probability of a cell being open in the maze."""

    grid_size: tuple[int, int] = (15, 15)
    """Size of the maze grid (number of cells in width and height)."""

    cell_size: float = 2.0
    """Size of each cell in the maze grid (in meters)."""

    wall_height: float = 1.5
    """Height of the walls (in meters). Defaults to 1.5."""

    # =========================================================================
    # Terrain Features
    # =========================================================================

    add_goal: Any = MISSING
    """Enable goal sampling data generation."""

    add_noise_to_flat: Any = MISSING
    """Add noise to flat areas of the maze."""

    randomize_wall: Any = MISSING
    """Use randomized obstacle shapes instead of full walls."""

    random_wall_ratio: float = 0.5
    """Mix ratio between randomized and standard walls. Defaults to 0.5."""

    non_maze_terrain: bool = False
    """Use non-maze terrain with random obstacles. Defaults to False."""

    stairs: bool = False
    """Add stairs to empty map. Defaults to False."""

    add_stairs_to_maze: bool = False
    """Add stairs to the maze. Defaults to False."""

    dynamic_obstacles: bool = False
    """Enable pit/trough obstacles. Defaults to False."""

    # =========================================================================
    # Random Number Generator
    # =========================================================================

    rng: Optional[np.random.Generator] = None
    """Random number generator for reproducible terrain generation.

    Set by the terrain generator (patches.py) before calling the terrain function.
    If None, will create a new unseeded generator (non-reproducible).
    """


