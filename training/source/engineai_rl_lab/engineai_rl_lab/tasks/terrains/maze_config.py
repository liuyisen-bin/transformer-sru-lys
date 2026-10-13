# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for maze terrains."""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_maze_cfg import HfMazeTerrainCfg

MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,  # Border around the entire terrain grid (not per-tile)
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=True,
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "stairs": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=False,
            random_wall_ratio=1.0,
            non_maze_terrain=False,
            stairs=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,  # Enables pit/trough generation
        ),
    },
)
"""Maze terrain configuration for navigation tasks."""
