# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Monkey-patches for Isaac Lab terrain system to support navigation tasks.

This module patches TerrainGenerator and TerrainImporter to:
1. Store height field data for goal/spawn sampling
2. Use optimized mesh generation to reduce GPU memory

The patches add these attributes to TerrainImporter (accessed via env.scene.terrain):
- _height_field_visual: Heights for Z-lookup (num_terrains, W, H)
- _height_field_valid_mask: Valid goal positions with safety padding
- _height_field_platform_mask: Platform positions for curriculum
- _height_field_spawn_mask: Valid spawn positions with larger padding
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_generator import TerrainGenerator
    from isaaclab.terrains.terrain_importer import TerrainImporter

# Flag to track if patches have been applied
_patches_applied = False

# Shared storage for passing height field data from TerrainGenerator to TerrainImporter
# Works because TerrainImporter creates TerrainGenerator synchronously in __init__
_height_field_storage = {
    "height_field_visual": None,
    "height_field_valid_mask": None,
    "height_field_platform_mask": None,
    "height_field_spawn_mask": None,
}


def apply_terrain_patches():
    """Apply monkey-patches to Isaac Lab terrain classes.

    Patches applied:
    1. height_field_to_mesh - optimized mesh generation (reduces GPU memory)
    2. TerrainGenerator - collects height field data for goal sampling
    3. TerrainImporter - stores height field data as attributes
    4. PinholeCameraPatternCfg - adds from_ros_camera_info convenience method

    Safe to call multiple times - patches are only applied once.
    """
    global _patches_applied
    if _patches_applied:
        return

    _patch_height_field_to_mesh()
    _patch_terrain_generator()
    _patch_terrain_importer()
    _patch_pinhole_camera_pattern_cfg()

    _patches_applied = True


def _convert_height_field_to_mesh_with_optimization_dynamic(
    height_field: np.ndarray, horizontal_scale: float, vertical_scale: float, block_size: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a height-field array to a triangle mesh, optimizing large flat ground blocks (20x20, 10x10, 5x5).

    This function optimizes mesh generation by detecting flat regions and simplifying them
    to just 2 triangles instead of generating full detail. It uses a hierarchical approach:
    - 20x20 blocks are checked first
    - Non-flat blocks are subdivided into 10x10, then 5x5
    - Only 5x5 or smaller blocks with height variation get full detail

    This can dramatically reduce vertex count for terrains with large flat areas,
    significantly reducing GPU memory usage during simulation.

    Args:
        height_field: The input height-field array.
        horizontal_scale: The discretization of the terrain along the x and y axis.
        vertical_scale: The discretization of the terrain along the z axis.
        block_size: Initial block size for optimization (default 20).

    Returns:
        The vertices and triangles of the mesh:
        - **vertices** (np.ndarray(float)): Array of shape (num_vertices, 3).
        - **triangles** (np.ndarray(int)): Array of shape (num_triangles, 3).
    """
    num_rows, num_cols = height_field.shape
    vertices = []
    triangles = []
    vertex_count = 0

    def process_block(i, j, block_size):
        """Process a block and decide whether to simplify or subdivide into smaller blocks."""
        nonlocal vertex_count

        # Skip if starting position is at or beyond the terrain bounds
        if i >= num_rows or j >= num_cols:
            return

        # Determine block dimensions within bounds of the height field
        block_end_row = min(i + block_size + 1, num_rows)  # +1 to handle the right and bottom edges
        block_end_col = min(j + block_size + 1, num_cols)  # +1 to handle the right and bottom edges

        # Extract the block of height data
        block = height_field[i:block_end_row, j:block_end_col]

        # Skip empty or single-element blocks (can't form triangles)
        if block.size == 0 or block.shape[0] < 2 or block.shape[1] < 2:
            return

        # Check if the entire block is flat (all heights are the same)
        if np.all(block == block[0, 0]):
            # Simplify the block by using two large triangles for the whole flat region
            v0 = [i * horizontal_scale, j * horizontal_scale, block[0, 0] * vertical_scale]
            v1 = [block_end_row * horizontal_scale, j * horizontal_scale, block[0, 0] * vertical_scale]
            v2 = [i * horizontal_scale, block_end_col * horizontal_scale, block[0, 0] * vertical_scale]
            v3 = [block_end_row * horizontal_scale, block_end_col * horizontal_scale, block[0, 0] * vertical_scale]

            # Add the vertices for the large triangles
            vertices.extend([v0, v1, v2, v3])

            # Add the two triangles for the block
            triangles.append([vertex_count, vertex_count + 1, vertex_count + 2])
            triangles.append([vertex_count + 1, vertex_count + 3, vertex_count + 2])

            vertex_count += 4
        else:
            # If block is not flat and the block size is larger than 5x5, subdivide into smaller blocks
            if block_size > 5:
                half_size = block_size // 2
                # Process each of the four quadrants of the block
                process_block(i, j, half_size)
                process_block(i + half_size, j, half_size)
                process_block(i, j + half_size, half_size)
                process_block(i + half_size, j + half_size, half_size)
            else:
                # If block size is 5x5 or smaller, generate detailed triangles for each grid point
                for x in range(block_end_row - i):
                    for y in range(block_end_col - j):
                        # Get the height at this point
                        z = block[x, y] * vertical_scale
                        v = [(i + x) * horizontal_scale, (j + y) * horizontal_scale, z]
                        vertices.append(v)

                # Now create triangles for all internal points including the last row and last column
                for x in range(block_end_row - i - 1):  # Handle rows, including the last row
                    for y in range(block_end_col - j - 1):  # Handle columns, including the last column
                        ind0 = vertex_count + x * (block_end_col - j) + y
                        ind1 = ind0 + 1
                        ind2 = ind0 + (block_end_col - j)
                        ind3 = ind2 + 1

                        # Create two triangles for this grid cell
                        triangles.append([ind0, ind3, ind1])
                        triangles.append([ind0, ind2, ind3])

                vertex_count += (block_end_row - i) * (block_end_col - j)

    # Start by processing blocks with the initial block size (20x20)
    for i in range(0, num_rows, block_size):
        for j in range(0, num_cols, block_size):
            process_block(i, j, block_size)

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    # Return vertices and triangles arrays
    return vertices, triangles


def _patch_height_field_to_mesh():
    """Patch height_field_to_mesh decorator to use optimized mesh generation.

    This patch replaces the mesh generation in the height_field_to_mesh decorator
    with an optimized version that reduces vertex count for flat terrain regions,
    significantly reducing GPU memory usage during simulation with many environments.
    """
    import copy
    import functools
    from isaaclab.terrains.height_field import utils as hf_utils

    def _patched_height_field_to_mesh(func):
        """Patched decorator that uses optimized mesh generation."""
        @functools.wraps(func)
        def wrapper(difficulty: float, cfg):
            # check valid border width
            if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
                raise ValueError(
                    f"The border width ({cfg.border_width}) must be greater than or equal to the"
                    f" horizontal scale ({cfg.horizontal_scale})."
                )
            # allocate buffer for height field (with border)
            width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
            length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
            border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
            heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
            # override size of the terrain to account for the border
            sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
            sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]
            # update the config
            terrain_size = copy.deepcopy(cfg.size)
            cfg.size = tuple(sub_terrain_size)
            # generate the height field
            z_gen = func(difficulty, cfg)
            # handle the border for the terrain
            heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
            # set terrain size back to config
            cfg.size = terrain_size

            # PATCH: Use optimized mesh generation to reduce GPU memory usage
            vertices, triangles = _convert_height_field_to_mesh_with_optimization_dynamic(
                heights, cfg.horizontal_scale, cfg.vertical_scale, 20
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

            # compute origin
            x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
            x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
            y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
            y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
            origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
            origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
            # return mesh and origin
            return [mesh], origin

        return wrapper

    # Apply patch to the height_field_to_mesh decorator
    hf_utils.height_field_to_mesh = _patched_height_field_to_mesh


def _patch_terrain_generator():
    """Patch TerrainGenerator to collect height field data for goal sampling."""
    from isaaclab.terrains.terrain_generator import TerrainGenerator
    from isaaclab.terrains.utils import color_meshes_by_height
    from isaaclab.utils.timer import Timer
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.dict import dict_to_md5_hash

    # Height field attribute names we collect
    _HEIGHT_FIELD_ATTRS = [
        "height_field_visual",
        "height_field_valid_mask",
        "height_field_platform_mask",
        "height_field_spawn_mask",
    ]

    def _patched_init(self, cfg, device: str = "cpu"):
        """Patched __init__ that collects height field data."""
        # Validate inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified in the configuration!")
        if cfg.curriculum and cfg.difficulty_range is None:
            raise ValueError("Curriculum learning requires 'difficulty_range' to be specified!")

        # Store inputs
        self.cfg = cfg
        self.device = device

        # Set common values for all sub-terrains
        for sub_cfg in self.cfg.sub_terrains.values():
            sub_cfg.size = self.cfg.size
            sub_cfg.horizontal_scale = self.cfg.horizontal_scale
            sub_cfg.vertical_scale = self.cfg.vertical_scale
            sub_cfg.slope_threshold = self.cfg.slope_threshold

        # Set seed for reproducibility
        # - seed=None (default): random terrain each time (for training variety)
        # - seed=<int>: reproducible terrain (for debugging/evaluation)
        # Note: Always create RNG because Isaac Lab's base code uses self.np_rng
        self.np_rng = np.random.default_rng(self.cfg.seed)
        self._reproducible = self.cfg.seed is not None

        # Initialize buffers
        self.flat_patches = {}
        self.terrain_meshes = []
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # PATCH: Temporary lists to collect height field data during generation
        self._height_field_lists = {attr: [] for attr in _HEIGHT_FIELD_ATTRS}

        # Generate terrains
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                self._generate_random_terrains()

        # Add border and combine meshes
        self._add_terrain_border()
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # PATCH: Consolidate collected height fields into tensors and store in shared storage
        for attr in _HEIGHT_FIELD_ATTRS:
            data_list = self._height_field_lists[attr]
            if data_list:
                setattr(self, attr, torch.cat(data_list, dim=0))
                _height_field_storage[attr] = getattr(self, attr)
            else:
                setattr(self, attr, None)
                _height_field_storage[attr] = None
        del self._height_field_lists  # Free memory

        # Color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme != "none":
            raise ValueError(f"Unknown color scheme: {self.cfg.color_scheme}")

        # Move flat patches to device
        for name in self.flat_patches:
            self.flat_patches[name] = self.flat_patches[name].to(self.device)

    def _patched_get_terrain_mesh(self, difficulty: float, cfg) -> tuple:
        """Patched _get_terrain_mesh that collects height field data from each terrain."""
        # Copy configuration and set parameters
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed

        # Clear non-serializable fields for hashing
        for attr in _HEIGHT_FIELD_ATTRS:
            if hasattr(cfg, attr):
                setattr(cfg, attr, None)
        # Clear RNG for hashing (seed already included, RNG is derived from it)
        if hasattr(cfg, 'rng'):
            cfg.rng = None

        # Generate hash and cache paths
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # Check cache
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            return mesh, origin

        # Set RNG for terrain generation (after hashing, before generation)
        # Only pass RNG if seed was explicitly set (for reproducibility)
        # Otherwise cfg.rng stays None and terrain uses fresh random each time
        if hasattr(cfg, 'rng') and self._reproducible:
            cfg.rng = self.np_rng.spawn(1)[0]

        # Generate mesh
        meshes, origin = cfg.function(difficulty, cfg)
        if not isinstance(meshes, list):
            meshes = [meshes]
        mesh = trimesh.util.concatenate(meshes)

        # Center the mesh
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]

        # PATCH: Collect height field data from cfg (set by terrain generation function)
        for attr in _HEIGHT_FIELD_ATTRS:
            if hasattr(cfg, attr):
                data = getattr(cfg, attr)
                if data is not None:
                    self._height_field_lists[attr].append(data)
                    setattr(cfg, attr, None)  # Clear after collecting

        # Cache if enabled
        if self.cfg.use_cache:
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)

        return mesh, origin

    # Apply patches
    TerrainGenerator.__init__ = _patched_init
    TerrainGenerator._get_terrain_mesh = _patched_get_terrain_mesh


def _patch_terrain_importer():
    """Patch TerrainImporter to store height field data as attributes.

    After TerrainGenerator populates shared storage, we capture the data
    and store it as attributes on TerrainImporter for access via env.scene.terrain.
    """
    from isaaclab.terrains.terrain_importer import TerrainImporter

    _original_importer_init = TerrainImporter.__init__

    # Attribute names (with underscore prefix for storage on importer)
    _HEIGHT_FIELD_ATTRS = [
        "height_field_visual",
        "height_field_valid_mask",
        "height_field_platform_mask",
        "height_field_spawn_mask",
    ]

    def _patched_importer_init(self, cfg):
        """Patched __init__ that captures height field data from TerrainGenerator."""
        # Clear previous storage
        for attr in _HEIGHT_FIELD_ATTRS:
            _height_field_storage[attr] = None

        # Call original __init__ - creates TerrainGenerator which populates storage
        _original_importer_init(self, cfg)

        # Capture height field data from storage and store as attributes
        for attr in _HEIGHT_FIELD_ATTRS:
            setattr(self, f"_{attr}", _height_field_storage.get(attr))

    TerrainImporter.__init__ = _patched_importer_init


def _patch_pinhole_camera_pattern_cfg():
    """Patch PinholeCameraPatternCfg to add from_ros_camera_info method."""
    from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PinholeCameraPatternCfg

    # Check if already patched
    if hasattr(PinholeCameraPatternCfg, 'from_ros_camera_info'):
        return

    # Add downsample_factor attribute if not present
    if not hasattr(PinholeCameraPatternCfg, 'downsample_factor'):
        PinholeCameraPatternCfg.downsample_factor = 1

    # Store original from_intrinsic_matrix
    _original_from_intrinsic_matrix = PinholeCameraPatternCfg.from_intrinsic_matrix

    @classmethod
    def _patched_from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        focal_length: float = 1.0,
        downsample_factor: int = 1,
    ) -> PinholeCameraPatternCfg:
        """Create a PinholeCameraPatternCfg from an intrinsic matrix with downsampling support.

        Args:
            intrinsic_matrix: The intrinsic matrix as a 9-element list [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1].
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            focal_length: Focal length of the camera (in cm). Defaults to 1.0 cm.
            downsample_factor: Downsampling factor for RL training. Defaults to 1 (no downsampling).

        Returns:
            An instance of the PinholeCameraPatternCfg class.
        """
        # Extract standard intrinsic parameters (in pixels)
        f_x = intrinsic_matrix[0]
        c_x = intrinsic_matrix[2]
        f_y = intrinsic_matrix[4]
        c_y = intrinsic_matrix[5]

        # Apply downsampling adjustments
        if downsample_factor > 1:
            f_x = f_x / downsample_factor
            f_y = f_y / downsample_factor
            c_x = c_x / downsample_factor
            c_y = c_y / downsample_factor
            width = width // downsample_factor
            height = height // downsample_factor

        # Convert to USD camera parameters
        horizontal_aperture = width * focal_length / f_x
        vertical_aperture = height * focal_length / f_y

        # Convert principal point offset from pixels to physical units (cm)
        horizontal_aperture_offset = (c_x - width / 2) * horizontal_aperture / width
        vertical_aperture_offset = (c_y - height / 2) * vertical_aperture / height

        return cls(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
            width=width,
            height=height,
        )

    @classmethod
    def from_ros_camera_info(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        focal_length: float = 1.0,
        downsample_factor: int = 1,
    ) -> PinholeCameraPatternCfg:
        """Create a PinholeCameraPatternCfg from ROS CameraInfo parameters.

        This is a convenience method for creating camera configurations directly from
        ROS camera_info topic parameters, avoiding the need to construct the full
        intrinsic matrix.

        Args:
            fx: Focal length in x-direction (pixels).
            fy: Focal length in y-direction (pixels).
            cx: Principal point x-coordinate (pixels).
            cy: Principal point y-coordinate (pixels).
            width: Image width (pixels).
            height: Image height (pixels).
            focal_length: USD focal length scaling factor (cm). Defaults to 1.0 cm.
            downsample_factor: Downsampling factor for RL training. Defaults to 1 (no downsampling).

        Returns:
            A PinholeCameraPatternCfg instance.

        Example:
            # From your camera info topic:
            # K: [419.663, 0, 425.728, 0, 419.663, 238.272, 0, 0, 1]
            # size: 848 x 480

            # Original resolution
            cfg = PinholeCameraPatternCfg.from_ros_camera_info(
                fx=419.663, fy=419.663, cx=425.728, cy=238.272,
                width=848, height=480
            )

            # 4x downsampled for RL training (212x120)
            cfg_rl = PinholeCameraPatternCfg.from_ros_camera_info(
                fx=419.663, fy=419.663, cx=425.728, cy=238.272,
                width=848, height=480, downsample_factor=4
            )
        """
        # Create intrinsic matrix in row-major format
        intrinsic_matrix = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        return cls.from_intrinsic_matrix(intrinsic_matrix, width, height, focal_length, downsample_factor)

    # Apply patches
    PinholeCameraPatternCfg.from_intrinsic_matrix = _patched_from_intrinsic_matrix
    PinholeCameraPatternCfg.from_ros_camera_info = from_ros_camera_info
