# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Constants for terrain generation and goal sampling.

This module centralizes all height field values and terrain parameters used across
terrain generation and goal sampling code. Using these constants ensures consistency
and makes the height semantics explicit.

Height Field Value Semantics (in discretized units):
====================================================
The height field stores heights as integers where:
    actual_height_meters = height_value * VERTICAL_SCALE

Key height values:
- GROUND: 0 (flat walkable ground at z=0)
- PLATFORM: ~200 (flat raised platforms at ~1.0m, valid for goals)
- WALL: ~300 (obstacles/walls at ~1.5m, always excluded from goals)
- PIT: ~-300 (negative obstacles/troughs at ~-1.5m, always excluded)

Goal Sampling Valid Ranges:
- Ground: -10 to 50 (allows small noise/variation)
- Platform: 150 to 250 (captures platform height with margin)
- Excluded: < -10 (pits) or > 250 (walls)
"""

from dataclasses import dataclass
from typing import Tuple


# =============================================================================
# Scale Factors
# =============================================================================

HORIZONTAL_SCALE: float = 0.1
"""Horizontal resolution of height field in meters per cell."""

VERTICAL_SCALE: float = 0.005
"""Vertical resolution of height field (height_meters = height_value * VERTICAL_SCALE)."""

CELL_SIZE: float = 2.0
"""Size of each maze cell in meters (default: 2m x 2m)."""

CELL_PIXELS: int = int(CELL_SIZE / HORIZONTAL_SCALE)
"""Number of height field pixels per maze cell (20 at default scale)."""


# =============================================================================
# Height Values (in discretized units, multiply by VERTICAL_SCALE for meters)
# =============================================================================

@dataclass(frozen=True)
class HeightValues:
    """Standard height values for terrain features."""

    # Ground level (valid for goals/spawn)
    GROUND: int = 0

    # Platform height: wall_height - 0.5m in vertical scale units
    # = 1.5/0.005 - 0.5/0.005 = 300 - 100 = 200
    PLATFORM: int = 200

    # Wall/obstacle height: 1.5m in vertical scale units
    # = 1.5/0.005 = 300
    WALL: int = 300

    # Pit/trough height (negative obstacle)
    PIT: int = -300

    @property
    def platform_meters(self) -> float:
        """Platform height in meters."""
        return self.PLATFORM * VERTICAL_SCALE

    @property
    def wall_meters(self) -> float:
        """Wall height in meters."""
        return self.WALL * VERTICAL_SCALE


HEIGHTS = HeightValues()
"""Singleton instance of standard height values."""


# =============================================================================
# Goal Sampling Thresholds
# =============================================================================

@dataclass(frozen=True)
class SamplingThresholds:
    """Thresholds for classifying height values during goal/spawn sampling.

    These values match the old implementation for strict height classification.
    """

    # Ground range (strict: 0-10, allows minimal noise)
    GROUND_MIN: int = 0
    GROUND_MAX: int = 10

    # Platform range (strict: 195-205 for stair platforms at ~1.0m)
    PLATFORM_MIN: int = 195
    PLATFORM_MAX: int = 205

    # Wall threshold (anything above 10 is obstacle, but also check platform edges)
    WALL_THRESHOLD: int = 10

    # Pit threshold (anything below 0 is pit/excluded)
    PIT_THRESHOLD: int = 0

    # Edge detection threshold for Sobel filter (old: any edge > 0.0)
    EDGE_THRESHOLD: float = 0.0

    # Obstacle marker value (used to mark excluded cells)
    OBSTACLE_MARKER: int = 110

    # Extended platform range for edge detection (180-220)
    # Used during padding to detect platform edges with safety margin
    PLATFORM_EDGE_MIN: int = 180
    PLATFORM_EDGE_MAX: int = 220


THRESHOLDS = SamplingThresholds()
"""Singleton instance of sampling thresholds."""


# =============================================================================
# Padding and Border Configuration
# =============================================================================

@dataclass(frozen=True)
class PaddingConfig:
    """Configuration for obstacle padding and borders.

    These values match the old implementation for consistent safety margins.
    With horizontal_scale=0.1m/cell, these translate to:
    - GOAL_PADDING (3 cells) = 0.3m around obstacles for goal positions
    - SPAWN_PADDING (8 cells) = 0.8m around obstacles for spawn positions

    The larger spawn padding accounts for:
    - Robot body dimensions (~0.5m x 0.3m for quadrupeds)
    - Random yaw orientation (diagonal ~0.58m requires ~0.3m radius)
    - Platform edge clearance (prevent falling when spawning near stair edges)
    - Controller startup imprecision
    """

    # Padding around obstacles for goals (in height field cells)
    # 5 cells * 0.1m/cell = 0.5m padding (robot just needs to reach)
    GOAL_PADDING: int = 5

    # Larger padding for spawn positions (in height field cells)
    # 6 cells * 0.1m/cell = 0.6m padding
    # This accounts for:
    # - Robot body radius (~0.5m) with worst-case orientation
    # - Additional safety margin for platform edges
    # - Controller startup behavior
    SPAWN_PADDING: int = 6

    # Border exclusion (keep away from terrain edges)
    # Old implementation used 2 layers (inner + outer), we use 2 cells
    BORDER_CELLS: int = 2

    # Pillar expanded footprint size (5x5 cells around pillar center)
    PILLAR_FOOTPRINT: int = 5

    # Pillar safe margin from cell edges
    PILLAR_EDGE_MARGIN: int = 2

    # Height transition detection threshold (height field units)
    # Marks cells with height differences >= this value as transitions
    HEIGHT_TRANSITION_THRESHOLD: int = 100

    # Padding around detected height transitions (in cells)
    HEIGHT_TRANSITION_PADDING: int = 1


PADDING = PaddingConfig()
"""Singleton instance of padding configuration."""


# =============================================================================
# Stair Configuration
# =============================================================================

@dataclass(frozen=True)
class StairConfig:
    """Configuration for stair generation."""

    # Number of steps in a staircase
    NUM_STEPS: int = 5

    # Height of each step in meters
    STEP_HEIGHT_METERS: float = 0.2

    # Grid size for 3x3 stair structures
    STAIR_GRID_SIZE: int = 3

    # Single cell size in pixels (same as CELL_PIXELS)
    SINGLE_CELL_PIXELS: int = 20

    @property
    def step_height_units(self) -> float:
        """Step height in discretized units."""
        return self.STEP_HEIGHT_METERS / VERTICAL_SCALE

    @property
    def step_resolution(self) -> int:
        """Pixels per step (cell_pixels / num_steps)."""
        return self.SINGLE_CELL_PIXELS // self.NUM_STEPS


STAIRS = StairConfig()
"""Singleton instance of stair configuration."""


# =============================================================================
# Obstacle Structure Types
# =============================================================================

class ObstacleType:
    """Enumeration of obstacle structure types for random generation."""

    PILLAR = 0
    BAR = 1
    CROSS = 2
    SHIFTED_BLOCK = 3

    # Total number of obstacle types
    NUM_TYPES = 4


# =============================================================================
# Obstacle Generation Parameters
# =============================================================================

@dataclass(frozen=True)
class ObstacleConfig:
    """Configuration for random obstacle generation.

    Controls the randomness parameters for obstacle shapes, sizes, and pit ratios.
    All probabilities are in [0, 1] range.
    """

    # Height scale range (multiplier for wall_height)
    SCALE_MIN: float = 0.5
    SCALE_MAX: float = 1.5

    # Thickness range (margin cells from obstacle center)
    # Higher thickness = thinner pillar, thicker bar
    # Range 7-9: pillars 2-6 pixels (0.2-0.6m), bars 7-9 pixels (0.7-0.9m)
    THICKNESS_MIN: int = 7
    THICKNESS_MAX: int = 10  # exclusive upper bound

    # Default pit probability when not specified
    DEFAULT_PIT_PROB: float = 0.15

    # Pit environment specific settings
    PITS_BAR_RATIO: float = 0.6  # Ratio of bars vs random obstacles in pit terrain
    PITS_BAR_PIT_PROB: float = 0.75  # Pit probability for bar obstacles
    PITS_RANDOM_PIT_PROB: float = 0.5  # Pit probability for random obstacles

    # Bridge configuration for pit terrain
    BRIDGE_COUNT_MIN: int = 3
    BRIDGE_COUNT_MAX: int = 6  # exclusive upper bound

    # Pit terrain layout (cell indices from grid edge)
    PITS_TRENCH_ROW_OFFSET: int = 2  # Pit trenches at rows [offset, grid_h - offset - 1]
    PITS_EDGE_MARGIN: int = 2  # Margin from grid edges for middle obstacles

    # Obstacle density multipliers (applied to difficulty)
    NON_MAZE_DENSITY: float = 0.5  # For non-maze terrain (increased from 0.35)
    PITS_DENSITY: float = 0.6  # For pit terrain middle area
    STAIRS_OBSTACLE_DENSITY: float = 0.35  # For random obstacles in stair terrain
    STAIRS_PLACEMENT_PROB: float = 0.75  # Probability of placing stairs at valid locations

    # Non-maze terrain pillar weight (higher = more pillars)
    # Default uniform is 0.25 (1/4 types), 0.5 means ~50% pillars
    NON_MAZE_PILLAR_WEIGHT: float = 0.5


OBSTACLES = ObstacleConfig()
"""Singleton instance of obstacle configuration."""


# =============================================================================
# Helper Functions
# =============================================================================

def height_to_meters(height_value: int) -> float:
    """Convert discretized height value to meters."""
    return height_value * VERTICAL_SCALE


def meters_to_height(meters: float) -> int:
    """Convert meters to discretized height value."""
    return int(meters / VERTICAL_SCALE)


def is_valid_ground(height: int) -> bool:
    """Check if height value represents valid ground."""
    return THRESHOLDS.GROUND_MIN <= height <= THRESHOLDS.GROUND_MAX


def is_valid_platform(height: int) -> bool:
    """Check if height value represents valid platform."""
    return THRESHOLDS.PLATFORM_MIN <= height <= THRESHOLDS.PLATFORM_MAX


def is_valid_goal_position(height: int) -> bool:
    """Check if height value is valid for goal/spawn placement."""
    return is_valid_ground(height) or is_valid_platform(height)


def is_obstacle(height: int) -> bool:
    """Check if height value represents an obstacle (wall or pit)."""
    return height > THRESHOLDS.WALL_THRESHOLD or height < THRESHOLDS.PIT_THRESHOLD


def is_pit(height: int) -> bool:
    """Check if height value represents a pit."""
    return height < THRESHOLDS.PIT_THRESHOLD


def is_wall(height: int) -> bool:
    """Check if height value represents a wall."""
    return height > THRESHOLDS.WALL_THRESHOLD


def cell_to_pixels(cell_idx: int) -> Tuple[int, int]:
    """Convert cell index to pixel range.

    Args:
        cell_idx: Cell index in maze grid.

    Returns:
        Tuple of (start_pixel, end_pixel).
    """
    start = cell_idx * CELL_PIXELS
    end = (cell_idx + 1) * CELL_PIXELS
    return start, end
