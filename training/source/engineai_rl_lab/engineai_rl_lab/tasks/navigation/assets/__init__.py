# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom robot configurations and assets for navigation tasks."""

import os

ISAACLAB_NAV_TASKS_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
"""Path to the navigation tasks assets data directory."""

from .pm01 import *

__all__ = [
    "ISAACLAB_NAV_TASKS_ASSETS_DIR",
    "PM01_CYLINDER_CFG",
    "PM01_ACTIVE_JOINT_NAMES",
    "PM01_ACTION_SCALE",
]
