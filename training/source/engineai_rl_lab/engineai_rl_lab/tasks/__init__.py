# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Isaac Lab navigation tasks.

On import, this module:
1. Applies monkey-patches to Isaac Lab terrain system for height field storage
2. Registers maze terrain types
3. Registers navigation task environments
"""

from .. import __version__

# Patch Isaac Lab terrains before any terrain generation. Import patches from
# the module file, not terrains/__init__.py, so terrain imports are not triggered.
import importlib.util
import os as _os
_patches_path = _os.path.join(_os.path.dirname(__file__), "terrains", "patches.py")
_spec = importlib.util.spec_from_file_location("patches", _patches_path)
_patches_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patches_module)
_patches_module.apply_terrain_patches()
del _patches_path, _spec, _patches_module

from .terrains import (
    HfMazeTerrainCfg,
    MAZE_TERRAIN_CFG,
)

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", "terrains", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
