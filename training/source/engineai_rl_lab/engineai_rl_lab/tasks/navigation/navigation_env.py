# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom navigation environment with observation delay support."""
from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from engineai_rl_lab.tasks.navigation.mdp.delay_manager import ObservationDelayManager

if TYPE_CHECKING:
    from engineai_rl_lab.tasks.navigation.config.g1.navigation_env_cfg import G1NavigationEnvCfg


class NavigationEnv(ManagerBasedRLEnv):
    """Navigation environment with observation delay manager.

    This environment extends the base ManagerBasedRLEnv to add support for
    simulating sensor delays. The delay_manager is created before managers
    are loaded, ensuring observations can access it during initialization.
    """

    cfg: G1NavigationEnvCfg

    def load_managers(self):
        """Load managers with delay manager initialization.

        The delay_manager is created before other managers are loaded,
        ensuring observations can access it during initialization.
        """
        # Create the delay manager before loading other managers
        # At this point, self.num_envs and self.device are available via scene
        self.delay_manager = ObservationDelayManager(
            cfg=self.cfg.delay_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Now load all other managers
        super().load_managers()
