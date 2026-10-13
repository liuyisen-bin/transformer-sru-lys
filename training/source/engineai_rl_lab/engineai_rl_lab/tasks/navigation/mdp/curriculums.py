# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Curriculum functions for navigation tasks.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def disable_backward_penalty_after_steps(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str = "backward_movement_penalty", 
    num_steps: int = 1000
) -> torch.Tensor:
    """Curriculum that disables the backward movement penalty after a certain number of steps.
    
    This helps with early training by preventing backward movement, but removes the constraint
    later to allow more natural movement patterns.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the backward movement penalty term.
        num_steps: The number of steps after which the penalty should be disabled.
        
    Returns:
        Current step counter as float for logging purposes.
    """
    if env.common_step_counter > num_steps:
        # Check if the term exists and has a non-zero weight
        if hasattr(env.reward_manager, 'get_term_cfg'):
            try:
                term_cfg = env.reward_manager.get_term_cfg(term_name)
                if term_cfg.weight != 0.0:
                    # Disable the penalty by setting weight to 0
                    term_cfg.weight = 0.0
                    env.reward_manager.set_term_cfg(term_name, term_cfg)
                    print(f"Disabled backward movement penalty at step {env.common_step_counter}")
            except KeyError:
                # Term doesn't exist, which is fine
                pass
    
    return torch.tensor(float(env.common_step_counter))
