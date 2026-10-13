"""Explicit Unitree Passive -> FixStand -> zero-velocity policy lifecycle.

The pose, gains and interpolation time are copied from the official G1 FSM.
Transitions follow explicit operator requests; measured error is diagnostic only.
"""
import json
from pathlib import Path
import numpy as np
from core import vector

STARTUP = json.loads(Path(__file__).with_name('startup_contract.json').read_text())


class Startup:
    def __init__(self):
        self.q = vector(STARTUP['fixstand_q_sdk'], 29, 'fixstand_q')
        self.kp = vector(STARTUP['fixstand_kp_sdk'], 29, 'fixstand_kp')
        self.kd = vector(STARTUP['fixstand_kd_sdk'], 29, 'fixstand_kd')
        self.passive_kd = vector(STARTUP['passive_kd_sdk'], 29, 'passive_kd')
        self.passive()

    def passive(self):
        self.phase = 'PASSIVE'
        self.started = None
        self.q_error = None

    def prepare(self, q, now):
        if self.phase != 'PASSIVE':
            raise ValueError('prepare_requires_PASSIVE')
        self.start_q = vector(q, 29, 'prepare_q').copy()
        self.started = now
        self.phase = 'FIX_STAND'

    def observe(self, state, now):
        if self.phase not in ('FIX_STAND', 'FIX_STAND_READY'):
            return
        self.q_error = float(np.max(abs(state['q'] - self.q)))
        # READY reports interpolation completion, not measured posture approval.
        if now - self.started >= STARTUP['fixstand_duration_s']:
            self.phase = 'FIX_STAND_READY'

    def start_policy(self, state, now):
        if self.phase not in ('FIX_STAND', 'FIX_STAND_READY'):
            raise ValueError('start_policy_requires_FIX_STAND')
        self.phase = 'ZERO_VELOCITY'
        self.started = now

    def position_command(self, state, now):
        if self.phase == 'PASSIVE':
            return state['q'].copy(), np.zeros(29), self.passive_kd.copy()
        if self.phase in ('FIX_STAND', 'FIX_STAND_READY'):
            ratio = np.clip((now - self.started) / STARTUP['fixstand_duration_s'], 0., 1.)
            return (1. - ratio) * self.start_q + ratio * self.q, self.kp.copy(), self.kd.copy()
        return None


def remote_transition(previous, buttons):
    """Unitree SDK bit layout; require a new up/X edge, not a held chord."""
    if buttons & (1 << 5) and buttons & (1 << 9):
        return 'stop'
    if previous is None:
        return None
    pressed = buttons & ~previous
    if buttons & (1 << 5) and pressed & (1 << 12):
        return 'prepare'
    if buttons & 1 and pressed & (1 << 10):
        return 'start_policy'
    return None


class ButtonDoubleClick:
    """Two rising edges within 0.5s; initial held buttons are ignored."""
    def __init__(self, bit, latch=False):
        self.mask = 1 << bit
        self.latch = latch
        self.reset()

    def reset(self):
        self.previous = None
        self.first_press_at = None
        self.latched = False

    def update(self, buttons, now):
        pressed = bool(buttons & self.mask)
        edge = self.previous is False and pressed
        self.previous = pressed
        if self.latched:
            return True
        if edge:
            if self.first_press_at is not None and 0 <= now - self.first_press_at <= .5:
                self.first_press_at = None
                self.latched = self.latch
                return True
            else:
                self.first_press_at = now
        return self.latched


class SelectDoubleClick(ButtonDoubleClick):
    """Select/Back is SDK bit 3; stop remains latched until process exit."""
    def __init__(self):
        super().__init__(3, latch=True)
