"""Keep exact navigation inputs in memory; write after motor publishing stops."""
import json
from pathlib import Path
import numpy as np


class NavigationTrace:
    def __init__(self):
        self.rows = []
        self.meta = []

    def append(self, nav, h_in, c_in, metadata):
        self.rows.append((nav.last_obs[0].copy(), h_in.copy(), c_in.copy(),
                          nav.h.copy(), nav.c.copy(), nav.previous.copy(), nav.command.copy()))
        self.meta.append(metadata)

    def save(self, log_path):
        if not self.rows:
            return None
        path = Path(log_path).with_suffix('.navigation.npz')
        names = ('obs', 'h_in', 'c_in', 'h_out', 'c_out', 'raw_action', 'command')
        arrays = {name: np.stack([r[i] for r in self.rows]) for i, name in enumerate(names)}
        arrays['metadata_json'] = np.array([json.dumps(m) for m in self.meta])
        np.savez_compressed(path, **arrays)
        return path
