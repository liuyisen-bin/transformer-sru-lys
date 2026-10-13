"""Production C++ LiDAR projection, shared by the ROS adapter and local simulation."""
import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get('SRU_LIDAR_LIBRARY',
    str(ROOT/'ros/build_standalone/liblidar_projection_c.so')))


class LidarProjection:
    def __init__(self, library=LIBRARY):
        if not Path(library).is_file():
            raise FileNotFoundError(f'LiDAR projection library not found: {library}. '
                                    'Build ros with -DSRU_STANDALONE=ON or set SRU_LIDAR_LIBRARY.')
        self.lib = ctypes.CDLL(str(library))
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        floats = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.lib.lidar_value_count.restype = ctypes.c_int
        if self.lib.lidar_value_count() != 1968:
            raise ValueError('Compiled LiDAR contract must be 16x41x3 for 24999')
        self.lib.project_cloud.argtypes = [doubles, ctypes.c_int, doubles, doubles, doubles, floats]
        self.lib.project_cloud.restype = ctypes.c_int
        self.lib.complete_floor.argtypes = [floats, doubles, ctypes.c_double, floats]
        self.lib.complete_floor.restype = ctypes.c_int
        self.lib.estimate_floor.argtypes = [doubles, ctypes.c_int, doubles, doubles, doubles, doubles]
        self.lib.estimate_floor.restype = ctypes.c_int
        self.lib.virtual_sensor_geometry.argtypes = [doubles, doubles, doubles]
        self.lib.virtual_sensor_geometry.restype = ctypes.c_double
        self.rays = np.empty((656, 3), np.float64)
        self.mount = np.empty((3, 3), np.float64)
        self.origin = np.empty(3, np.float64)
        self.max_distance = self.lib.virtual_sensor_geometry(self.rays, self.mount, self.origin)

    def project(self, raw_xyz, physical_rotation, physical_translation, torso_q_wxyz):
        arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in
                  (raw_xyz, physical_rotation, physical_translation, torso_q_wxyz)]
        points, rotation, translation, quat = arrays
        if points.ndim != 2 or points.shape[1] != 3 or rotation.shape != (3, 3) or translation.shape != (3,) or quat.shape != (4,):
            raise ValueError('Invalid projection array shape')
        out = np.zeros(1968, np.float32)
        count = self.lib.project_cloud(points, len(points), rotation, translation, quat, out)
        if count < 0:
            raise ValueError('Invalid projection calibration or orientation')
        return out, count

    def from_virtual_ranges(self, distances, torso_q_wxyz):
        distances = np.asarray(distances, dtype=np.float64)
        if distances.shape != (656,):
            raise ValueError('Expected one range per training ray')
        valid = np.isfinite(distances) & (distances > 1e-6) & (distances <= self.max_distance)
        points = self.rays[valid]*distances[valid, None]
        return self.project(points, self.mount, self.origin, torso_q_wxyz)

    def complete_floor(self, observed_xyz, torso_q_wxyz, height_m):
        observed = np.ascontiguousarray(observed_xyz, dtype=np.float32)
        quat = np.ascontiguousarray(torso_q_wxyz, dtype=np.float64)
        if observed.shape != (1968,) or quat.shape != (4,):
            raise ValueError('Invalid floor completion array shape')
        out = np.empty(1968, np.float32)
        count = self.lib.complete_floor(observed, quat, height_m, out)
        if count < 0:
            raise ValueError('Invalid floor height, observed values or torso orientation')
        return out, count

    def estimate_floor(self, raw_xyz, physical_rotation, physical_translation, torso_q_wxyz):
        points, rotation, translation, quat = [np.ascontiguousarray(x, dtype=np.float64) for x in
                                              (raw_xyz, physical_rotation, physical_translation, torso_q_wxyz)]
        if points.ndim != 2 or points.shape[1] != 3 or rotation.shape != (3,3) or translation.shape != (3,) or quat.shape != (4,):
            raise ValueError('Invalid floor estimation array shape')
        out = np.zeros(2, np.float64)
        count = self.lib.estimate_floor(points, len(points), rotation, translation, quat, out)
        if count < 0:
            raise ValueError('Invalid floor estimation calibration or orientation')
        return dict(valid=count > 0, height_m=float(out[0]), rms_m=float(out[1]), inlier_points=count)
