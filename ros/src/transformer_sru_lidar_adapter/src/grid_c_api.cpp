// Numerical API for deployment simulation and source parity tests. No ROS I/O.
#include "transformer_sru_lidar_adapter/grid.hpp"
#include <algorithm>

extern "C" int project_cloud(const double* xyz, int count, const double* rotation,
                             const double* translation, const double* q, float* out) {
  if (!rotation || !translation || !q || !out || count < 0 || (count && !xyz)) return -1;
  try {
    using namespace transformer_sru_lidar_adapter;
    Calibration calibration;
    for (int i = 0; i < 3; ++i) {
      calibration.torso_from_lidar_translation[i] = translation[i];
      for (int j = 0; j < 3; ++j) calibration.torso_from_lidar_rotation(i,j) = rotation[3*i+j];
    }
    std::vector<Eigen::Vector3d> points;
    points.reserve(count);
    for (int i = 0; i < count; ++i) points.emplace_back(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
    const auto result = project(points, calibration, Eigen::Quaterniond(q[0],q[1],q[2],q[3]));
    std::copy(result.xyz.begin(), result.xyz.end(), out);
    return static_cast<int>(result.occupied_cells);
  } catch (...) { return -1; }
}

extern "C" double virtual_sensor_geometry(double* rays, double* rotation, double* translation) {
  using namespace transformer_sru_lidar_adapter;
  const auto directions = virtual_rays();
  const auto mount = virtual_mount_rotation();
  const auto origin = virtual_mount_translation();
  for (std::size_t i = 0; i < kPointCount; ++i)
    for (int j = 0; j < 3; ++j) rays[3*i+j] = directions[i][j];
  for (int i = 0; i < 3; ++i) {
    translation[i] = origin[i];
    for (int j = 0; j < 3; ++j) rotation[3*i+j] = mount(i,j);
  }
  return kMaxDistance;
}

extern "C" int lidar_value_count() {
  return static_cast<int>(transformer_sru_lidar_adapter::kValueCount);
}

extern "C" int complete_floor(const float* observed, const double* q, double height, float* out) {
  if (!observed || !q || !out) return -1;
  try {
    using namespace transformer_sru_lidar_adapter;
    std::array<float, kValueCount> xyz;
    std::copy(observed, observed + kValueCount, xyz.begin());
    const auto completed = complete_flat_floor(xyz, Eigen::Quaterniond(q[0],q[1],q[2],q[3]), height);
    std::copy(completed.xyz.begin(), completed.xyz.end(), out);
    return static_cast<int>(completed.filled_cells);
  } catch (...) { return -1; }
}

extern "C" int estimate_floor(const double* xyz, int count, const double* rotation,
                              const double* translation, const double* q, double* out) {
  if (!xyz || count < 0 || !rotation || !translation || !q || !out) return -1;
  try {
    using namespace transformer_sru_lidar_adapter;
    Calibration calibration;
    for (int i = 0; i < 3; ++i) {
      calibration.torso_from_lidar_translation[i] = translation[i];
      for (int j = 0; j < 3; ++j) calibration.torso_from_lidar_rotation(i,j) = rotation[3*i+j];
    }
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < count; ++i) points.emplace_back(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
    const auto floor = estimate_flat_floor(points, calibration, Eigen::Quaterniond(q[0],q[1],q[2],q[3]));
    out[0] = floor.height_m; out[1] = floor.rms_m;
    return floor.valid ? static_cast<int>(floor.inlier_points) : 0;
  } catch (...) { return -1; }
}
