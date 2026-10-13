#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace transformer_sru_lidar_adapter {

inline constexpr std::size_t kRows = 16;
inline constexpr std::size_t kCols = 41;
inline constexpr std::size_t kPointCount = kRows * kCols;
inline constexpr std::size_t kValueCount = 3 * kPointCount;
inline constexpr double kMaxDistance = 5.0;

// p_torso = torso_from_lidar_rotation * p_lidar + translation.
// These are physical installation extrinsics, not the training sensor mount.
struct Calibration {
  Eigen::Matrix3d torso_from_lidar_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d torso_from_lidar_translation = Eigen::Vector3d::Zero();
};

struct Projection {
  // Rows: elevation -45..0 deg; columns: azimuth -60..60 deg; then XYZ.
  // Empty cells are zero. Nonempty XYZ is relative to the virtual sensor,
  // expressed in its yaw-only frame and divided by kMaxDistance.
  std::array<float, kValueCount> xyz{};
  std::size_t input_points = 0;
  // Invalid, zero, out-of-range or out-of-FOV returns; excludes valid returns
  // discarded because a nearer return already occupies the same cell.
  std::size_t rejected_points = 0;
  std::size_t occupied_cells = 0;
};

// Horizontal floor is inferred from the dominant lower plane in the raw cloud.
// Height is measured below the training virtual origin, not below the torso.
struct FloorEstimate {
  bool valid = false;
  double height_m = 0.0;
  double rms_m = 0.0;
  std::size_t inlier_points = 0;
};

struct FloorCompletion {
  std::array<float, kValueCount> xyz{};
  std::size_t filled_cells = 0;
};

Eigen::Matrix3d virtual_mount_rotation();
Eigen::Vector3d virtual_mount_translation();
std::array<Eigen::Vector3d, kPointCount> virtual_rays();

// Bins points in the *tilted virtual sensor frame* before removing sensor yaw.
// Nearest return in each 3-degree cell supplies the range of its center ray.
// No accumulation, interpolation, ground filling or IMU timing is performed.
// Throws std::invalid_argument for invalid calibration/torso orientation.
Projection project(const std::vector<Eigen::Vector3d>& raw,
                   const Calibration& calibration,
                   const Eigen::Quaterniond& world_from_torso);

FloorEstimate estimate_flat_floor(const std::vector<Eigen::Vector3d>& raw,
                                 const Calibration& calibration,
                                 const Eigen::Quaterniond& world_from_torso);

// Preserve every observed XYZ bit-for-bit. Fill only zero cells whose actual
// tilted training ray intersects z=-height in the sensor yaw frame within 5 m.
FloorCompletion complete_flat_floor(const std::array<float, kValueCount>& observed,
                                    const Eigen::Quaterniond& world_from_torso,
                                    double floor_height_m);

}  // namespace transformer_sru_lidar_adapter
