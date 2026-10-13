#include "transformer_sru_lidar_adapter/grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace transformer_sru_lidar_adapter {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegrees = 180.0 / kPi;
constexpr double kStepDegrees = 3.0;
constexpr double kMinimumElevation = -45.0;
constexpr double kMinimumAzimuth = -60.0;
constexpr double kZeroReturn = 1e-6;
// PointCloud2 XYZ commonly arrives as float32. Quantization plus rigid-frame
// transforms can put a true 5 m endpoint a few 1e-7 m outside the range.
// Admit only this numerical margin and clamp retained ranges back to 5 m.
constexpr double kRangeRoundoff = 1e-6;

Eigen::Vector3d center_ray(std::size_t row, std::size_t col) {
  const double elevation = (kMinimumElevation + kStepDegrees * row) / kDegrees;
  const double azimuth = (kMinimumAzimuth + kStepDegrees * col) / kDegrees;
  return {std::cos(elevation) * std::cos(azimuth),
          std::cos(elevation) * std::sin(azimuth), std::sin(elevation)};
}

Eigen::Matrix3d yaw_from_virtual(const Eigen::Quaterniond& torso) {
  const Eigen::Matrix3d world = torso.normalized().toRotationMatrix() * virtual_mount_rotation();
  const double yaw = std::atan2(world(1, 0), world(0, 0));
  return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * world;
}

double median(std::vector<double> values) {
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  const double upper = *middle;
  return values.size() % 2 ? upper : (upper + *std::max_element(values.begin(), middle)) / 2.0;
}

void validate(const Calibration& calibration,
              const Eigen::Quaterniond& world_from_torso) {
  const auto& rotation = calibration.torso_from_lidar_rotation;
  if (!rotation.allFinite() || !calibration.torso_from_lidar_translation.allFinite() ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
      std::abs(rotation.determinant() - 1.0) > 1e-6) {
    throw std::invalid_argument("LiDAR extrinsics must be a finite proper rigid transform");
  }
  const double norm = world_from_torso.norm();
  if (!world_from_torso.coeffs().allFinite() || !std::isfinite(norm) || norm <= 1e-12) {
    throw std::invalid_argument("Torso orientation must be a finite nonzero quaternion");
  }
}
}  // namespace

Eigen::Matrix3d virtual_mount_rotation() {
  return Eigen::AngleAxisd(0.04014257279586953, Eigen::Vector3d::UnitY()).toRotationMatrix();
}

Eigen::Vector3d virtual_mount_translation() {
  return {0.0002835, 0.00003, 0.40618};
}

std::array<Eigen::Vector3d, kPointCount> virtual_rays() {
  std::array<Eigen::Vector3d, kPointCount> rays;
  for (std::size_t i = 0; i < kPointCount; ++i) {
    rays[i] = center_ray(i / kCols, i % kCols);
  }
  return rays;
}

Projection project(const std::vector<Eigen::Vector3d>& raw,
                   const Calibration& calibration,
                   const Eigen::Quaterniond& world_from_torso) {
  validate(calibration, world_from_torso);
  Projection result;
  result.input_points = raw.size();
  std::array<double, kPointCount> ranges;
  ranges.fill(std::numeric_limits<double>::infinity());

  const Eigen::Matrix3d torso_from_virtual = virtual_mount_rotation();
  const Eigen::Matrix3d virtual_from_lidar =
      torso_from_virtual.transpose() * calibration.torso_from_lidar_rotation;
  const Eigen::Vector3d virtual_translation = torso_from_virtual.transpose() *
      (calibration.torso_from_lidar_translation - virtual_mount_translation());

  for (const auto& raw_point : raw) {
    // Livox zero placeholders must be rejected BEFORE translated extrinsics:
    // otherwise a zero becomes a phantom point at the sensor displacement.
    if (!raw_point.allFinite() || raw_point.norm() <= kZeroReturn) {
      ++result.rejected_points;
      continue;
    }
    const Eigen::Vector3d point = virtual_from_lidar * raw_point + virtual_translation;
    const double distance = point.norm();
    if (!point.allFinite() || !std::isfinite(distance) || distance <= kZeroReturn ||
        distance > kMaxDistance + kRangeRoundoff) {
      ++result.rejected_points;
      continue;
    }
    const double azimuth = std::atan2(point.y(), point.x()) * kDegrees;
    const double elevation = std::atan2(point.z(), std::hypot(point.x(), point.y())) * kDegrees;
    // Half-cell support at the field edges. llround resolves exact half-bin
    // ties away from zero in these normalized grid coordinates.
    const auto row = std::llround((elevation - kMinimumElevation) / kStepDegrees);
    const auto col = std::llround((azimuth - kMinimumAzimuth) / kStepDegrees);
    if (row < 0 || row >= static_cast<long long>(kRows) ||
        col < 0 || col >= static_cast<long long>(kCols)) {
      ++result.rejected_points;
      continue;
    }
    const auto index = static_cast<std::size_t>(row) * kCols + static_cast<std::size_t>(col);
    ranges[index] = std::min(ranges[index], std::min(distance, kMaxDistance));
  }

  const Eigen::Matrix3d output_rotation = yaw_from_virtual(world_from_torso);
  for (std::size_t index = 0; index < kPointCount; ++index) {
    if (!std::isfinite(ranges[index])) {
      continue;
    }
    ++result.occupied_cells;
    const Eigen::Vector3d normalized = output_rotation *
        center_ray(index / kCols, index % kCols) * (ranges[index] / kMaxDistance);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      result.xyz[3 * index + axis] = static_cast<float>(normalized[axis]);
    }
  }
  return result;
}

FloorEstimate estimate_flat_floor(const std::vector<Eigen::Vector3d>& raw,
                                 const Calibration& calibration,
                                 const Eigen::Quaterniond& torso) {
  validate(calibration, torso);
  constexpr double minimum_height = 0.5, maximum_height = 2.0, bin_size = 0.02;
  constexpr std::size_t bins = 76, minimum_inliers = 200;
  std::array<std::size_t, bins> histogram{};
  std::vector<double> heights;
  const auto world_from_torso = torso.normalized().toRotationMatrix();
  const auto world_from_raw = world_from_torso * calibration.torso_from_lidar_rotation;
  const Eigen::Vector3d shift = world_from_torso *
      (calibration.torso_from_lidar_translation - virtual_mount_translation());
  for (const auto& p : raw) {
    if (!p.allFinite() || p.norm() <= kZeroReturn) continue;
    const Eigen::Vector3d relative = world_from_raw * p + shift;
    if (!relative.allFinite()) continue;
    const double horizontal = std::hypot(relative.x(), relative.y());
    const double height = -relative.z();
    if (horizontal < 0.4 || horizontal > 4.0 || height < minimum_height || height > maximum_height) continue;
    heights.push_back(height);
    ++histogram[static_cast<std::size_t>((height - minimum_height) / bin_size)];
  }
  FloorEstimate result;
  if (heights.size() < minimum_inliers) return result;
  const auto peak = static_cast<std::size_t>(std::max_element(histogram.begin(), histogram.end()) - histogram.begin());
  double height = minimum_height + (peak + 0.5) * bin_size;
  std::vector<double> inliers;
  for (int iteration = 0; iteration < 3; ++iteration) {
    inliers.clear();
    for (const double value : heights) if (std::abs(value - height) <= 0.04) inliers.push_back(value);
    if (inliers.size() < minimum_inliers) return result;
    height = median(inliers);
  }
  double squared_error = 0.0;
  for (const double value : inliers) squared_error += (value - height) * (value - height);
  result.valid = true;
  result.height_m = height;
  result.rms_m = std::sqrt(squared_error / inliers.size());
  result.inlier_points = inliers.size();
  return result;
}

FloorCompletion complete_flat_floor(const std::array<float, kValueCount>& observed,
                                    const Eigen::Quaterniond& torso,
                                    double height) {
  validate(Calibration{}, torso);
  if (!std::isfinite(height) || height <= 0.0) throw std::invalid_argument("Floor height must be finite and positive");
  FloorCompletion result;
  result.xyz = observed;
  const auto output_rotation = yaw_from_virtual(torso);
  for (std::size_t i = 0; i < kPointCount; ++i) {
    const Eigen::Vector3f point(observed[3*i], observed[3*i+1], observed[3*i+2]);
    if (!point.allFinite()) throw std::invalid_argument("Observed XYZ must be finite");
    if (!point.isZero(0.0f)) continue;
    const Eigen::Vector3d direction = output_rotation * center_ray(i / kCols, i % kCols);
    if (direction.z() >= -1e-9) continue;
    const double distance = height / -direction.z();
    if (distance > kMaxDistance) continue;
    const Eigen::Vector3d endpoint = direction * (distance / kMaxDistance);
    for (int axis = 0; axis < 3; ++axis) result.xyz[3*i+axis] = static_cast<float>(endpoint[axis]);
    ++result.filled_cells;
  }
  return result;
}

}  // namespace transformer_sru_lidar_adapter
