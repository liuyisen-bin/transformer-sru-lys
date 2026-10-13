#include "transformer_sru_lidar_adapter/grid.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace adapter = transformer_sru_lidar_adapter;
namespace {
constexpr double kPi = 3.14159265358979323846;
std::size_t checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void near(const Eigen::Vector3d& actual, const Eigen::Vector3d& expected,
          const std::string& message, double tolerance = 8e-8) {
  require((actual - expected).cwiseAbs().maxCoeff() < tolerance, message);
}

Eigen::Vector3d ray(double elevation_deg, double azimuth_deg, double range) {
  const double elevation = elevation_deg * kPi / 180.0;
  const double azimuth = azimuth_deg * kPi / 180.0;
  return range * Eigen::Vector3d(std::cos(elevation) * std::cos(azimuth),
                                std::cos(elevation) * std::sin(azimuth),
                                std::sin(elevation));
}

Eigen::Quaterniond rpy(double roll, double pitch, double yaw) {
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
}

Eigen::Vector3d cell(const adapter::Projection& projection, std::size_t row, std::size_t col) {
  const std::size_t offset = 3 * (row * adapter::kCols + col);
  return {projection.xyz[offset], projection.xyz[offset + 1], projection.xyz[offset + 2]};
}

adapter::Calibration virtual_calibration() {
  return {adapter::virtual_mount_rotation(), adapter::virtual_mount_translation()};
}

Eigen::Vector3d physical_point(const Eigen::Vector3d& virtual_point,
                               const adapter::Calibration& calibration) {
  return calibration.torso_from_lidar_rotation.transpose() *
      (adapter::virtual_mount_rotation() * virtual_point + adapter::virtual_mount_translation() -
       calibration.torso_from_lidar_translation);
}

// Same output convention as the already validated MuJoCo lidar_at_pose: cast
// in sensor axes, then world endpoints minus origin, then remove sensor yaw.
Eigen::Matrix3d expected_output_rotation(const Eigen::Quaterniond& torso) {
  const Eigen::Matrix3d sensor = torso.normalized().toRotationMatrix() *
      adapter::virtual_mount_rotation();
  const double yaw = std::atan2(sensor(1, 0), sensor(0, 0));
  Eigen::Matrix3d yaw_rotation;
  yaw_rotation << std::cos(yaw), -std::sin(yaw), 0,
                  std::sin(yaw), std::cos(yaw), 0,
                  0, 0, 1;
  return yaw_rotation.transpose() * sensor;
}

void full_grid_and_extrinsics() {
  require(adapter::kRows == 16 && adapter::kCols == 41 && adapter::kValueCount == 1968,
          "24999 LiDAR shape must be exactly 16 x 41 x XYZ");
  const adapter::Calibration calibration{
      rpy(kPi, 0.0928, -0.071).toRotationMatrix(), {0.04, -0.03, 0.31}};
  std::vector<Eigen::Vector3d> virtual_points;
  std::vector<Eigen::Vector3d> raw;
  for (std::size_t row = 0; row < 16; ++row) {
    for (std::size_t col = 0; col < 41; ++col) {
      const double range = 1.0 + 0.005 * (41 * row + col);
      virtual_points.push_back(ray(-45.0 + 3 * row, -60.0 + 3 * col, range));
      raw.push_back(physical_point(virtual_points.back(), calibration));
    }
  }
  for (const auto& angles : std::vector<Eigen::Vector3d>{
           {0, 0, 0}, {0.18, -0.23, 0}, {-0.25, 0.12, 1.5707963267948966},
           {0.18, -0.23, -2.4}, {0.18, -0.23, 3.1}}) {
    const auto torso = rpy(angles.x(), angles.y(), angles.z());
    const auto projection = adapter::project(raw, calibration, torso);
    require(projection.input_points == 656 && projection.occupied_cells == 656 &&
                projection.rejected_points == 0, "all 656 center rays must retain unique cells");
    const Eigen::Matrix3d output_rotation = expected_output_rotation(torso);
    for (std::size_t index = 0; index < 656; ++index) {
      const auto actual = cell(projection, index / 41, index % 41);
      near(actual, output_rotation * virtual_points[index] / 5.0,
           "wrong row-major XYZ, extrinsics, mount, yaw or normalization");
      require(std::abs(actual.norm() * 5.0 - (1.0 + 0.005 * index)) < 3e-7,
              "per-cell unique ranges must preserve monotonic slot order");
    }
  }
  const auto base = adapter::project(raw, calibration, rpy(0.18, -0.23, 0));
  const auto yaw_changed = adapter::project(raw, calibration, rpy(0.18, -0.23, 2.8));
  for (std::size_t index = 0; index < 656; ++index) {
    near(cell(base, index / 41, index % 41), cell(yaw_changed, index / 41, index % 41),
         "global heading must disappear from yaw-aligned XYZ");
  }
}

void nearest_return_and_center_ray() {
  const auto calibration = virtual_calibration();
  // Cancel the virtual mounting pitch so output axes coincide with virtual axes.
  const Eigen::Quaterniond torso(adapter::virtual_mount_rotation().transpose());
  std::vector<Eigen::Vector3d> raw{
      ray(-15, 0, 4), ray(-14.1, 1.1, 1.7), ray(-15.9, -1.1, 2.1)};
  const auto projection = adapter::project(raw, calibration, torso);
  require(projection.occupied_cells == 1 && projection.input_points == 3 &&
              projection.rejected_points == 0, "three valid returns must compete in one bin");
  near(cell(projection, 10, 20), ray(-15, 0, 1.7) / 5.0,
       "nearest range must reconstruct center ray, not preserve off-center endpoint");
  std::reverse(raw.begin(), raw.end());
  require(adapter::project(raw, calibration, torso).xyz == projection.xyz,
          "nearest selection must be independent of point order");
  for (std::size_t index = 0; index < 656; ++index) {
    if (index != 10 * 41 + 20) {
      require(cell(projection, index / 41, index % 41).isZero(0),
              "unobserved cell must stay zero without interpolation");
    }
  }
  const auto empty = adapter::project({}, calibration, torso);
  require(empty.occupied_cells == 0 && empty.input_points == 0 &&
              std::all_of(empty.xyz.begin(), empty.xyz.end(), [](float x) { return x == 0; }),
          "each frame must clear empty bins, with no implicit temporal accumulation");
}

void invalid_returns_and_range() {
  auto calibration = virtual_calibration();
  // A zero raw return would become a valid forward virtual point if transformed
  // before rejection. This checks the previously audited phantom-return bug.
  calibration.torso_from_lidar_translation += adapter::virtual_mount_rotation() * ray(-15, 0, 0.3);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto invalid = adapter::project({{0, 0, 0}, {1e-8, 0, 0}, {nan, 1, 0}, {inf, 0, 0},
                                        {1e308, 1e308, 1e308},
                                        physical_point(Eigen::Vector3d::Zero(), calibration)},
                                       calibration, Eigen::Quaterniond::Identity());
  require(invalid.occupied_cells == 0 && invalid.rejected_points == 6,
          "raw zero/nonfinite and virtual zero/overflow returns must be rejected");
  calibration = virtual_calibration();
  const auto range = adapter::project({ray(-15, -3, 5.0), ray(-15, 0, 5.001), ray(-15, 3, 4.999)},
                                     calibration, Eigen::Quaterniond::Identity());
  require(range.occupied_cells == 2 && range.rejected_points == 1,
          "5 m range limit must include the endpoint and reject farther objects");
  require(std::abs(cell(range, 10, 19).norm() - 1.0) < 8e-8,
          "5 m return must normalize to unit range");
}

void exact_range_float32_cloud() {
  // Exercise the real PointCloud2 storage boundary, rather than only double
  // synthetic points. Rounding a 5 m endpoint to float32 can increase its
  // reconstructed virtual range even though the source endpoint was valid.
  const std::vector<adapter::Calibration> calibrations{
      virtual_calibration(),
      {rpy(kPi, 0.0928, -0.071).toRotationMatrix(), {0.04, -0.03, 0.31}},
      {rpy(-0.2, 0.12, 0.3).toRotationMatrix(), {-0.08, 0.05, 0.43}}};
  for (const auto& calibration : calibrations) {
    std::vector<Eigen::Vector3d> raw;
    std::vector<Eigen::Vector3d> beyond_range;
    for (std::size_t row = 0; row < 16; ++row) {
      for (std::size_t col = 0; col < 41; ++col) {
        raw.push_back(physical_point(ray(-45.0 + 3 * row, -60.0 + 3 * col, 5.0), calibration)
                          .cast<float>().cast<double>());
        beyond_range.push_back(
            physical_point(ray(-45.0 + 3 * row, -60.0 + 3 * col, 5.0001), calibration)
                .cast<float>().cast<double>());
      }
    }
    const auto torso = rpy(0.18, -0.23, 1.2);
    const auto projection = adapter::project(raw, calibration, torso);
    require(projection.occupied_cells == 656 && projection.rejected_points == 0,
            "float32 exact 5 m cloud must retain all 656 cells under physical extrinsics");
    const Eigen::Matrix3d output_rotation = expected_output_rotation(torso);
    for (std::size_t row = 0; row < 16; ++row) {
      for (std::size_t col = 0; col < 41; ++col) {
        const auto actual = cell(projection, row, col);
        near(actual, output_rotation * ray(-45.0 + 3 * row, -60.0 + 3 * col, 1.0),
             "float32 exact range must preserve normalized center-ray output", 1.2e-7);
        require(actual.norm() <= 1.0 + 8e-8,
                "numeric range tolerance must clamp accepted returns to 5 m");
      }
    }
    const auto rejected = adapter::project(beyond_range, calibration, torso);
    require(rejected.occupied_cells == 0 && rejected.rejected_points == 656,
            "float32 5.0001 m cloud must remain outside the permitted numeric tolerance");
  }
}

void half_bin_edges_and_tilt_preserves_slots() {
  const auto calibration = virtual_calibration();
  const auto torso = rpy(0.18, -0.23, 1.2);
  const auto edges = adapter::project({ray(-15, -61.499, 2), ray(-15, 61.499, 2),
                                      ray(-46.499, 0, 2), ray(1.499, 0, 2),
                                      ray(-15, -61.501, 2), ray(-15, 61.501, 2),
                                      ray(-46.501, 0, 2), ray(1.501, 0, 2)},
                                     calibration, torso);
  require(edges.occupied_cells == 4 && edges.rejected_points == 4,
          "half-bin FOV edges must accept near edge returns and reject beyond-edge ones");
  const Eigen::Matrix3d output_rotation = expected_output_rotation(torso);
  near(cell(edges, 10, 0), output_rotation * ray(-15, -60, 2) / 5.0, "left edge cell");
  near(cell(edges, 10, 40), output_rotation * ray(-15, 60, 2) / 5.0, "right edge cell");
  near(cell(edges, 0, 20), output_rotation * ray(-45, 0, 2) / 5.0, "lower edge cell");
  near(cell(edges, 15, 20), output_rotation * ray(0, 0, 2) / 5.0, "upper edge cell");

  const auto tilted = adapter::project({ray(-15, 0, 2)}, calibration, torso);
  const auto output = cell(tilted, 10, 20);
  require(tilted.occupied_cells == 1 && output.norm() > 0.39,
          "torso tilt must not relabel the virtual sensor grid slot");
  const double old_elevation = std::atan2(output.z(), std::hypot(output.x(), output.y())) * 180 / kPi;
  require(std::llround((old_elevation + 45) / 3) != 10,
          "test pose must expose the old yaw-frame-before-binning error");
}

template <typename Function>
void throws_invalid(Function function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    require(true, message);
    return;
  }
  require(false, message);
}

void invalid_calibration_and_quaternion() {
  const auto test_calibration = [](const adapter::Calibration& calibration) {
    adapter::project({}, calibration, Eigen::Quaterniond::Identity());
  };
  auto calibration = virtual_calibration();
  calibration.torso_from_lidar_rotation(0, 0) += 0.1;
  throws_invalid([&] { test_calibration(calibration); }, "nonorthogonal extrinsics must fail");
  calibration = virtual_calibration();
  calibration.torso_from_lidar_rotation.col(0) *= -1;
  throws_invalid([&] { test_calibration(calibration); }, "reflection extrinsics must fail");
  calibration = virtual_calibration();
  calibration.torso_from_lidar_translation.x() = std::numeric_limits<double>::quiet_NaN();
  throws_invalid([&] { test_calibration(calibration); }, "nonfinite extrinsics must fail");
  calibration = virtual_calibration();
  throws_invalid([&] { adapter::project({}, calibration, Eigen::Quaterniond(0, 0, 0, 0)); },
                 "zero quaternion must fail");
  throws_invalid([&] { adapter::project({}, calibration, Eigen::Quaterniond(
      std::numeric_limits<double>::quiet_NaN(), 0, 0, 0)); }, "nonfinite quaternion must fail");
  auto unit = rpy(0.13, -0.1, 1.1);
  auto scaled = unit;
  scaled.coeffs() *= 2.3;
  const std::vector<Eigen::Vector3d> point{ray(-15, 0, 2)};
  const auto a = adapter::project(point, calibration, unit);
  const auto b = adapter::project(point, calibration, scaled);
  near(cell(a, 10, 20), cell(b, 10, 20), "finite nonunit quaternion must normalize");
}
void flat_floor_geometry_and_estimation() {
  const auto rays = adapter::virtual_rays();
  const adapter::Calibration calibration{rpy(3.1, .12, -.3).toRotationMatrix(), {.06, -.02, .35}};
  for (const auto& angles : std::vector<Eigen::Vector3d>{{0,0,0},{.2,-.17,2.1},{-.19,.14,-2.5}}) {
    const auto torso = rpy(angles.x(),angles.y(),angles.z());
    const auto output = expected_output_rotation(torso);
    for (double height : {1.2, 1.45}) {
      std::array<float, adapter::kValueCount> observed{};
      // A real nearer obstacle must survive even where a floor ray is possible.
      observed[60]=.11f; observed[61]=.02f; observed[62]=-.07f;
      const auto result=adapter::complete_flat_floor(observed,torso,height);
      std::size_t expected_count=0;
      for (std::size_t i=0; i<adapter::kPointCount; ++i) {
        const Eigen::Vector3d actual(result.xyz[3*i],result.xyz[3*i+1],result.xyz[3*i+2]);
        if (i==20) {
          require(result.xyz[60]==observed[60] && result.xyz[61]==observed[61] && result.xyz[62]==observed[62],"observed obstacle overwritten");
          continue;
        }
        const Eigen::Vector3d direction=output*rays[i];
        const double distance=direction.z()<0 ? height/-direction.z() : 1e10;
        if (distance<=5.) {
          near(actual,direction*distance/5.,"floor endpoint differs from training ray");
          require(std::abs(actual.z()+height/5.)<3e-8,"floor not horizontal");
          ++expected_count;
        } else require(actual.isZero(),"upward or distant ray must remain zero");
      }
      require(result.filled_cells==expected_count,"floor provenance count incorrect");
      std::vector<Eigen::Vector3d> raw;
      for (int ix=-20;ix<=20;++ix) for(int iy=-20;iy<=20;++iy) {
        const Eigen::Vector3d world(.12*ix,.12*iy,-height + .001*std::sin(ix+iy));
        const Eigen::Vector3d torso_relative=torso.toRotationMatrix().transpose()*world;
        raw.push_back(calibration.torso_from_lidar_rotation.transpose()*(torso_relative+adapter::virtual_mount_translation()-calibration.torso_from_lidar_translation));
      }
      for(int i=0;i<300;++i) raw.emplace_back(.3+i*.001,.1,-.2); // non-floor clutter
      raw.emplace_back(0,0,0);
      const auto floor=adapter::estimate_flat_floor(raw,calibration,torso);
      require(floor.valid && floor.inlier_points>1000,"floor unavailable among dense ground returns");
      require(std::abs(floor.height_m-height)<.0002 && floor.rms_m<.002,"floor height or residual wrong");
    }
  }
  require(!adapter::estimate_flat_floor({},calibration,Eigen::Quaterniond::Identity()).valid,"empty cloud inferred ground");
  throws_invalid([&]{adapter::complete_flat_floor({},Eigen::Quaterniond::Identity(),0);},"nonpositive floor height accepted");
  std::array<float,adapter::kValueCount> invalid{}; invalid[0]=std::numeric_limits<float>::quiet_NaN();
  throws_invalid([&]{adapter::complete_flat_floor(invalid,Eigen::Quaterniond::Identity(),1.2);},"nonfinite observed point accepted");
}
}  // namespace

int main() {
  try {
    full_grid_and_extrinsics();
    nearest_return_and_center_ray();
    invalid_returns_and_range();
    exact_range_float32_cloud();
    half_bin_edges_and_tilt_preserves_slots();
    invalid_calibration_and_quaternion();
    flat_floor_geometry_and_estimation();
    std::cout << "PASS: LiDAR geometry contract (" << checks << " checks)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
