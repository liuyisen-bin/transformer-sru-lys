// SPDX-License-Identifier: Apache-2.0
// Receive-only Livox adapter for the fixed Transformer-SRU 16 x 41 contract.
#include "transformer_sru_lidar_adapter/grid.hpp"
#include "transformer_sru_lidar_adapter/msg/lidar_tensor.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <unitree_hg/msg/imu_state.hpp>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace transformer_sru_lidar_adapter
{
namespace
{
using SteadyClock = std::chrono::steady_clock;
using PointCloud = sensor_msgs::msg::PointCloud2;
using Status = diagnostic_msgs::msg::DiagnosticStatus;
constexpr double kPi = 3.14159265358979323846;
constexpr double kImuHistorySeconds = 2.0;
constexpr std::size_t kMaxImuHistorySamples = 4096;

float read_float32(const std::uint8_t * data, bool big_endian)
{
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
    "PointCloud2 requires IEEE-754 float32");
  const std::uint32_t bits = big_endian ?
    (std::uint32_t(data[0]) << 24 | std::uint32_t(data[1]) << 16 |
    std::uint32_t(data[2]) << 8 | std::uint32_t(data[3])) :
    (std::uint32_t(data[3]) << 24 | std::uint32_t(data[2]) << 16 |
    std::uint32_t(data[1]) << 8 | std::uint32_t(data[0]));
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void write_float32_le(float value, std::uint8_t * data)
{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (unsigned int byte = 0; byte < 4; ++byte) {
    data[byte] = static_cast<std::uint8_t>((bits >> (8 * byte)) & 0xff);
  }
}

// Do not use unchecked PointCloud2 iterators on externally supplied layouts.
bool xyz_layout(const PointCloud & cloud, std::array<std::uint32_t, 3> & offsets,
  std::string & error)
{
  const std::uint64_t count = std::uint64_t(cloud.width) * cloud.height;
  if (count == 0 || count > std::numeric_limits<std::uint32_t>::max() ||
    cloud.point_step < 12 ||
    std::uint64_t(cloud.row_step) < std::uint64_t(cloud.width) * cloud.point_step ||
    std::uint64_t(cloud.data.size()) < std::uint64_t(cloud.row_step) * cloud.height)
  {
    error = "invalid_cloud_layout: dimensions, row_step, point_step or data size";
    return false;
  }
  std::array<bool, 3> seen{{false, false, false}};
  for (const auto & field : cloud.fields) {
    int axis = field.name == "x" ? 0 : field.name == "y" ? 1 : field.name == "z" ? 2 : -1;
    if (axis < 0) {continue;}
    if (seen[axis] || field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
      field.count != 1 || std::uint64_t(field.offset) + 4 > cloud.point_step)
    {
      error = "invalid_cloud_layout: XYZ must be unique scalar FLOAT32 fields within point_step";
      return false;
    }
    seen[axis] = true;
    offsets[axis] = field.offset;
  }
  if (!std::all_of(seen.begin(), seen.end(), [](bool present) {return present;})) {
    error = "invalid_cloud_layout: missing XYZ fields";
    return false;
  }
  for (std::size_t a = 0; a < 3; ++a) {
    for (std::size_t b = a + 1; b < 3; ++b) {
      if (std::uint64_t(offsets[a]) < std::uint64_t(offsets[b]) + 4 &&
        std::uint64_t(offsets[b]) < std::uint64_t(offsets[a]) + 4)
      {
        error = "invalid_cloud_layout: overlapping XYZ fields";
        return false;
      }
    }
  }
  return true;
}
}  // namespace

class LidarAdapterNode : public rclcpp::Node
{
public:
  LidarAdapterNode() : Node("transformer_sru_lidar_adapter")
  {
    const auto input_topic = fixed<std::string>("input_topic", "/livox/lidar");
    const auto imu_topic = fixed<std::string>("imu_topic", "/secondary_imu");
    const auto tensor_topic = fixed<std::string>("tensor_output_topic", "/transformer_sru/lidar");
    const auto observed_topic = fixed<std::string>("observed_tensor_topic", "/transformer_sru/lidar_observed");
    const auto cloud_topic = fixed<std::string>("cloud_output_topic", "/transformer_sru/lidar_xyz_norm");
    const auto status_topic = fixed<std::string>("status_topic", "/transformer_sru/lidar_status");
    expected_lidar_frame_ = fixed<std::string>("expected_lidar_frame", "livox_frame");
    output_frame_ = fixed<std::string>("output_frame", "transformer_sru_virtual_lidar_yaw");
    extrinsics_verified_ = fixed<bool>("extrinsics_verified", false);
    floor_completion_enabled_ = fixed<bool>("flat_floor_completion_enabled", false);
    const auto rpy = fixed<std::vector<double>>("lidar_to_torso_rpy", {kPi, 0.0927991563, 0.0});
    const auto xyz = fixed<std::vector<double>>("lidar_to_torso_xyz", {0.0002835, 0.00003, 0.40618});
    max_cloud_age_ = fixed<double>("max_cloud_age_s", 0.25);
    max_imu_age_ = fixed<double>("max_imu_age_s", 0.10);
    max_imu_cloud_offset_ = fixed<double>("max_imu_cloud_offset_s", 0.05);
    future_tolerance_ = fixed<double>("future_tolerance_s", 0.05);
    for (const auto & value : {input_topic, imu_topic, tensor_topic, observed_topic, cloud_topic,
        status_topic, expected_lidar_frame_, output_frame_})
    {
      if (value.empty()) {throw std::invalid_argument("topics and frame names must not be empty");}
    }
    if (rpy.size() != 3 || xyz.size() != 3 ||
      !std::all_of(rpy.begin(), rpy.end(), [](double v) {return std::isfinite(v);}) ||
      !std::all_of(xyz.begin(), xyz.end(), [](double v) {return std::isfinite(v);}))
    {
      throw std::invalid_argument("lidar_to_torso_rpy and lidar_to_torso_xyz must each contain three finite numbers");
    }
    if (!std::isfinite(max_cloud_age_) || max_cloud_age_ <= 0 ||
      !std::isfinite(max_imu_age_) || max_imu_age_ <= 0 ||
      !std::isfinite(max_imu_cloud_offset_) || max_imu_cloud_offset_ < 0 ||
      !std::isfinite(future_tolerance_) || future_tolerance_ < 0 ||
      max_imu_age_ > kImuHistorySeconds || max_imu_cloud_offset_ > kImuHistorySeconds)
    {
      throw std::invalid_argument("invalid age/offset limits; IMU age/offset must fit the 2 second history");
    }
    calibration_.torso_from_lidar_rotation =
      (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
    calibration_.torso_from_lidar_translation = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
    const auto & rotation = calibration_.torso_from_lidar_rotation;
    if (!rotation.allFinite() ||
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-9 ||
      std::abs(rotation.determinant() - 1.0) > 1e-9)
    {
      throw std::invalid_argument("invalid lidar-to-torso rotation");
    }
    tensor_publisher_ = create_publisher<msg::LidarTensor>(tensor_topic, rclcpp::SensorDataQoS());
    observed_publisher_ = create_publisher<msg::LidarTensor>(observed_topic, rclcpp::SensorDataQoS());
    cloud_publisher_ = create_publisher<PointCloud>(cloud_topic, rclcpp::SensorDataQoS());
    status_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(status_topic, 1);
    cloud_subscription_ = create_subscription<PointCloud>(input_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&LidarAdapterNode::cloud_callback, this, std::placeholders::_1));
    imu_subscription_ = create_subscription<unitree_hg::msg::IMUState>(imu_topic,
      rclcpp::SensorDataQoS(), std::bind(&LidarAdapterNode::imu_callback, this, std::placeholders::_1));
    status_timer_ = create_wall_timer(std::chrono::seconds(1),
      std::bind(&LidarAdapterNode::publish_status, this));
    RCLCPP_INFO(get_logger(),
      "Receive-only adapter: 16x41 XYZ / 5m; IMU time is receipt time (not acquisition sync). "
      "Calibration verified=%s; flat-floor completion=%s", extrinsics_verified_ ? "true" : "false",
      floor_completion_enabled_ ? "true (estimated points tagged separately)" : "false");
    if (!extrinsics_verified_) {
      RCLCPP_WARN(get_logger(), "Nominal legacy extrinsics are unverified. Publishing preview tensors with extrinsics_verified=false; motor control must remain disabled.");
    }
  }

private:
  template<typename T>
  T fixed(const std::string & name, const T & default_value)
  {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    return declare_parameter<T>(name, default_value, descriptor);
  }

  struct ImuSample
  {
    std::int64_t receipt_ns;
    SteadyClock::time_point steady_receipt;
    Eigen::Quaterniond world_from_torso;
  };

  void observe_clock(std::int64_t now_ns)
  {
    if (last_clock_ns_ != 0 && now_ns < last_clock_ns_) {
      imu_history_.clear();
      has_valid_imu_ = false;
      last_accepted_cloud_ns_ = 0;
      has_output_ = false;
      last_issue_ = "ros_clock_rollback: waiting for fresh IMU and cloud";
      ++clock_resets_;
    }
    last_clock_ns_ = now_ns;
  }

  void imu_callback(const unitree_hg::msg::IMUState::ConstSharedPtr message)
  {
    const auto now_ns = now().nanoseconds();
    observe_clock(now_ns);
    const Eigen::Quaterniond quaternion(message->quaternion[0], message->quaternion[1],
      message->quaternion[2], message->quaternion[3]);
    const double norm = quaternion.norm();
    if (!quaternion.coeffs().allFinite() || !std::isfinite(norm) || std::abs(norm - 1.0) > 0.05) {
      imu_history_.clear();
      has_valid_imu_ = false;
      last_issue_ = "invalid_imu_quaternion";
      ++invalid_imus_;
      return;
    }
    const auto steady_now = SteadyClock::now();
    imu_history_.push_back({now_ns, steady_now, quaternion.normalized()});
    while (!imu_history_.empty() &&
      (double(now_ns - imu_history_.front().receipt_ns) * 1e-9 > kImuHistorySeconds ||
      std::chrono::duration<double>(steady_now - imu_history_.front().steady_receipt).count() >
      kImuHistorySeconds || imu_history_.size() > kMaxImuHistorySamples))
    {
      imu_history_.pop_front();
    }
    has_valid_imu_ = true;
  }

  void reject(const std::string & issue)
  {
    last_issue_ = issue;
    ++rejected_clouds_;
  }

  void cloud_callback(const PointCloud::ConstSharedPtr cloud)
  {
    ++received_clouds_;
    const auto now_ns = now().nanoseconds();
    observe_clock(now_ns);
    if (cloud->header.frame_id != expected_lidar_frame_) {reject("unexpected_lidar_frame"); return;}
    if (cloud->header.stamp.sec < 0 || cloud->header.stamp.nanosec >= 1000000000U) {
      reject("invalid_cloud_timestamp"); return;
    }
    const std::int64_t cloud_ns = std::int64_t(cloud->header.stamp.sec) * 1000000000LL +
      cloud->header.stamp.nanosec;
    if (cloud_ns == 0) {reject("zero_cloud_timestamp"); return;}
    const double cloud_age = double(now_ns - cloud_ns) * 1e-9;
    if (cloud_age < -future_tolerance_ || cloud_age > max_cloud_age_) {
      reject("cloud_timestamp_outside_age_limits"); return;
    }
    if (cloud_ns <= last_accepted_cloud_ns_) {reject("repeated_or_out_of_order_cloud"); return;}
    if (!has_valid_imu_ || imu_history_.empty()) {reject("missing_valid_imu"); return;}
    const auto steady_now = SteadyClock::now();
    const double latest_imu_age = double(now_ns - imu_history_.back().receipt_ns) * 1e-9;
    if (latest_imu_age < -future_tolerance_ || latest_imu_age > max_imu_age_ ||
      std::chrono::duration<double>(steady_now - imu_history_.back().steady_receipt).count() >
      max_imu_age_)
    {
      reject("stale_imu_receipt"); return;
    }
    const auto nearest = std::min_element(imu_history_.begin(), imu_history_.end(),
      [cloud_ns](const ImuSample & a, const ImuSample & b) {
        return std::abs(double(a.receipt_ns - cloud_ns)) < std::abs(double(b.receipt_ns - cloud_ns));
      });
    const double offset = double(nearest->receipt_ns - cloud_ns) * 1e-9;
    const double imu_age = double(now_ns - nearest->receipt_ns) * 1e-9;
    // A Livox cloud can be stamped at frame start. The historical attitude
    // matched to that stamp is intentionally older than the newest IMU sample.
    const double selected_imu_age_limit = max_cloud_age_ + max_imu_cloud_offset_;
    if (std::abs(offset) > max_imu_cloud_offset_) {reject("imu_cloud_receipt_offset_too_large"); return;}
    if (imu_age < -future_tolerance_ || imu_age > selected_imu_age_limit ||
      std::chrono::duration<double>(steady_now - nearest->steady_receipt).count() >
      selected_imu_age_limit)
    {
      reject("stale_selected_imu"); return;
    }
    std::array<std::uint32_t, 3> offsets;
    std::string error;
    if (!xyz_layout(*cloud, offsets, error)) {reject(error); return;}
    std::vector<Eigen::Vector3d> points;
    points.reserve(std::size_t(cloud->width) * cloud->height);
    for (std::uint32_t row = 0; row < cloud->height; ++row) {
      for (std::uint32_t col = 0; col < cloud->width; ++col) {
        const auto * point = cloud->data.data() + std::size_t(row) * cloud->row_step +
          std::size_t(col) * cloud->point_step;
        points.emplace_back(read_float32(point + offsets[0], cloud->is_bigendian),
          read_float32(point + offsets[1], cloud->is_bigendian),
          read_float32(point + offsets[2], cloud->is_bigendian));
      }
    }
    Projection projection;
    try {
      projection = project(points, calibration_, nearest->world_from_torso);
    } catch (const std::exception & exception) {
      reject(std::string("projection_error: ") + exception.what()); return;
    }
    if (projection.occupied_cells == 0) {reject("no_valid_lidar_cells"); return;}
    FloorEstimate floor;
    FloorCompletion completed;
    completed.xyz = projection.xyz;
    if (floor_completion_enabled_) {
      floor = estimate_flat_floor(points, calibration_, nearest->world_from_torso);
      if (floor.valid) completed = complete_flat_floor(projection.xyz, nearest->world_from_torso, floor.height_m);
    }
    // Large frames or a clock jump must not turn a once-fresh input into a
    // stale policy observation while it is being parsed/projected.
    const auto publish_ns = now().nanoseconds();
    if (publish_ns < now_ns) {
      observe_clock(publish_ns);
      reject("ros_clock_rollback_during_projection"); return;
    }
    if (double(publish_ns - cloud_ns) * 1e-9 > max_cloud_age_ ||
      double(publish_ns - nearest->receipt_ns) * 1e-9 > selected_imu_age_limit ||
      std::chrono::duration<double>(SteadyClock::now() - nearest->steady_receipt).count() >
      selected_imu_age_limit ||
      double(publish_ns - imu_history_.back().receipt_ns) * 1e-9 > max_imu_age_ ||
      std::chrono::duration<double>(SteadyClock::now() - imu_history_.back().steady_receipt).count() >
      max_imu_age_)
    {
      reject("inputs_expired_during_projection"); return;
    }
    msg::LidarTensor tensor;
    tensor.header = cloud->header;
    tensor.header.frame_id = output_frame_;
    tensor.imu_received_stamp = rclcpp::Time(nearest->receipt_ns, get_clock()->get_clock_type());
    tensor.imu_cloud_offset_s = static_cast<float>(offset);
    tensor.extrinsics_verified = extrinsics_verified_;
    tensor.rows = kRows;
    tensor.columns = kCols;
    tensor.normalization_distance_m = static_cast<float>(kMaxDistance);
    tensor.input_points = static_cast<std::uint32_t>(projection.input_points);
    tensor.rejected_points = static_cast<std::uint32_t>(projection.rejected_points);
    tensor.occupied_cells = static_cast<std::uint32_t>(projection.occupied_cells);
    tensor.observed_cells = static_cast<std::uint32_t>(projection.occupied_cells);
    tensor.floor_filled_cells = 0;
    tensor.floor_height_m = floor.height_m;
    tensor.floor_fit_rms_m = floor.rms_m;
    tensor.floor_fit_inlier_points = static_cast<std::uint32_t>(floor.inlier_points);
    const auto & q = nearest->world_from_torso;
    tensor.torso_quaternion_wxyz = {q.w(), q.x(), q.y(), q.z()};
    tensor.xyz = projection.xyz;
    const auto observed = tensor;
    tensor.xyz = completed.xyz;
    tensor.floor_filled_cells = static_cast<std::uint32_t>(completed.filled_cells);
    tensor.occupied_cells += tensor.floor_filled_cells;
    PointCloud normalized;
    normalized.header = tensor.header;
    normalized.height = kRows;
    normalized.width = kCols;
    normalized.is_bigendian = false;
    normalized.is_dense = true;  // Empty cells use finite zero placeholders.
    normalized.point_step = 12;
    normalized.row_step = normalized.width * normalized.point_step;
    normalized.fields.resize(3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      auto & field = normalized.fields[axis];
      field.name = std::string(1, "xyz"[axis]);
      field.offset = static_cast<std::uint32_t>(axis * 4);
      field.datatype = sensor_msgs::msg::PointField::FLOAT32;
      field.count = 1;
    }
    normalized.data.resize(kValueCount * 4);
    for (std::size_t i = 0; i < kValueCount; ++i) {
      write_float32_le(tensor.xyz[i], normalized.data.data() + i * 4);
    }
    tensor_publisher_->publish(tensor);
    observed_publisher_->publish(observed);
    cloud_publisher_->publish(normalized);
    last_accepted_cloud_ns_ = cloud_ns;
    last_output_steady_ = SteadyClock::now();
    has_output_ = true;
    ++published_clouds_;
    last_occupied_cells_ = tensor.occupied_cells;
    last_observed_cells_ = projection.occupied_cells;
    last_filled_cells_ = completed.filled_cells;
    last_floor_ = floor;
    last_offset_s_ = offset;
    last_issue_ = "ok";
  }

  void publish_status()
  {
    const auto stamp = now();
    observe_clock(stamp.nanoseconds());
    const auto steady_now = SteadyClock::now();
    const double imu_age = imu_history_.empty() ? -1.0 :
      std::chrono::duration<double>(steady_now - imu_history_.back().steady_receipt).count();
    const double imu_ros_age = imu_history_.empty() ? -1.0 :
      double(stamp.nanoseconds() - imu_history_.back().receipt_ns) * 1e-9;
    const double output_age = has_output_ ?
      std::chrono::duration<double>(steady_now - last_output_steady_).count() : -1.0;
    Status status;
    status.name = std::string(get_name()) + "/input_contract";
    status.hardware_id = "livox_to_transformer_sru";
    status.level = Status::OK;
    status.message = last_issue_;
    if (!has_valid_imu_ || imu_age < 0 || imu_age > max_imu_age_ ||
      imu_ros_age < -future_tolerance_ || imu_ros_age > max_imu_age_)
    {
      status.level = Status::ERROR; status.message = "missing_or_stale_imu";
    } else if (last_issue_ != "ok") {
      status.level = Status::ERROR;
    } else if (!has_output_ || output_age > max_cloud_age_) {
      status.level = Status::ERROR; status.message = "missing_or_stale_lidar_output";
    } else if (double(stamp.nanoseconds() - last_accepted_cloud_ns_) * 1e-9 > max_cloud_age_) {
      status.level = Status::ERROR; status.message = "stale_input_cloud_stamp";
    } else if (!extrinsics_verified_) {
      status.level = Status::WARN; status.message = "extrinsics_unverified: preview tensors only, motor control disabled";
    }
    const auto add = [&status](const std::string & key, const std::string & value) {
        diagnostic_msgs::msg::KeyValue item; item.key = key; item.value = value;
        status.values.push_back(item);
      };
    add("imu_time_source", "receipt_time_not_acquisition_time");
    add("extrinsics_verified", extrinsics_verified_ ? "true" : "false");
    add("last_issue", last_issue_);
    add("grid", "16x41x3_vertical_major");
    add("imu_receipt_age_s", std::to_string(imu_age));
    add("imu_receipt_ros_age_s", std::to_string(imu_ros_age));
    add("last_output_age_s", std::to_string(output_age));
    add("last_imu_cloud_offset_s", std::to_string(last_offset_s_));
    add("received_clouds", std::to_string(received_clouds_));
    add("published_clouds", std::to_string(published_clouds_));
    add("rejected_clouds", std::to_string(rejected_clouds_));
    add("invalid_imus", std::to_string(invalid_imus_));
    add("clock_resets", std::to_string(clock_resets_));
    add("last_occupied_cells", std::to_string(last_occupied_cells_));
    add("flat_floor_completion_enabled", floor_completion_enabled_ ? "true" : "false");
    add("last_observed_cells", std::to_string(last_observed_cells_));
    add("last_floor_filled_cells", std::to_string(last_filled_cells_));
    add("floor_estimate_valid", last_floor_.valid ? "true" : "false");
    add("floor_height_m", std::to_string(last_floor_.height_m));
    add("floor_fit_rms_m", std::to_string(last_floor_.rms_m));
    add("floor_fit_inlier_points", std::to_string(last_floor_.inlier_points));
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = stamp;
    array.status.push_back(status);
    status_publisher_->publish(array);
  }

  Calibration calibration_;
  std::string expected_lidar_frame_, output_frame_;
  bool extrinsics_verified_{false};
  bool floor_completion_enabled_{false};
  FloorEstimate last_floor_;
  double max_cloud_age_{0.25}, max_imu_age_{0.10}, max_imu_cloud_offset_{0.05}, future_tolerance_{0.05};
  std::deque<ImuSample> imu_history_;
  bool has_valid_imu_{false}, has_output_{false};
  std::int64_t last_clock_ns_{0}, last_accepted_cloud_ns_{0};
  SteadyClock::time_point last_output_steady_;
  std::uint64_t received_clouds_{0}, rejected_clouds_{0}, published_clouds_{0}, invalid_imus_{0}, clock_resets_{0};
  std::size_t last_occupied_cells_{0};
  std::size_t last_observed_cells_{0}, last_filled_cells_{0};
  double last_offset_s_{0};
  std::string last_issue_{"waiting_for_inputs"};
  rclcpp::Subscription<PointCloud>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<unitree_hg::msg::IMUState>::SharedPtr imu_subscription_;
  rclcpp::Publisher<msg::LidarTensor>::SharedPtr tensor_publisher_;
  rclcpp::Publisher<msg::LidarTensor>::SharedPtr observed_publisher_;
  rclcpp::Publisher<PointCloud>::SharedPtr cloud_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};
}  // namespace transformer_sru_lidar_adapter

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<transformer_sru_lidar_adapter::LidarAdapterNode>());
  } catch (const std::exception & exception) {
    RCLCPP_FATAL(rclcpp::get_logger("transformer_sru_lidar_adapter"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
