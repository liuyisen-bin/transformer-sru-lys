# Livox → Transformer-SRU

ROS 2 Humble 传感器适配器：生成 `16×41×3=1968` 维 LiDAR 切片，不发布电机指令。
高层模型的完整输入为1984维，剩余16维由[部署控制器](../../../deploy/transformer_sru_real/README.md)组装。

## 安装与编译

Ubuntu 22.04、ROS 2 Humble、系统 Python 3.10。先按[ROS 官方说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)安装 Humble。
以下从项目根目录执行，请将仓库放在纯英文路径；本机 Humble 的 rosidl 在中文构建路径下会生成错误的 IDL 路径。

```bash
sudo apt install build-essential cmake libeigen3-dev python3-colcon-common-extensions \
  ros-humble-eigen3-cmake-module ros-humble-rosidl-default-generators \
  ros-humble-rosidl-generator-dds-idl ros-humble-rmw-cyclonedds-cpp
source /opt/ros/humble/setup.bash
mkdir -p third_party
git clone https://github.com/unitreerobotics/unitree_ros2.git third_party/unitree_ros2
git -C third_party/unitree_ros2 checkout 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
colcon --log-base ros/log build \
  --base-paths ros/src third_party/unitree_ros2/cyclonedds_ws/src/unitree \
  --build-base ros/build --install-base ros/install \
  --packages-up-to transformer_sru_lidar_adapter unitree_api \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source ros/install/setup.bash
```

若已安装 Unitree 消息包，也可先 source 其 `install/setup.bash`，仅编译 `--base-paths ros/src`、
`--packages-select transformer_sru_lidar_adapter`。本包本身依赖 `unitree_hg`，部署控制器另需 `unitree_api`。
[Unitree ROS 2](https://github.com/unitreerobotics/unitree_ros2) 提供这些接口；Livox 驱动、里程计应另行配置。

## 输入、启动与标定

| 接口 | 类型 | 用途 |
|---|---|---|
| `/livox/lidar` | `sensor_msgs/PointCloud2` | FLOAT32 x/y/z 原始点云，携带有效时间戳 |
| `/secondary_imu` | `unitree_hg/IMUState` | torso 四元数，wxyz |
| `/transformer_sru/lidar` | `LidarTensor` | 带时间戳和标定状态的导航输入 |
| `/transformer_sru/lidar_observed` | `LidarTensor` | 未补全的真实回波 |
| `/transformer_sru/lidar_status` | `diagnostic_msgs/DiagnosticArray` | 输入年龄、覆盖格数、拒绝原因 |

```bash
source /opt/ros/humble/setup.bash
source ros/install/setup.bash
ros2 launch transformer_sru_lidar_adapter livox_24999.launch.py
```

配置位于 `config/livox_24999.yaml`。默认 `extrinsics_verified: false`，输出可用于只读推理。
复制配置并填写本机点云话题、frame、LiDAR 相对 torso 的旋转/平移；核实 IMU 轴及时间基准后，
才在自己的配置中标记已验证，并通过 `config:=/absolute/path/calibrated.yaml` 加载。
所有附带安装数值都是示例，不能替代本机标定。

投影俯仰角 −45°～0°、方位角 −60°～60°、间隔3°，按行/列/XYZ展平；
每格取最近回波，转至虚拟雷达 yaw 对齐坐标后除以5，空格为0，最大距离5米。
虚拟雷达相对 torso 的平移为 `[0.0002835,0.00003,0.40618]` 米、旋转 `Ry(0.04014257279586953)`。
物理雷达外参与此虚拟安装独立。

`livox_24999_floor_compensated.yaml` 是可选水平地面补全示例：仅用本帧地面证据补空格，
真实回波不变；它不能恢复缺失障碍，不适用于未经验证的非水平地面。该示例也默认未标定。
`IMUState` 无源时间戳，使用 ROS 接收时间匹配点云；这不等于硬件同步或逐点去畸变。
默认拒绝过期、乱序、错误frame、空覆盖等输入。

## 离线检查

仅跑 MuJoCo 时不需要 ROS，直接构建同一投影源码：

```bash
cmake -S ros/src/transformer_sru_lidar_adapter -B ros/build_standalone -DSRU_STANDALONE=ON
cmake --build ros/build_standalone -j2
ctest --test-dir ros/build_standalone --output-on-failure
```

完整 ROS 构建后可运行合成输入检查。脚本固定使用 loopback 和独立 ROS domain：

```bash
python3 ros/src/transformer_sru_lidar_adapter/test/test_ros_adapter.py \
  --node ros/install/transformer_sru_lidar_adapter/lib/transformer_sru_lidar_adapter/lidar_adapter_node \
  --output outputs/ros_adapter_test
```

这些检查验证几何、消息和拒绝逻辑，不证明本机标定或机器人导航效果。
