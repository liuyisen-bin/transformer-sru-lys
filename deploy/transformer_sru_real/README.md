# G1 部署与 MuJoCo 验证

部署使用 `checkpoints/model_24999.onnx`，由仓库中的 `model_24999.pt` 导出。训练从 0 计数，24999 对应第 25000 轮；仓库未提供名为 `25000.pt` 的另一份模型。低层使用训练配套的 Unitree G1 `velocity/v0` ONNX。`contract.json` 固定模型哈希、观测布局和关节顺序；发布权重无需重新导出，导出说明见 [工具文档](../../tools/readonly_24999/README.md)。

以下命令均从仓库根目录执行。支持 Ubuntu 22.04 / Python 3.10；独立部署不需要安装 Isaac Lab。

## 本地 MuJoCo

```bash
sudo apt-get install python3-venv build-essential cmake libeigen3-dev libglfw3
python3.10 -m venv .venv-deploy
source .venv-deploy/bin/activate
python -m pip install -r deploy/transformer_sru_real/requirements-sim.txt
cmake -S ros/src/transformer_sru_lidar_adapter -B ros/build_standalone -DSRU_STANDALONE=ON
cmake --build ros/build_standalone -j
ctest --test-dir ros/build_standalone --output-on-failure
mkdir -p third_party
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
git -C third_party/unitree_mujoco checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
bash deploy/transformer_sru_real/mujoco_navigation.sh --check --seconds 15
bash deploy/transformer_sru_real/mujoco_navigation.sh
```

窗口按键：`F` 预备，等待约 3 秒；`P` 启动零速度低层；`G` 设置前方 7 m 目标；空格暂停；Backspace 停止；`R` 重置；Esc 退出。`--auto` 自动完成预备、策略和目标步骤。结果写到 `experiments/mujoco_navigation/`。

默认读取项目内的 Unitree 场景和独立投影库；已有安装可用 `--xml /absolute/path/scene_29dof.xml`（或 `SRU_MUJOCO_XML`）及 `SRU_LIDAR_LIBRARY` 指定。MuJoCo 使用理想 16×41 雷达、500 Hz 物理、50 Hz 低层、5 Hz 导航；它不连接 ROS/DDS。仿真运行成功只说明该场景和接口可运行，不证明实机传感器对齐或导航泛化。

## ROS 2 与真机

先安装 [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 和 [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)（需要 `unitree_hg`、`unitree_api`、Cyclone DDS），按 [LiDAR 适配器文档](../../ros/src/transformer_sru_lidar_adapter/README.md) 构建本项目 ROS 包。真机 Python 必须与 ROS Humble 的 Python 3.10 ABI 一致：

```bash
/usr/bin/python3 -m venv --system-site-packages .venv-robot
source .venv-robot/bin/activate
python -m pip install -r deploy/transformer_sru_real/requirements.txt
export SRU_PYTHON="$PWD/.venv-robot/bin/python"
export SRU_UNITREE_SETUP="$PWD/ros/install/setup.bash"
export SRU_NETWORK_INTERFACE=YOUR_ROBOT_NIC  # 替换为连接机器人的真实网卡名
source deploy/transformer_sru_real/env.sh
ros2 launch transformer_sru_lidar_adapter livox_24999.launch.py
```

`SRU_ROS_SETUP` 默认 `/opt/ros/humble/setup.bash`；`SRU_UNITREE_SETUP` 与 `SRU_ADAPTER_SETUP` 默认本项目 `ros/install/setup.bash`（按适配器文档一起构建消息与节点）。若 Unitree 消息装在独立工作区，将 `SRU_UNITREE_SETUP` 改为其 `install/setup.bash`。已有 `CYCLONEDDS_URI`、`ROS_DOMAIN_ID` 会保留。每个控制终端设置相同环境。启动前需自行提供并检查以下输入：

| 输入 | 要求 |
|---|---|
| `/lowstate` | Unitree G1 29 DOF，`mode_machine=5`、PR 模式，关节和 IMU 数据持续更新 |
| `/dog_odom` | `nav_msgs/Odometry`，`odom` → `robot_center`；位置在 odom，线速度在与 pelvis 对齐的 child frame；本仓库不提供里程计/SLAM |
| `/livox/lidar`、`/secondary_imu` | PointCloud2 与 torso IMU，供适配器生成 1968 维张量；本仓库不包含 Livox 驱动 |
| `/transformer_sru/lidar` | 16×41×3、XYZ/5、有效时间戳；实际安装外参、IMU 轴向、时序和覆盖率需现场验证 |

先保持默认只读，检查状态、观测及目标输入。只读模式不创建电机发布者，不释放已有运动服务：

```bash
bash deploy/transformer_sru_real/run.sh --duration 30
# 另一个使用相同环境的终端：
ros2 topic echo /transformer_sru/status
bash deploy/transformer_sru_real/goal.sh --forward 1 --left 0
```

完成机械支撑、场地与传感器核对，且无其他 `/lowcmd` 发布者后，才显式进入电机模式。导航要求标定配置的 `extrinsics_verified=true`；该标志必须基于实测，不能代替标定。先退出只读进程，再运行：

```bash
bash deploy/transformer_sru_real/run.sh --enable-lowcmd
# 另一个使用相同环境的终端逐条执行：
bash deploy/transformer_sru_real/stage.sh prepare
# 观察并确认预备姿态；插值时长 3 秒
bash deploy/transformer_sru_real/stage.sh start-policy
# 先验证零速度低层稳定，再发低幅目标
bash deploy/transformer_sru_real/goal.sh --forward 0.5 --left 0
bash deploy/transformer_sru_real/stop.sh
```

`FIX_STAND_READY` 仅表示插值结束，不表示实测稳定性通过；程序保留原切换逻辑。停止、Ctrl+C、手柄 `LT+B` 或 0.5 秒内双击 Select 均清空目标并退出；电机模式发送 `kp=0, kd=3` 阻尼，撤去站立刚度，需机械支撑。该软件停止不能代替硬件急停。手柄也支持 `LT+↑` 预备、`RB+X` 策略（先按住肩键，再按方向/X）；0.5 秒内双击 A 会设置前方 **7 m** 目标。

控制器保留模型哈希校验、输入时效、倾角、发布者所有权和发送看门狗。关节目标/估算力矩越界目前只记录诊断，不做额外裁剪。默认 `low_action_scale=0.25`；部署参数见 `config.yaml`。日志及实际送入模型的循环状态保存到 `experiments/real_deploy_logs/`。

## 离线回归

MuJoCo 依赖构建后可执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s deploy/transformer_sru_real -p test_mujoco_navigation.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s deploy/transformer_sru_real -p test_navigation_trace.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s deploy/transformer_sru_real -p test_startup.py
```

`test_contract.py` 还需要训练/导出环境的 PyTorch、ONNX 和仓库内 RSL-RL，用合成序列对照原生 Actor、训练观测与低层输出。未安装 ROS 时相关测试明确跳过；可选的原始 FSM/SDK 对照分别通过 `SRU_UNITREE_FSM_CONFIG` 和 `SRU_UNITREE_CRC_LIBRARY` 指定。`test_ros_isolated.py` 是 loopback domain 221 的模拟消息集成测试，包含模拟 `/lowcmd`，不属于纯数值测试。
