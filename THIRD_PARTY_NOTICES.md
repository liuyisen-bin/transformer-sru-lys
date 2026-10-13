# 第三方来源与许可证

本文件记录已有源码声明及第三方来源。根目录 [MIT 许可证](LICENSE) 仅覆盖 [LICENSE_SCOPE.md](LICENSE_SCOPE.md) 所述的 lys 原创贡献，不替换第三方许可。源码中的原版权声明继续有效。下列第三方许可原文于 2026-10-13 从所列官方地址取得，未修改正文；路径中的不同来源文件仍按各自声明适用。

| 本地范围 | 来源与原版权 | 许可原文 |
| --- | --- | --- |
| `training/scripts/nav/`、`training/source/engineai_rl_lab/engineai_rl_lab/tasks/` 中标有 MIT 的文件 | [SRU Navigation Sim](https://github.com/leggedrobotics/sru-navigation-sim)，Fan Yang、Per Frivik、Robotic Systems Lab、ETH Zurich | [MIT](licenses/sru-navigation-sim-MIT.txt) · [官方原文](https://raw.githubusercontent.com/leggedrobotics/sru-navigation-sim/main/LICENSE) |
| `training/sru-navigation-learning/` 中标有 BSD-3-Clause 的文件 | [SRU Navigation Learning](https://github.com/leggedrobotics/sru-navigation-learning)，基于 RSL-RL；ETH Zurich、Nikita Rudin、NVIDIA、Fan Yang；其他版权见各文件头 | [BSD-3-Clause](licenses/sru-navigation-learning-BSD-3-Clause.txt) · [官方原文](https://raw.githubusercontent.com/leggedrobotics/sru-navigation-learning/main/LICENSE) |
| `training/source/engineai_rl_lab/` 中标有 Isaac Lab 版权及 BSD-3-Clause 的文件 | [Isaac Lab](https://github.com/isaac-sim/IsaacLab)，The Isaac Lab Project Developers | [BSD-3-Clause](licenses/IsaacLab-BSD-3-Clause.txt) · [官方 v2.1.1 原文](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.1.1/LICENSE) |
| `training/sru-navigation-learning/rsl_rl/networks/sru_memory/attention.py` 中注明改编来源的位置编码 | [multidim-positional-encoding](https://github.com/tatp22/multidim-positional-encoding)，Peter Tatkowski | [MIT](licenses/multidim-positional-encoding-MIT.txt) · [官方原文](https://raw.githubusercontent.com/tatp22/multidim-positional-encoding/master/LICENSE) |
| `ros/src/transformer_sru_lidar_adapter/` 的现有包许可声明及 `src/lidar_adapter_node.cpp` 的 SPDX 声明 | 现有声明为 Apache-2.0；不据此推定未标注文件的来源或版权人 | [Apache-2.0](licenses/Apache-2.0.txt) · [Apache 官方原文](https://www.apache.org/licenses/LICENSE-2.0.txt) |

## 权重与机器人资产

以下文件保留用于训练或部署。2026-10-13 已对照下列官方仓库的固定提交核实文件内容；许可说明依据对应提交的上游许可证，适用于其覆盖的上游内容，不构成对任何额外第三方权利的授权或保证。

以下路径均相对 `training/source/engineai_rl_lab/engineai_rl_lab/tasks/navigation/assets/data/`：

| 文件 | 固定版本来源与核实结果 | 上游许可 |
| --- | --- | --- |
| `Policies/unitree_g1_velocity/policy.onnx`、`deploy.yaml` | [Unitree RL Lab，提交 `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/g1_29dof/config/policy/velocity/v0)：分别与 `exported/policy.onnx`、`params/deploy.yaml` 逐字节一致 | [Apache-2.0 原文](licenses/unitree_rl_lab-Apache-2.0.txt) · [固定版本官方原文](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/LICENCE) |
| `Robots/g1_29dof/g1_29dof.urdf`、`Robots/g1_29dof/meshes/` | [Unitree ROS，提交 `5994d4faef0a9cadd3287f8de0199a67eeb2a259`](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description)：36 个本地 mesh 的 Git blob SHA 均与上游同名文件一致；URDF 为本地修改版，主要差异见下文 | [BSD-3-Clause 原文](licenses/unitree_ros-BSD-3-Clause.txt) · [固定版本官方原文](https://raw.githubusercontent.com/unitreerobotics/unitree_ros/5994d4faef0a9cadd3287f8de0199a67eeb2a259/LICENSE)，Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") |
| `Policies/heightscan_encoder/vae_heightscan3.pth` | [SRU Navigation Sim，提交 `80f6c01c2d9f171374c9f9dba289abd38bcf0c6c` 的同名文件](https://github.com/leggedrobotics/sru-navigation-sim/blob/80f6c01c2d9f171374c9f9dba289abd38bcf0c6c/isaaclab_nav_task/navigation/assets/data/Policies/heightscan_encoder/vae_heightscan3.pth)：逐字节一致 | [MIT 原文](licenses/sru-navigation-sim-MIT.txt) · [固定版本官方原文](https://raw.githubusercontent.com/leggedrobotics/sru-navigation-sim/80f6c01c2d9f171374c9f9dba289abd38bcf0c6c/LICENSE)，Fan Yang、Per Frivik、Robotic Systems Lab、ETH Zurich |

本地 `g1_29dof.urdf` 相对上述 Unitree ROS 版本的主要修改：将上游两个 IMU 链接及关节替换为 `imu_link` / `imu_joint`；增加左右橡胶手的碰撞体；将左右踝俯仰、踝侧摆及腰部俯仰、侧摆共 6 个关节的 `effort` / `velocity` 从 `35` / `30` 改为 `50` / `37`；将材质颜色内联到各视觉元素。这些是文件差异记录，不代表实机参数或传感器外参已验证。

逐字节匹配权重及配置的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `Policies/unitree_g1_velocity/policy.onnx` | `610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f` |
| `Policies/unitree_g1_velocity/deploy.yaml` | `64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63` |
| `Policies/heightscan_encoder/vae_heightscan3.pth` | `3b07961f836e59bf03cb7031560d5bed330bb5957e5ae06bbb3ef303c3c5ca26` |

`checkpoints/model_24999.pt` 及其 ONNX 导出中，lys 拥有并有权授权的权利适用根目录 MIT 许可，具体边界见 [LICENSE_SCOPE.md](LICENSE_SCOPE.md)；这不授予第三方权利，也不替代第三方权重或资产的再分发依据。外部安装的 Isaac Sim、ROS、PyTorch、ONNX Runtime 等依赖遵循各自许可。
