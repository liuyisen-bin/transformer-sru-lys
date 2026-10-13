# 24999 MuJoCo 障碍物演示

复现 2026-10-06 的场景：20 个错列障碍物和两侧墙体，目标为 `(12, 0, 0.5)` 米。
障碍物水平尺寸已按当时版本缩到原来的 60%，同列净空约 1.36–1.59 米。
`scene.json` 保留当时的布局；窗口显示的 22 个障碍物包含两侧墙。

[观看本次复测录像](demo.mp4) · [查看到达画面](preview.png)

直接加载仓库内的 `checkpoints/model_24999.pt`，采用原 Transformer-SRU 推理和冻结的
Unitree G1 低层策略。物理 500 Hz、低层 50 Hz、导航 5 Hz；CPU 推理，不需要 Isaac Lab、
ROS、机器人连接或 CUDA。没有额外规划器或中间路点。

## 安装

按用途选择下面一种方式，**不需要同时创建两个环境**。共同依赖沿用训练的版本约束：
Python 3.11、PyTorch 2.7.0、NumPy 1.26.0、ONNX 1.18.0。
以下命令均从仓库根目录执行。

**已有训练环境：** 复用 `transformer_sru`，保留已有 CUDA 版 PyTorch，只补装仿真依赖。
按新版[训练文档](../../training/README.md)安装过根目录 `requirements.txt` 的用户可跳过补装。

```bash
conda activate transformer_sru
python -m pip install -r deploy/sim2sim/requirements.txt
```

**只运行 CPU 避障演示：** 无需 NVIDIA GPU、CUDA 或 Isaac Lab，也不用安装训练包。

```bash
conda create -n transformer_sru_sim python=3.11 -y
conda activate transformer_sru_sim
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deploy/sim2sim/requirements.txt
python -m pip check
```

**两种方式均需下载场景资源：**

```bash
# 仅首次安装场景时执行；已有该目录则跳过 clone。
mkdir -p third_party
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
git -C third_party/unitree_mujoco checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
```

显示窗口需要本机桌面和可用的 OpenGL 驱动；无桌面的服务器使用 `--headless`。
机器人网格和 XML 来自固定版本的 Unitree MuJoCo，许可保留在下载目录中。
本演示不依赖空场景 ONNX 演示的 C++ 投影库。

## 运行

```bash
bash deploy/sim2sim/run.sh
```

窗口倒计时 5 秒，机器人稳定站立 3 秒后自动导航。绿色圆盘是目标。
最多测试 90 秒，到达、碰撞或跌倒后冻结画面，保留 20 秒；按 **Esc** 退出。
无需按 F/P/G。脚本使用当前环境的 Python；可通过 `SRU_PYTHON` 显式指定解释器。

无窗口评估：

```bash
bash deploy/sim2sim/run.sh --headless --start-delay 0 --hold 0
```

每次结果保存到新的 `experiments/sim2sim_obstacles/<时间>/`：

- `manifest.json`：模型、场景哈希及 Python/PyTorch/NumPy/MuJoCo 版本。
- `result.json`：到达、碰撞、跌倒或超时结果及距离、耗时。
- `trajectory.csv`：实际轨迹、控制指令和雷达命中数。
- 图形模式额外生成 `evaluation.mp4`、`initial.png` 和 `final.png`。

到达条件为平面距离小于 0.5 米并持续 1 秒。到达返回退出码 0，碰撞、跌倒或超时返回 1；
手动关闭窗口返回 0，但结果会标记为 `user_closed` 或 `closed_before_start`。
结束时是冻结仿真，不代表验证了自主停车。这是固定场景和理想射线雷达演示，不代表总体成功率或实机表现。

2026-10-13 验证：复用训练环境与全新纯 CPU 环境均约 **21.91 秒到达**，结束距离
**0.227 米**，未触发碰撞或跌倒。纯 CPU 环境使用 Python 3.11.16、PyTorch 2.7.0+cpu、
NumPy 1.26.0、ONNX 1.18.0、MuJoCo 3.13.0，未安装 Isaac Lab/CUDA，`pip check` 通过。
训练环境使用 PyTorch 2.7.0+cu128 做 CPU 推理。Isaac PPO 冒烟的验证范围见[训练文档](../../training/README.md)。

更换同网络结构、同观测与动作配置的 PT 权重：

```bash
bash deploy/sim2sim/run.sh --checkpoint /absolute/path/model_29999.pt
```

场景可通过 `--scene-json` 更换；官方 XML 可用 `--xml` 指定。
本 PT 演示不读取部署 `contract.json`，不会切换 ONNX 或真机使用的模型。
ONNX 模型切换见[导出与选择部署模型](../../tools/readonly_24999/README.md)。
