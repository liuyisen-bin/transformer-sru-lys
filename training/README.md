# 训练与回放

运行目录均为仓库根目录。任务为 `Isaac-Navigation-PPO-G1-29DOF-v0`：
CNN + 两层空间 Transformer + SRU，PPO 训练高层导航；冻结的 Unitree 低层策略随仓库提供。

## 训练与 sim2sim 共用环境

Linux x86_64、NVIDIA GPU，Python **3.11**、Isaac Sim **5.1.0**、Isaac Lab **v2.3.0**、
PyTorch **2.7.0 + CUDA 12.8**、torchvision **0.22.0**、NumPy **1.26.0**。
先按 [Isaac Lab v2.3 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
安装驱动并确认系统满足 Isaac Sim 要求。以下从仓库根目录执行；环境名可自定（本次验收使用 `lys`）。
已有环境及源码可跳过 `conda create` 和 `git clone`，其余安装步骤仍在同一环境执行。
任何命令报错都应先停止处理；不要仅凭上游安装脚本退出码判断成功。

```bash
conda create -n transformer_sru python=3.11 -y
conda activate transformer_sru
# 从本仓库根目录执行，使 Isaac Sim / Lab 安装共享同一组版本约束。
# pip 26.2 的运行依赖约束与隔离构建约束是独立的。
python -m pip install pip==26.2.1
export PIP_CONSTRAINT="$PWD/training/requirements.txt"
export PIP_BUILD_CONSTRAINT="$PWD/training/build-constraints.txt"
# wheel 仅在隔离构建环境使用，避免运行环境与 packaging==23.0 冲突。
python -m pip uninstall -y wheel
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
mkdir -p third_party
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
# 单独安装核心包，确保其错误不会被上游批量安装脚本的退出码掩盖。
python -m pip install -e third_party/IsaacLab/source/isaaclab
cd third_party/IsaacLab
./isaaclab.sh --install none
cd ../..
```

然后进入本仓库根目录，在同一环境中安装：

```bash
# 上游切换 Torch 时可能卸载 torchaudio，在同一 CUDA 索引补齐匹配版本。
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e training/sru-navigation-learning -e training/source/engineai_rl_lab
python -m pip check
python - <<'PYTHON'
from importlib.metadata import version
from isaaclab.app import AppLauncher
import torch, torchvision, torchaudio, rsl_rl
assert version("isaaclab") == "0.47.2"
assert torch.__version__ == "2.7.0+cu128"
assert torchvision.__version__ == "0.22.0+cu128"
assert torchaudio.__version__ == "2.7.0+cu128"
print(rsl_rl.__file__)
PYTHON
```

根目录 `requirements.txt` 同时安装训练和 MuJoCo 演示依赖；两者共用 `transformer_sru`，
无需另建 Python 环境或改装 CPU 版 PyTorch。已有本仓库训练环境只需补装
`python -m pip install -r deploy/sim2sim/requirements.txt`，再按[场景安装与启动](../deploy/sim2sim/README.md)运行。

所有安装命令和 `pip check` 必须成功；有错误时先停止，不要直接运行训练。
最后一行应指向本仓库 `training/sru-navigation-learning/rsl_rl`。
本项目需要随仓库提供的 SRU 版 `rsl_rl`；Isaac Lab 安装时使用 `--install none`，避免额外安装
覆盖同名模块的 `rsl-rl-lib`。G1 URDF/网格、Unitree ONNX 和训练 Critic 所需高度编码器已包含在
`source/engineai_rl_lab/engineai_rl_lab/tasks/navigation/assets/data/`。首次场景启动仍需下载 NVIDIA 材质/天空纹理。

版本依据：发布整理时只读核对了原环境，Isaac Lab Git 标签为 v2.3.0（Python 包元数据为
`isaaclab==0.47.2`），原二进制 Isaac Sim 为 `5.1.0-rc.19`，Python 为 3.11.16，Torch 为 2.7.0+cu128。
正式 pip 版与原二进制环境的附加包约束不同：
[`isaacsim-kernel 5.1.0`](https://pypi.org/pypi/isaacsim-kernel/5.1.0/json)
固定 `psutil==5.9.8` 和 `typing_extensions==4.12.2`，因此本仓库采用
[`onnx==1.18.0`](https://pypi.org/pypi/onnx/1.18.0/json)，满足 Isaac Lab 的 `onnx>=1.18.0`。
不能直接复制原环境的 ONNX 1.22.0 / psutil 7.2.2。两个约束环境变量应保留到安装结束。
`flatdict==4.0.1` 的构建仍需要 `pkg_resources`，因此隔离构建使用 setuptools 81.0.0；
其中的 wheel / packaging 与 Isaac Sim 的运行环境隔离。不要将构建约束文件用 `pip install -r` 安装。
构建约束机制见 [pip 官方说明](https://pip.pypa.io/en/stable/user_guide/#build-constraints)。
验证范围（2026-10-13）：在 Ubuntu 22.04 / RTX 3060 Laptop 6 GiB / Python 3.11.17 的
`lys` 环境按上述约束修复安装后，`pip check`、实际包导入和 2 环境 × 2 轮 Isaac PPO 冒烟通过。
TensorBoard 包含第 0、1 轮损失记录，生成 `model_0.pt`、`model_1.pt`，优化器和参数均有更新。
这是首次全新安装失败后的同一环境修复验证；未再次重建全新环境，也不代表正式训练收敛。

## 训练

先用少量环境检查完整训练链路，再运行正式训练：

```bash
python -u training/scripts/nav/train.py --num_envs 2 --max_iterations 2 --smoke_test --headless
python training/scripts/nav/train.py --num_envs 2048 --max_iterations 25000 --headless
```

`-u` 让训练日志立即输出，避免 Isaac 退出时丢失缓冲中的最后一轮日志。
`--smoke_test` 使用小型 PLAY 地形且只设一个 mini-batch，验证启动/采样/更新，不用于报告策略效果。
正式训练按显存调整 `--num_envs`；其默认地形为 6×30 个地块，仅降低环境数不保证适合所有显卡。
日志和权重写入 `logs/rsl_rl/g1_29dof_unitree_transformer_sru_ppo/<时间>/`，每 100 次保存。
TensorBoard：`tensorboard --logdir logs/rsl_rl`。

继续训练时给出目标总迭代次数：

```bash
python training/scripts/nav/train.py --num_envs 2048 --headless \
  --checkpoint checkpoints/model_24999.pt --max_iterations 30000
```

## 回放与部署权重

仓库中的实际文件名是 `model_24999.pt`，保留原始文件名和权重，不另造 `25000.pt`。

```bash
python training/scripts/nav/play.py --num_envs 1 --checkpoint checkpoints/model_24999.pt
```

回放使用小型 PLAY 场景，退出用 Ctrl+C。实机部署与不启动 Isaac 的 ONNX 导出见
[仓库说明](../README.md)。高层检查点不能直接发送关节控制命令。

`config/pm01/`、SRU/PointNet 基类和高度编码器仍被 G1 配置或网络继承/引用，故保留；
本发行目录只注册 G1 任务，不提供 PM01 机器人训练资产。
