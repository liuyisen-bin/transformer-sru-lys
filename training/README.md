# 训练与回放

运行目录均为仓库根目录。任务为 `Isaac-Navigation-PPO-G1-29DOF-v0`：
CNN + 两层空间 Transformer + SRU，PPO 训练高层导航；冻结的 Unitree 低层策略随仓库提供。

## 训练与 sim2sim 共用环境

Linux x86_64、NVIDIA GPU，Python **3.11**、Isaac Sim **5.1.0**、Isaac Lab **v2.3.0**、
PyTorch **2.7.0 + CUDA 12.8**、torchvision **0.22.0**、NumPy **1.26.0**。
先按 [Isaac Lab v2.3 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
安装驱动并确认 Isaac Sim 可启动。以下为新的 Conda 环境安装示例，Isaac Lab 可以位于任意目录：

```bash
conda create -n transformer_sru python=3.11 -y
conda activate transformer_sru
# 从本仓库根目录执行，使 Isaac Sim / Lab 安装共享同一组版本约束。
export PIP_CONSTRAINT="$PWD/training/requirements.txt"
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
mkdir -p third_party
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
cd third_party/IsaacLab
./isaaclab.sh --install none
cd ../..
```

然后进入本仓库根目录，在同一环境中安装：

```bash
python -m pip install -r requirements.txt
python -m pip install -e training/sru-navigation-learning -e training/source/engineai_rl_lab
python -m pip check
python -c "import rsl_rl; print(rsl_rl.__file__)"
```

根目录 `requirements.txt` 同时安装训练和 MuJoCo 演示依赖；两者共用 `transformer_sru`，
无需另建 Python 环境或改装 CPU 版 PyTorch。已有本仓库训练环境只需补装
`python -m pip install -r deploy/sim2sim/requirements.txt`，再按[场景安装与启动](../deploy/sim2sim/README.md)运行。

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
不能直接复制原环境的 ONNX 1.22.0 / psutil 7.2.2。`PIP_CONSTRAINT` 应保留到安装结束，
确保 Isaac Lab 安装阶段不会先升级这些包；安装后用 `pip check` 检查依赖一致性。
已核对官方包依赖元数据，但本次未在全新环境安装或完成 Isaac 训练，不能视为完整环境锁文件或重新训练复现结果。
离线检查通过了发布权重严格加载、2 个合成环境的 4 步 PPO 采样与 1 次更新、Unitree ONNX 的批量推理；这些不代表物理仿真或训练收敛。

## 训练

先用少量环境检查完整训练链路，再运行正式训练：

```bash
python training/scripts/nav/train.py --num_envs 2 --max_iterations 2 --smoke_test --headless
python training/scripts/nav/train.py --num_envs 2048 --max_iterations 25000 --headless
```

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
