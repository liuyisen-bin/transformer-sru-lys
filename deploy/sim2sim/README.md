# 24999 MuJoCo 障碍物演示

复现 2026-10-06 的场景：20 个错列障碍物和两侧墙体，目标为 `(12, 0, 0.5)` 米。
障碍物水平尺寸已按当时版本缩到原来的 60%，同列净空约 1.36–1.59 米。
`scene.json` 保留当时的布局；窗口显示的 22 个障碍物包含两侧墙。

[观看本次复测录像](demo.mp4) · [查看到达画面](preview.png)

直接加载仓库内的 `checkpoints/model_24999.pt`，采用原 Transformer-SRU 推理和冻结的
Unitree G1 低层策略。物理 500 Hz、低层 50 Hz、导航 5 Hz；CPU 推理，不需要 Isaac Lab、
ROS、机器人连接或 CUDA。没有额外规划器或中间路点。

## 安装

训练、Isaac 回放和本演示共用 **`transformer_sru` / Python 3.11**。先完成
[训练环境安装](../../training/README.md)，其中根目录 `requirements.txt` 已包含 MuJoCo 依赖。
共同依赖的版本统一由 `training/requirements.txt` 约束：PyTorch 2.7.0+cu128、
NumPy 1.26.0、ONNX 1.18.0；演示仍在 CPU 上推理，无需更换 PyTorch。

从仓库根目录执行；已按新版训练文档安装的环境可跳过 pip 补装：

```bash
conda activate transformer_sru
python -m pip install -r deploy/sim2sim/requirements.txt

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

2026-10-13 复测：图形模式与无窗口模式均在约 **21.91 秒**到达，结束距离 **0.227 米**，
未触发碰撞或跌倒判定。实际环境为 Python 3.11.16、PyTorch 2.7.0+cu128（CPU 推理）、
NumPy 1.26.0、MuJoCo 3.13.0、ONNX 1.22.0。

统一依赖复测：保持上述 Python / PyTorch / NumPy / MuJoCo，改用训练要求的
ONNX 1.18.0、ONNX Runtime 1.23.2，无窗口避障结果与原轨迹逐字节一致；
ONNX 仿真 15 秒接口检查通过，数值契约与 MuJoCo 回归共 9 项通过、3 项因缺少可选依赖跳过。
依赖解析通过；为保留本机环境，新增/调整的验证包放在项目内，仅对验证进程生效。
完整的新环境安装与 Isaac 训练未重新验证。

其他检查点或场景可使用 `--checkpoint`、`--scene-json`；官方 XML 可用 `--xml` 指定。
