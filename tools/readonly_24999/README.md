# 导出部署模型

运行环境见 [训练安装说明](../../training/README.md)。从项目根目录执行：

```bash
python tools/readonly_24999/export_onnx.py \
  --checkpoint checkpoints/model_24999.pt \
  --output outputs/model_24999.onnx
```

导出器读取 checkpoint 同级的 `<checkpoint文件名去扩展名>_config/agent.yaml`。
导出新训练的模型时，将该次训练的 `params/agent.yaml` 复制到对应目录；此工具仅支持
本项目 `ActorCriticSRURangeImage`、1984维观测和关闭 empirical normalization 的配置。

输出包含 ONNX 和 `.export.json`，并执行64步有状态 PyTorch/ONNX 数值对照。
接口为 `obs[1,1984]`、`h_in/c_in[1,1,512]` → `actions[1,3]`、`h_out/c_out[1,1,512]`。
每个新 episode 将 h/c 清零，后续逐步传递输出状态。ONNX 仅含高层导航策略，
动作映射和 Unitree 低层控制见 [部署说明](../../deploy/transformer_sru_real/README.md)。

仓库已附带 `checkpoints/model_24999.onnx`，部署该权重无需重新导出。
数值对照只验证导出接口，不代表导航效果或实机验证。

## 新训练模型的导出与切换

以下示例使用同网络结构、同观测处理和动作映射的新权重；将 `RUN_DIR` 改成实际训练输出目录，
并将 `model_29999` 改成实际检查点名称。命令从仓库根目录执行：

```bash
conda activate transformer_sru
RUN_DIR="/absolute/path/to/training/run"
mkdir -p "$RUN_DIR/model_29999_config"
cp "$RUN_DIR/params/agent.yaml" "$RUN_DIR/model_29999_config/agent.yaml"
python tools/readonly_24999/export_onnx.py \
  --checkpoint "$RUN_DIR/model_29999.pt" \
  --output outputs/model_29999.onnx
```

停止正在运行的仿真/部署进程，再选择新模型：

```bash
python tools/readonly_24999/select_model.py --onnx outputs/model_29999.onnx
# 已完成部署文档中的场景和 C++ 投影库构建后，先运行离线检查。
bash deploy/transformer_sru_real/mujoco_navigation.sh --check --seconds 15
```

选择工具检查配套 `.export.json` 的 64 步数值对照结果、实际文件 SHA256、ONNX 输入/输出的
名称/形状/类型、观测与动作元数据，并执行一次有限值推理检查。全部通过后，仅更新
`deploy/transformer_sru_real/contract.json` 的 `high_model` 和 `high_sha256`；
失败不修改该文件。路径相对仓库根目录，因此模型和报告须存放在本仓库内。
低层模型、关节顺序和控制参数保持原值。此工具适用于相同接口和预处理的权重更新；
若训练时修改了观测、动作映射或低层策略，必须先适配部署代码，不能仅更新哈希。

新选择会影响随后启动的 ONNX MuJoCo 检查和 ROS 部署，不影响直接读取 PT 的避障演示。
`outputs/` 被 Git 忽略；迁移到另一台机器时需一起复制 ONNX、同名 `.export.json` 和更新后的
`contract.json`，保持模型的仓库相对路径一致。接口检查不代表新策略的避障效果或实机验证。

恢复附带的 24999 模型：

```bash
python tools/readonly_24999/select_model.py --onnx checkpoints/model_24999.onnx
```
