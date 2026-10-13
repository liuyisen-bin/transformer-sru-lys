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
