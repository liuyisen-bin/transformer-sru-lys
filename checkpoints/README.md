# 发布权重

`model_24999.pt` 是本项目约25000轮训练的检查点（保存的迭代编号为24999）。
本发布不包含名为 `25000.pt` 或 `model_25000.pt` 的文件，也未重命名权重。

- `model_24999.pt`：完整训练检查点，可用于回放、恢复训练或导出。
- `model_24999.onnx`：对应高层导航 Actor，部署使用；不包含 Unitree 低层控制。
- `model_24999.export.json`：原导出时的接口、哈希和数值对照记录。
- `model_24999_config/agent.yaml`：重建网络所需原训练配置。
- `model_24999_config/env.yaml`：原训练环境记录，仅将两条本机绝对资产路径改为仓库相对路径；
  它含 Python 类型标签，不是独立可执行配置，训练入口使用 `training/source/` 中的配置类。

SHA256：

```text
79be96b2877af6cf0542800114ef9244a9d1d6431ceb64ad5d6b4863f0e303ab  model_24999.pt
539bbe3fd3edddf920514830ae1b8d052c8d0d08f69cc2bdc3d1ec00618991ad  model_24999.onnx
```

重新导出见 [导出说明](../tools/readonly_24999/README.md)，部署见 [部署说明](../deploy/transformer_sru_real/README.md)。
