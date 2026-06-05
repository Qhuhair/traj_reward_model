# vLLM 服务

该模块用于通过 vLLM 的 OpenAI 兼容服务启动本地 Qwen3.5-4B 模型。当前默认
配置基于本项目已验证过的运行方式：

- `CUDA_VISIBLE_DEVICES` 由 `--gpu-ids` 参数设置
- 默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`
- 日志追加写入 `models/vllm/log/`

## 启动

```bash
export MODEL_PATH=/datasets/zhehu/models/Qwen3.5-4B
python -m models.vllm.start --gpu-ids 3
```

`MODEL_PATH` 是本地模型目录。`--gpu-ids 3` 表示只使用物理 3 号 GPU。启动后
服务默认监听 `0.0.0.0:8000`，对外模型名为 `Qwen3.5-4B`。

兼容旧入口：

```bash
python models/serve.py --gpu-ids 3
```

## 测试

```bash
python models/test_model.py --base-url http://127.0.0.1:8000/v1
```

`--base-url` 是 OpenAI 兼容 API 根地址。默认测试 prompt 会要求模型回复
“连接成功”。

## 日志

默认日志文件：

```text
models/vllm/log/vllm_qwen35_4b.log
```

启动器以追加模式打开日志文件，因此每次重启都会保留之前的日志内容。
