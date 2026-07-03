# vLLM 服务

该模块用于通过 vLLM 的 OpenAI 兼容服务启动本地 Qwen3.5-4B 模型。当前默认
配置基于本项目已验证过的运行方式：

- `CUDA_VISIBLE_DEVICES` 由 `--gpu-ids` 参数设置
- 默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`
- 日志追加写入 `models/vllm/log/`

## 启动

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
python -m models.vllm.start --gpu-ids 4
```

`MODEL_PATH` 是本地模型目录。`--gpu-ids 4` 表示只使用物理 4 号 GPU。启动后
服务默认监听 `0.0.0.0:8002`，对外模型名为 `Qwen3.5-4B`。

## base / LoRA 调用标识

该服务只加载一份 base 权重，LoRA 通过 vLLM adapter 方式挂载。对外区分方式是
OpenAI 请求中的 `model` 字段：

- `model="Qwen3.5-4B"`：调用 base。
- `model="crossapp_kto"`：调用 `crossapp_kto` LoRA adapter。

加载 LoRA 时显式传入 checkpoint：

```bash
python -m models.vllm.start --gpu-ids 4 --lora-path /path/to/checkpoint --lora-name crossapp_kto
```

`--lora-path` 指向 LoRA checkpoint 目录；`--lora-name` 是 adapter 名称。
底层命令会自动添加 `--enable-lora --lora-modules crossapp_kto=/path/to/checkpoint`。

多个 LoRA 可使用多个 `--lora-module`：

```bash
python -m models.vllm.start --gpu-ids 4 --lora-module crossapp_kto=/path/to/kto --lora-module crossapp_sft=/path/to/sft
```

兼容旧入口：

```bash
python models/serve.py --gpu-ids 4
```

## 测试

```bash
python models/test_model.py --base-url http://127.0.0.1:8002/v1 --model Qwen3.5-4B --lora-model crossapp_kto
```

`--model` 是 base 模型名；`--lora-model` 是 LoRA adapter 名称。默认测试 prompt
会要求模型回复“连接成功”。

## 日志

默认日志文件：

```text
models/vllm/log/vllm_qwen35_4b_<YYYYMMDD>.log
```

启动器以追加模式打开日志文件，因此每次重启都会保留之前的日志内容。
