# Models 模块

`models/` 是本仓库的本地模型服务辅助模块。它不是主流水线阶段，而是为
`LLM_caller/` 提供本地 OpenAI 兼容模型服务的启动、测试和下载工具。

## 目录结构

```text
models/
├── serve.py              # vLLM 启动兼容入口
├── test_model.py         # OpenAI 兼容接口测试入口
├── down_model.py         # 参数化 Hugging Face 文件下载脚本
└── vllm/
    ├── config.py         # vLLM 默认服务配置
    ├── command.py        # 命令和环境变量构建
    ├── process.py        # 后台进程启动和追加写入日志
    ├── cli.py            # CLI 参数解析和启动编排
    ├── start.py          # python -m 入口
    ├── client.py         # OpenAI 兼容聊天接口冒烟测试
    └── log/              # 运行日志目录
```

## 启动 vLLM 服务

先设置本地模型目录：

```bash
export MODEL_PATH=/datasets/zhehu/models/Qwen3.5-4B
```

`MODEL_PATH` 指向本地 Hugging Face 模型目录。该目录应包含 `config.json`、
tokenizer 文件和 safetensors 权重分片。

启动服务：

```bash
python -m models.vllm.start --gpu-ids 3
```

`python -m models.vllm.start` 表示从仓库根目录以模块方式启动 vLLM 服务。
`--gpu-ids 3` 会设置 `CUDA_VISIBLE_DEVICES=3`，让 vLLM 只使用物理 3 号 GPU。
默认配置会禁用 FlashInfer sampler，即设置 `VLLM_USE_FLASHINFER_SAMPLER=0`；
默认端口是 `8000`，对外模型名是 `Qwen3.5-4B`，日志追加写入
`models/vllm/log/vllm_qwen35_4b.log`。

兼容旧入口：

```bash
python models/serve.py --gpu-ids 3
```

## 测试接口

```bash
python models/test_model.py --base-url http://127.0.0.1:8000/v1
```

`--base-url` 指定 OpenAI 兼容 API 根地址。默认模型名为 `Qwen3.5-4B`。

## 停止服务

```bash
pkill -f "vllm"
```

该命令会终止命令行中包含 `vllm` 的后台进程。

## 维护规则

保持文件短小且职责单一。启动默认值放在 `vllm/config.py`，命令构建放在
`vllm/command.py`，进程处理放在 `vllm/process.py`，CLI 行为放在
`vllm/cli.py`。不要硬编码机器相关模型路径；使用 `MODEL_PATH` 或
`--model-path` 传入。
