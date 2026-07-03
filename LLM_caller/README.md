# LLM Caller — LLM 评分生成器

调用大语言模型为 GUI 轨迹每一步生成结构化评分 (`<think>` / `<critique>` / `<score>`)。

## 目录结构

```
LLM_caller/
├── config.yaml                        # 模型及 API 配置
├── caller.py                          # LLMCaller — 核心调度 + 模板渲染
├── main.py                            # CLI 入口, 并发调用 + --indices 支持
├── utils.py                           # ResponseParser (解析 XML 标签) + DebugLogger
├── adapters/                          # API 适配器 (策略模式)
│   ├── base.py                        #   BaseAdapter 抽象类
│   ├── deepseek_adapters.py           #   OpenAIStyleAdapter (DeepSeek/OpenAI 兼容 API)
│   ├── qwen_vllm_adapter.py           #   QwenVLLMAdapter (OpenAI 兼容, 支持 before/after 截图)
│   ├── responses_adapter.py           #   ResponsesAdapter (Responses API 多模态基准)
│   └── ollama_adapter.py              #   OllamaAdapter (历史兼容)
└── prompts/
    ├── RRM_V1.yaml                    #   模板: DeepSeek 用, 中文指令
    └── RRM_Qwen.yaml                  #   模板: Qwen 本地模型用, 严格格式 + /no_think
```

## 快速开始

```bash
# 单步测试
cd LLM_caller
python -c "from caller import LLMCaller; r=LLMCaller().call(task_desc='...', ...); print(r)"

# 处理完整轨迹 (CLI)
python main.py ../output/20250113_21442_test/traj_007/standardized.json llm_output.json
```

## 核心设计

### 适配器模式

| 适配器 | 协议 | 端点 | 用途 |
|--------|------|------|------|
| `OpenAIStyleAdapter` | `openai_style` | `/v1/chat/completions` | DeepSeek, 远程 Qwen |
| `QwenVLLMAdapter` | `qwen_vllm` | `/chat/completions` | 当前默认本地 Qwen 后端 |
| `ResponsesAdapter` | `responses` | `/responses` | Codex/OpenAI Responses 多模态基准 |
| `OllamaAdapter` | `ollama` | `/api/chat` | 历史兼容的本地 Ollama 后端 |

切换模型只需改 `config.yaml` 一行:
```yaml
active_model: "deepseek"          # → DeepSeek
active_model: "qwen_vllm_text"    # → vLLM 纯文本 Qwen
active_model: "qwen_vllm_mm"      # → vLLM 多模态 Qwen
active_model: "codex_mm_baseline" # → Responses API 多模态基准
```

### Responses 多模态基准测试

`codex_mm_baseline` 使用 `OPENAI_API_KEY` 环境变量读取密钥，不要把 key 写入仓库。

离线检查 payload 结构：

```bash
python LLM_caller/test_responses_multimodal.py
```

命令说明：读取默认 `traj_007` 的第 1 步，构造 Responses API 多模态请求，不发起网络调用；输出图片块数量、文本块数量和模型名。

在线测试单步：

```bash
OPENAI_API_KEY="你的key" python LLM_caller/test_responses_multimodal.py --online --step-idx=1
```

命令说明：`OPENAI_API_KEY` 指定 API 密钥；`--online` 表示真实调用接口；`--step-idx=1` 只测试第 1 步，避免一次性产生大量调用费用。

### Prompt 模板

| 模板文件 | 适用模型 | 特点 |
|---------|---------|------|
| `RRM_V1.yaml` | DeepSeek | 中文指令, 详细格式要求 |
| `RRM_Qwen.yaml` | Qwen3.5-4B | 严格标签格式, 预置 `/no_think`, system_message |

切换模板: `config.yaml` → `active_prompt: "RRM_Qwen"`

### 并发调用

`main.py` 使用 `ThreadPoolExecutor(max_workers=4)` 并发调用 LLM，将 20 步轨迹耗时从 ~13 分钟降至 ~4 分钟。

### --indices 选择性重跑

```bash
# 仅重跑步骤 3,7,12
python main.py input.json output.json --indices=3,7,12
```

用于 QA 审查失败后的重试机制。

## 添加新模型

1. 在 `adapters/` 下创建新适配器，继承 `BaseAdapter`
2. 在 `caller.py` 的 `adapter_map` 中注册
3. 在 `config.yaml` 的 `models` 下添加配置

## 配置示例

```yaml
active_model: "qwen_vllm_text"
active_prompt: "RRM_Qwen"
debug: true

models:
  deepseek:                     # 云端 DeepSeek
    protocol: "openai_style"
    base_url: "https://api.deepseek.com"
    model_name: "deepseek-v4-pro"

  qwen_vllm_text:               # 本地 vLLM Qwen
    protocol: "qwen_vllm"
    base_url: "http://localhost:8002/v1"
    model_name: "Qwen3.5-4B"
    max_tokens: 2048
    system_message: "直接输出think critique score三个XML标签的评估结果"
```
