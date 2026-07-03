# Base API 模块

`framework_api/base_api/` 为 AndroidWorld 评测提供 base 策略模型接口。当前模块只实现
`POST /v1/act`，负责根据任务目标、截图、UI 元素和历史动作生成候选动作；PRM 打分接口不在本模块实现。

## 接口契约

请求：

```json
{
  "goal": "Add Alice to contacts",
  "step_index": 3,
  "screen_size": [1080, 2400],
  "screenshot": "base64_png",
  "ui_elements": [{"text": "Add contact", "type": "button", "bounds": [100, 200, 300, 280]}],
  "history": [],
  "n_candidates": 4,
  "model": "Qwen3.5-4B"
}
```

返回：

```json
{
  "candidates": [
    {
      "action": {"action_type": "click", "index": 0},
      "thought": "点击新增联系人按钮。"
    }
  ]
}
```

## 启动命令

```bash
python -m framework_api.base_api.start --host 127.0.0.1 --port 8101 --base-url http://127.0.0.1:8002/v1 --model Qwen3.5-4B
```

- `python -m framework_api.base_api.start`：启动 Base API 的 FastAPI 服务。
- `--host 127.0.0.1`：只监听本机，避免把服务暴露到外部网络。
- `--port 8101`：Base API 对外端口，orchestrator 的 `BaseClient` 应指向该端口。
- `--base-url http://127.0.0.1:8002/v1`：底层 vLLM/OpenAI 兼容服务地址。
- `--model Qwen3.5-4B`：调用 vLLM 中的 base 模型标识。

如果 base 模型支持图片输入，可追加 `--include-image`，服务会把 `screenshot` 作为
OpenAI 多模态 `image_url` 传入；默认不开启，避免纯文本模型报错。

## 环境变量

- `BASE_API_OPENAI_BASE_URL`：底层 OpenAI 兼容服务地址，默认 `http://127.0.0.1:8002/v1`。
- `BASE_API_MODEL`：默认 base 模型名，默认 `Qwen3.5-4B`。
- `BASE_API_KEY`：鉴权 token，默认 `EMPTY`，适配本地 vLLM。
- `BASE_API_INCLUDE_IMAGE`：设为 `1` 时传入截图。
- `BASE_API_TIMEOUT`：模型请求超时时间，默认 `120` 秒。

## 日志

Base API 日志追加写入：

```text
framework_api/logs/base/base_api_YYYYMMDD.log
```

日志内容包括：

- 启动命令产生的终端输出，包括 uvicorn 的 stdout/stderr。
- `/health` 和 `/v1/act` 的调用记录。
- 每次模型调用的模型名、底层服务地址、是否传入图片、耗时和响应长度。
- 调用失败时的异常堆栈。

PRM API 后续接入时使用：

```text
framework_api/logs/prm/
```

## 设计边界

本模块只做三件事：构造动作 Prompt、调用 base 模型、解析候选动作。解析失败时返回
`{"action_type": "wait"}` 保底动作，保证 AndroidWorld episode 不因一次格式错误直接中断。
