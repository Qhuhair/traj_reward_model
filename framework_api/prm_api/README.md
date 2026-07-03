# PRM API 模块

`framework_api/prm_api/` 为 AndroidWorld orchestrator 提供 `POST /v1/score` 接口。
该模块只负责对 base API 产生的候选动作评分，不执行动作。

## 支持模式

- `text_step`：只使用当前步骤文本信息和候选动作。
- `text_window`：使用当前步骤文本信息，并读取最近窗口步骤作为上下文。
- `multimodal_step`：使用当前步骤文本信息、截图和候选动作。
- `multimodal_window`：使用当前截图、当前文本信息和最近窗口步骤。
- `crossapp_multimodal_window`：复用 crossAPP 多模态窗口范式，使用 `LLMCaller`
  和 `RRM_Qwen_Sliding_Android_PRM` 对每个候选动作逐个评分。

默认模式是 `multimodal_window`，窗口大小由 `PRM_API_WINDOW_SIZE` 控制，默认 `3`。

## 启动命令

```bash
python -m framework_api.prm_api.start --host 127.0.0.1 --port 8102 --base-url http://127.0.0.1:8002/v1 --model crossapp_kto --mode crossapp_multimodal_window --window-size 3
```

- `python -m framework_api.prm_api.start`：启动 PRM API 服务。
- `--host 127.0.0.1`：只监听本机。
- `--port 8102`：PRM API 对外端口，orchestrator 的 `PRMClient` 应指向该端口。
- `--base-url http://127.0.0.1:8002/v1`：底层 vLLM/OpenAI 兼容服务地址。
- `--model crossapp_kto`：调用 vLLM 中加载 LoRA 后的 PRM 模型标识。
- `--mode crossapp_multimodal_window`：请求未传 `mode` 时使用的默认评分模式。
- `--window-size 3`：窗口模式读取最近 3 个已保存步骤。

`crossapp_multimodal_window` 通过 `PRM_API_LLM_MODEL` 和 `PRM_API_LLM_PROMPT`
控制复用的 LLM_caller 配置，默认分别为 `qwen_vllm_mm` 和
`RRM_Qwen_Sliding_Android_PRM`。该模式复用 crossAPP 的窗口上下文、多模态调用
和 XML 评分解析方式，但语义改为“候选动作执行前评分”，不会读取未来步骤。

## 轨迹与日志保存

服务日志写入：

```text
framework_api/logs/prm/prm_api_YYYYMMDD.log
```

每一步评分输入、截图和输出写入：

```text
framework_api/logs/prm/trajs/<任务类别>/<任务名>/<episode_id>/
├── images/
├── prompts/
├── steps/
├── score_inputs.jsonl
├── score_outputs.jsonl
└── finished.json
```

如果请求没有提供 `task_category`、`task_name` 或 `episode_id`，服务会从 `extras`
字段或 `goal` 中推断。最佳候选动作为
`{"action_type":"status","goal_status":"task_complete"}` 时，当前轨迹视为完成。
