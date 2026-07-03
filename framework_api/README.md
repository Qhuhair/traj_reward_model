# Framework API 模块

`framework_api/` 是整体轨迹评估框架的对外 API 抽象层。当前阶段只搭建接口地基，
不绑定 `pipelines/`、`LLM_caller/`、`core_prm/` 或 `filter/` 的具体实现。

## 设计目标

- 统一外部请求入口，支持轨迹评分、单步评分、数据筛选、流水线运行和结果对比等能力。
- 支持不同输入形态，例如路径、JSON、纯文本、多模态输入和批量输入。
- 保持模块化：API 层只做 schema、注册、分发和错误封装，具体能力由 handler 后续扩展。
- 遵守开闭原则：新增能力时注册新的 handler，不修改核心服务分发逻辑。

## 目录结构

```text
framework_api/
├── __init__.py        # 对外导出稳定接口
├── schema.py          # APIRequest / APIResponse / TaskType / InputMode
├── interfaces.py      # Handler / FrameworkService 协议
├── registry.py        # 能力注册表
├── service.py         # 最小服务编排层
├── errors.py          # API 层异常
├── logging_utils.py   # framework_api/base 与 prm 的日期日志工具
├── logs/              # API 服务日志目录，base/prm 分开存放
├── base_api/          # AndroidWorld Base 策略 HTTP API，提供 /v1/act
├── prm_api/           # AndroidWorld PRM 评分 HTTP API，提供 /v1/score
├── android_world/     # AndroidWorld 评测适配抽象
└── README.md
```

## 抽象请求格式

```python
from framework_api import APIRequest, InputMode, TaskType

request = APIRequest(
    task_type=TaskType.EVALUATE_TRAJECTORY,
    input_mode=InputMode.PATH,
    payload={"trajectory_path": "trajs/<set>/traj_007"},
    options={"input_mode": "multimodal"},
)
```

`payload` 存放输入数据，`options` 存放运行参数。具体字段由对应 handler 定义。

## 当前边界

当前模块不直接启动模型、不读取轨迹、不写入 `output/`，也不调用付费 API。
后续接入具体实现时，应在独立 handler 中适配已有模块，例如：

- `pipelines.full`：完整轨迹流水线。
- `core_prm`：过程奖励计算。
- `filter`：轨迹筛选。
- `LLM_caller`：单步或多模态评分。
- `framework_api/base_api`：AndroidWorld base 策略动作生成接口。
- `framework_api/android_world`：AndroidWorld 评测计划、命令和结果摘要。

## Base API

`framework_api/base_api/` 当前提供 `POST /v1/act`，用于 AndroidWorld orchestrator
向 base 策略模型请求候选动作。该接口内部调用 vLLM/OpenAI 兼容的
`/v1/chat/completions`，默认模型名为 `Qwen3.5-4B`。

启动示例：

```bash
python -m framework_api.base_api.start --host 127.0.0.1 --port 8101 --base-url http://127.0.0.1:8002/v1 --model Qwen3.5-4B
```

详细契约见：[base_api/README.md](base_api/README.md)。

## PRM API

`framework_api/prm_api/` 当前提供 `POST /v1/score`，用于 AndroidWorld orchestrator
向 PRM 模型请求候选动作分数。该接口支持 `text_step`、`text_window`、
`multimodal_step`、`multimodal_window` 和 `crossapp_multimodal_window` 等输入方式。
其中 `crossapp_multimodal_window` 复用 `LLMCaller`、crossAPP 窗口上下文组织和
XML 评分解析方式，默认调用模型名为 `crossapp_kto`。

启动示例：

```bash
python -m framework_api.prm_api.start --host 127.0.0.1 --port 8102 --base-url http://127.0.0.1:8002/v1 --model crossapp_kto --mode crossapp_multimodal_window --window-size 3
```

详细契约见：[prm_api/README.md](prm_api/README.md)。

## 日志约定

API 日志统一放在 `framework_api/logs/` 下：

- `framework_api/logs/base/base_api_YYYYMMDD.log`：Base API 日志。
- `framework_api/logs/prm/prm_api_YYYYMMDD.log`：PRM API 日志。
- `framework_api/logs/prm/trajs/`：PRM 每一步评分输入、截图和输出归档。

日志采用追加写入，记录服务启动后的终端输出、接口调用摘要、模型调用耗时和错误堆栈。

## 扩展方式

新增能力时实现 `Handler` 协议，并注册到 `HandlerRegistry`：

```python
registry.register(MyTrajectoryHandler())
service = ModularFrameworkService(registry)
response = service.execute(request)
```

每个 handler 只负责一种任务能力，避免把外部 API 层写成新的大流水线。
