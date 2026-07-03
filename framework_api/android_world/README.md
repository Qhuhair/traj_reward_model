# AndroidWorld 评测适配

`framework_api/android_world/` 用于把本项目的对外 API 抽象和 AndroidWorld 评测连接起来。
当前阶段只搭建评测配置、命令生成和结果摘要读取，不直接启动模拟器，也不修改
`~/code/android_mobile_world_bench/android_world` 源码。

## 目标实验

当前目标是支持以下对比：

1. `Base`：base 模型作为策略模型。
2. `Base + Baseline PRM`：base 策略 + 基准 PRM。
3. `Base + LoRA PRM`：base 策略 + LoRA 微调模型作为 PRM。

vLLM 服务侧约定：

- `model="Qwen3.5-4B"`：策略模型 base。
- `model="crossapp_kto"`：LoRA PRM。

## 当前模块边界

本模块现在只生成评测计划：

- AndroidWorld 根目录。
- `run.py` 命令。
- 策略模型和 PRM 的环境变量标识。
- checkpoint 结果摘要读取函数。

真正的 PRM 规划 agent 后续应独立实现，读取：

```text
PRM_API_BASE_URL
PRM_POLICY_MODEL
PRM_REWARD_MODEL
```

这样可以避免把 AndroidWorld、vLLM、PRM 策略逻辑写死在 API 层。

## API 示例

```python
from framework_api import APIRequest, InputMode, ModularFrameworkService, TaskType
from framework_api.android_world import AndroidWorldEvalHandler

service = ModularFrameworkService()
service.registry.register(AndroidWorldEvalHandler())

response = service.execute(APIRequest(
    task_type=TaskType.RUN_PIPELINE,
    input_mode=InputMode.JSON,
    payload={
        "tasks": ["ContactsAddContact"],
        "policy_model": "Qwen3.5-4B",
        "prm_model": "crossapp_kto",
        "base_url": "http://127.0.0.1:8002/v1",
    },
))

print(response.data["command"])
print(response.data["env"])
```

## 生成的命令含义

```bash
python run.py --suite_family=android_world --agent_name=prm_api_agent --tasks=ContactsAddContact
```

- `python run.py`：使用 AndroidWorld 原生评测入口。
- `--suite_family=android_world`：运行 AndroidWorld 手机任务。
- `--agent_name=prm_api_agent`：预留给后续 PRM agent 的名称。
- `--tasks=...`：指定小规模任务子集；不传则运行完整 suite。

注意：AndroidWorld 原版 `run.py` 目前不认识 `prm_api_agent`。后续需要新增一个极薄
agent 接入层，或在独立 runner 中复用 AndroidWorld 的 `suite_utils.run()`。
