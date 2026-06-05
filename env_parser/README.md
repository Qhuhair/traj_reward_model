# Env Parser — 环境解析器 (Phase 1)

将 `trajs/` 目录中的原始 GUI 轨迹数据解析为标准化的 State-Action 对，供下游模块 (LLM_caller, core_prm, logic_QA) 消费。

## 设计原则

- **不修改原始数据**：只读取 `trajs/`，输出到 `output/` 或内存，永不写回原始文件
- **开闭原则**：StateBuilder 采用策略模式，新增描述策略只需添加子类
- **高内聚低耦合**：loader (文件IO) / builder (文本转换) / formatter (输出格式) / pipeline (编排) 各司其职
- **Unix 哲学**：每个 util 函数是纯函数，一个函数只做一件事

## 目录结构

```
env_parser/
├── config/
│   └── parser_config.yaml        # 可调参数 (见下方说明)
├── core/
│   ├── loader.py                 # TrajectoryLoader: 读取 metadata + history + perception
│   ├── state_builder.py          # StateBuilder (策略): perception → 自然语言 state_desc
│   ├── formatter.py              # TrajectoryFormatter: 输出标准化 S-A 对
│   └── pipeline.py               # EnvParserPipeline: Facade 全流程调度
├── utils/
│   ├── perception_utils.py       # 纯函数: 清洗/聚类/格式化感知元素
│   └── path_utils.py             # 纯函数: 文件发现/路径解析
├── main.py                       # 测试入口
└── README.md
```

## 配置参数 (`parser_config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_perception_elements` | 30 | 每个 state_desc 最多引用的 UI 元素数 |
| `cluster_by_area` | true | 是否按屏幕区域 (top/mid/bottom) 分组排列元素 |
| `area_bins` | 3 | 区域分组数 (cluster_by_area=true 时生效) |
| `include_coordinates` | false | state_desc 是否包含像素坐标 |
| `strip_prefix` | true | 是否去除 perception 中的 "text:" / "icon:" 前缀 |
| `output_dir` | "output" | 批量处理时的输出目录名 |

## 使用方式

```bash
python env_parser/main.py             # 处理所有轨迹集
python env_parser/main.py traj_007    # 处理指定轨迹
```

```python
from core.pipeline import EnvParserPipeline
pipeline = EnvParserPipeline()
output = pipeline.run("trajs/20250113_21442_test/traj_007")
```

## 输出格式 (标准化的 S-A 对)

```json
{
  "trajectory_id": "traj_007",
  "app": "携程旅行",
  "task": "Navigate from the main homepage...",
  "steps": [{
    "step_idx": 1,
    "action": "Tap (538, 2324)",
    "action_desc": "On the main screen, tap...",
    "element_id": "elem_01c06911096c30a9",
    "state_desc_before": "首页 UI 元素...",
    "state_desc_after": "酒店搜索页 UI 元素...",
    "image_before": "/abs/path/to/00_start.jpg",
    "image_after": "/abs/path/to/01_Tap_538_2324_.jpg",
    "subgoal_text": "Navigate to hotel search...",
    "is_backtrack": false
  }]
}
```

> 注意: 每个 step JSON (`NN_action.json`) 的 perception 是该步骤**执行后**的屏幕。因此 Step i 的 `state_desc_before` = step_file[i] 的 perception, `state_desc_after` = step_file[i+1] 的 perception。

## 扩展指南

| 需求 | 如何实现 |
|------|---------|
| 新增 state 描述策略 | 继承 `BaseStateBuilder`，在 `create_state_builder()` 注册 |
| 支持新数据源格式 | 继承/扩展 `TrajectoryLoader` |
| 新增输出格式 | 在 `TrajectoryFormatter` 或新增子类 |
| 接入 Vision 模型 | loader 已返回 `image_before` / `image_after` 路径 |

## 下游集成

```python
# LLM_caller 可直接消费
for step in output["steps"]:
    result = caller.call(
        task_desc=output["task"],
        state_desc_before=step["state_desc_before"],
        state_desc_after=step["state_desc_after"],
        action_desc=step["action_desc"],
    )
```
