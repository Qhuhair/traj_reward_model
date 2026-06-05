# Filter — 轨迹筛选模块

根据流水线生成的评分报告，将轨迹按**任务完成度**和**步骤逻辑合理性**分为四类。

## 目录结构

```
filter/
├── config/
│   └── filter_config.yaml       # 筛选规则: 阈值、分类条件
├── core/
│   ├── __init__.py
│   ├── loader.py                # 数据加载: 读取 summary.json → TrajData
│   ├── scorer.py                # 指标计算: 15 项衍生指标
│   ├── classifier.py            # 规则引擎: 按 YAML 条件逐条匹配 → 分类
│   └── filter_engine.py         # FilterEngine: Facade 调度器
├── utils/
│   ├── __init__.py
│   └── conditions.py            # 纯函数: 条件求值 (ge/gt/le/lt/eq/ne)
├── main.py                      # 测试入口
└── README.md
```

### `core/loader.py` — 数据加载

`load_run_dir(run_dir)` 扫描 `output/<set>/traj_*/summary.json`，构建 `TrajData` dataclass (含每步 `StepDetail`)。

### `core/classifier.py` — 规则引擎

`RuleClassifier.classify(metrics)` 按 config 中 `category_rules` 的顺序依次匹配：
- 每条规则含 `all_of` (AND) 或 `any_of` (OR) 条件列表
- 第一命中即返回该分类
- 规则全在 YAML 中定义，不命中时返回 "未分类"

### `utils/conditions.py` — 条件求值

支持 6 种操作符: `ge`(≥), `gt`(>), `le`(≤), `lt`(<), `eq`(==), `ne`(≠)。浮点比较含 `EPSILON=1e-7` 容差。

---

## 筛选规则

### 两条核心指标

| 指标 | 计算方式 | 阈值 | 说明 |
|------|---------|------|------|
| `composite_completion` | `avg_q × 0.4 + norm_avg_gae × 0.6` | ≥ 0.5 | 综合 LLM 评分 (40%) + core_prm 时序价值 (60%)，判断是否完成目标 |
| `step_pass_ratio` | `score ≥ step_pass_threshold 的步骤数 / 总步骤数` | 见下表 | 单步 Q 值 ≥ 阈值视为"该步骤逻辑合理" |

### `composite_completion` 详解

```
avg_q      = mean(Q₁, Q₂, ..., Qₙ)          # LLM 对每步的主观评分均值
norm_gae   = (avg_gae - gae_min) / (gae_max - gae_min)   # GAE 归一化到 [0, 1]
composite  = avg_q × 0.4 + norm_gae × 0.6    # LLM 40% + core_prm 60%
```

### 四类定义 (基于实际 config)

```
               step_pass_ratio ≥ 阈值          step_pass_ratio < 阈值
composite     ┌────────────────────────┬────────────────────────┐
≥ 0.5          │  A: 完成目标_逻辑合理    │  B: 完成目标_逻辑可疑    │
(完成目标)      │    ≥ 90% 步骤 Q ≥ 0.7   │    < 90% 步骤 Q ≥ 0.7  │
composite     └────────────────────────┼────────────────────────┤
< 0.5          │  C: 未完成_局部合理      │  D: 未完成_逻辑异常      │
(未完成)        │    ≥ 85% 步骤 Q ≥ 0.7   │    < 85% 步骤 Q ≥ 0.7  │
               └────────────────────────┴────────────────────────┘
```

> 注意: 实际阈值在 `filter/config/filter_config.yaml` 中配置，A/B 用 `0.90`, C/D 用 `0.85`。`step_pass_threshold` 为 `0.7`。

| 类别 | 含义 | 典型处理 |
|------|------|---------|
| **A** — 完成目标_逻辑合理 | composite ≥ 0.5 且 ≥90% 步骤 Q ≥ 0.7 | 直接进入蒸馏训练集 |
| **B** — 完成目标_逻辑可疑 | composite ≥ 0.5 但 <90% 步骤 Q ≥ 0.7 | LLM+PRM 觉得完成了，但争议步骤多 |
| **C** — 未完成_局部合理 | composite < 0.5 但 ≥85% 步骤 Q ≥ 0.7 | 目标未达成但步骤本身无明显问题 |
| **D** — 未完成_逻辑异常 | composite < 0.5 且 <85% 步骤 Q ≥ 0.7 | 双低，建议丢弃 |

---

## QA 模块的角色变化

| 旧设计 | 新设计 |
|--------|--------|
| QA 的 `is_accepted` 直接参与筛选 | QA 仅作为**内部审查依据**，不参与分类 |
| 步骤级质量 = QA 评判 | 步骤级质量 = LLM 评分 ≥ 0.7 |
| QA 失败 = 步骤作废 | QA 失败 → 触发 LLM **重试** (最多 3 次) |

---

## 使用方式

```bash
python filter/main.py                  # 自动选最新 output 目录
python filter/main.py 20250113_21442_test  # 指定 output 子目录
```

```python
from filter.core.filter_engine import FilterEngine
report = FilterEngine().run("output/20250113_21442_test")
```

---

## 扩展指南

| 需求 | 操作 |
|------|------|
| 调整阈值 | 修改 `config/filter_config.yaml` 的 value 值 |
| 新增分类类别 | 在 yaml 加一条 `category_rules` |
| 新增评判指标 | 在 `core/scorer.py` 的 `compute_all()` 加一个 key |
| 新增条件操作符 | 在 `utils/conditions.py` 的 `OPERATORS` 字典加一种 |
