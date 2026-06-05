# LogicQA-Lite — 轻量化逻辑评判引擎

## 整体目的

LLM 产出 `<think>` + `<critique>` + `<score>` 后，logic_QA 充当**数据质检员**，检查这些产出是否存在**信息量不足**或**逻辑自相矛盾**的问题。只有通过审查的步骤才能进入训练集。

该模块是系统中的**质量守门员**：它不产生新数据，但通过逻辑门禁确保进入数据集的每一条轨迹都具备真实的教学价值。

---

## 目录结构与文件职责

```
logic_QA/
├── config/
│   └── qa_settings.yaml       # 阈值、权重及启用的评判器
├── evaluators/                # 评判策略 (策略模式，新增只需加文件)
│   ├── base.py                #   BaseEvaluator 抽象基类
│   ├── density.py             #   DensityEvaluator — 信息密度检查
│   └── conflict.py            #   ConflictEvaluator — 逻辑冲突检测
├── utils/
│   ├── text_processor.py      #   文本工具: 实体提取、关键词扫描、模板检测
│   └── aggregator.py          #   分值聚合: 加权 Meta-Score + is_accepted 裁决
├── engine.py                  #   QA_Orchestrator — Facade 调度器
└── main.py                    #   测试入口
```

---

## 输入与输出

**输入** (`evaluate(data)`):
```json
{
  "context":     { "task": "...", "step": 3 },
  "llm_output":  { "think": "基于执行前后的状态变化...", "critique": "...", "score": 0.85 },
  "state":       "执行后的 UI 状态描述文本",
  "prev_state":  "上一步的 UI 状态",
  "prev_score":  0.60
}
```

**输出** (`QualityReport`):
```json
{
  "is_accepted":   true,
  "meta_score":    0.92,
  "failed_reasons": [],
  "metrics": {
    "density_index": 0.80,
    "conflict_index": 0.00
  }
}
```

---

## 评判器详解

### 评判器一：DensityEvaluator — 信息密度检查

检测 LLM 的 `<think>` 是否在"敷衍"或缺乏具体信息。**三个子检查**：

#### ① 长度校验

```
阈值: min_think_length = 20 字符
```

| 情况 | 判定 |
|------|------|
| `<think>` < 20 字符 | **直接给 density_score = 0.0**，不进行后续检查 |
| `<think>` ≥ 20 字符 | 进入后续检查 |

**典型触发案例**:
```
think = "搜索结果第一项即为目标商家，直接进入。"  (19 字)
→ density_score=0.0, reason="Think length (19) below minimum (20)"
```

#### ② 实体对齐检查

```
步骤:
  1. 提取 think 中的文本实体 (方括号引用 [xxx] + 英文标识符)
  2. 提取 state_desc 中的 UI 元素关键词 (中文双字及以上词汇)
  3. 计算: alignment_score = min(|交集| / |UI实体|, 1.0)
```

| 情况 | 判定 |
|------|------|
| 交集 = 0 | 记录原因: "No UI entity references found in think" |
| 交集 > 0 | alignment_score 按比例计算 |

**典型触发案例**:
```
think = "这个动作合理有效，可以继续执行"  (完全没有引用具体 UI 元素)
state_desc 包含: 酒店预订, 携程酒店, 搜索栏, ...
→ 交集=0 → "No UI entity references found in think"
→ alignment_score 接近 0 (高分惩罚)
```

#### ③ 模板化检测

```
扫描 think 中是否包含以下一类通用废话:
  "点击以继续"  "进入下一页"  "等待加载完成"  "如图所示"
  "根据页面提示"  "继续执行"  "操作成功"  "如上所述"
  "显而易见"  "毫无疑问"  "click to continue"  ...

每命中 1 个 → 降 0.2 分
最多降 0.7 分 (下限 protection)
template_score = max(0.3, 1.0 - 0.2 × 命中数)
```

#### DensityEvaluator 综合评分

```
density_score = alignment_score × 0.5 + template_score × 0.5
```

**完整实例**:
```
think = "根据页面提示，点击以继续进入下一页。该动作逻辑正确。"  (60字 ≥ 20 ✓)
state_desc 中 UI 实体: {携程酒店, 酒店预订, 搜索, ...}
think 实体: {页面提示, 继续, 下一页} → 交集 = ∅
模板命中: "根据页面提示" ✓, "点击以继续" ✓, "进入下一页" ✓ → 3 个
→ alignment_score = 0.0
→ template_penalty = min(3×0.2, 0.7) = 0.6 → template_score = 0.4
→ density_score = 0.0×0.5 + 0.4×0.5 = 0.2
→ FAIL: "No UI entity references found in think; Detected 3 template phrase(s)"
```

---

### 评判器二：ConflictEvaluator — 逻辑冲突检测

检测 LLM 的评分和文字解释是否**自相矛盾**。**三个子检查**：

#### ① 进展一致性校验

```
δ = Q_t - Q_{t-1}

如果 δ < conflict_delta_limit (-0.2, Q 值骤降超过 0.2)
  且 critique 中不包含任何负面关键词
→ 冲突: 分数大跌却不承认有问题
→ 扣 0.4 分
```

**负面关键词列表**:
```
"错误" "失败" "无效" "不正确" "误导" "冗余" "多余"
"不应该" "没必要" "循环" "死循环" "卡住" "重复"
"wrong" "failed" "invalid" "incorrect" "misleading"
"redundant" "unnecessary" "loop" "stuck" "repeated"
```

**典型触发案例**:
```
Step 1: Q=1.00, Step 2: Q=0.10  →  δ = -0.90 < -0.2
critique = "该动作已经成功导航至会员权益页面，完全符合预期..."
→ 负面关键词数 = 0
→ 冲突! Q 值从 1.0 暴跌到 0.1 却说"完全符合预期"
→ 扣 0.4 分, reason: "Score dropped by -0.900 but critique lacks negative feedback"
```

#### ② 语义-分值对齐

```
如果 critique 中包含负面关键词 (>0 个)
  且 score ≥ semantic_score_threshold (0.8)
→ 冲突: 嘴上批评但给了高分
→ 扣 0.5 分
```

**典型触发案例**:
```
critique = "该动作是完全冗余的...严重偏离子目标...浪费了关键步骤机会"
score = 0.85 ≥ 0.8
→ 负面关键词: "冗余" ✓, "偏离" (NOT in list) → 1 个
→ 冲突! 骂着骂着给了0.85分
→ 扣 0.5 分
```

#### ③ 状态变化监测

```
如果 prev_state ≠ None 且 curr_state ≠ None
  且 has_state_changed(prev, curr) = False (状态未变化)
  且 score > 0.5
→ 逻辑幻觉: 什么都没变却认为取得了进展
→ 扣 0.3 分
```

**实现**: `has_state_changed()` 优先比较 `to_state` 哈希，或比较 UI 元素 ID 列表是否相同。

#### ConflictEvaluator 综合评分

```
penalty = 子检查①扣分 + 子检查②扣分 + 子检查③扣分
conflict_score = max(0.0, 1.0 - penalty)
```

---

## 聚合器 — Meta-Score 计算

### 加权聚合

```
Meta-Score = (density_score × w_density + conflict_score × w_conflict) / (w_density + w_conflict)
```

其中 `w_density = 0.4`, `w_conflict = 0.6`。

### 裁决逻辑

```
is_accepted = (Meta-Score ≥ accept_threshold) AND (每个子评判器 score ≥ accept_threshold)
```

其中 `accept_threshold = 0.5`。

**规则**: 即使 Meta-Score 高于阈值，如果某个子评判器单独得分 < 0.5，仍标记为不通过。

### QualityReport 结构

```json
{
  "is_accepted": true,
  "meta_score": 0.7000,
  "failed_reasons": [],
  "metrics": {
    "density_index": 0.50,
    "conflict_index": 1.00
  }
}
```

---

## traj_002 完整 QA 运行结果

最近一次流水线运行, 8 步全部通过审查:

```
Step  density   conflict   Meta-Score  is_accepted
  1    0.500     1.000      0.700      V
  2    0.500     1.000      0.700      V
  3    0.500     1.000      0.700      V
  4    0.500     1.000      0.700      V
  5    0.500     1.000      0.700      V
  6    0.500     1.000      0.700      V
  7    0.500     1.000      0.700      V
  8    0.500     1.000      0.700      V
```

density 得分中规中矩 (实体对齐部分匹配), conflict 全部满分 (未检测到评分与文本的明显矛盾)。这说明当前 LLM 产出的 think/critique/score 之间**逻辑自洽**。

---

## 与 core_prm 的协同关系

```
LLM_caller 产出 Q 值       →  core_prm 用 GAE 修正        →  得到稳健的步骤价值
LLM_caller 产出 think/crit  →  logic_QA 校验一致性        →  标记未被通过的步骤
                                      ↓
                             只有 QA 通过的步骤才进入训练集
                             core_prm 的 GAE 值用于样本权重分配
```

**典型决策案例**:
```
Step 2 (traj_002): Q=0.10, GAE=+0.545, QA=V
→ 单步 Q 低但全局 GAE 可接受, QA 未发现逻辑矛盾
→ 决策: 保留, 低权重

Step 7 (traj_002): Q=0.20, GAE=+0.931, QA=V
→ 单步倒数但整条路径最终成功
→ 决策: 保留, 中等权重

假设某步 QA=X:
→ 直接排除, 不进训练集 — 避免 LLM 幻觉评分污染数据
```

---

## 与流水线重试机制的集成

logic_QA 的 `is_accepted` 在 `run_pipeline.py` 中被用作**重试触发信号**:

```
LLM_caller 全量调用 → logic_QA 审查 → 发现失败步骤
       ↑                                    ↓
       └─── 仅重试失败步骤 (--indices) ←── (最多 3 次)
```

- QA 通过的步骤 → 保留首次 LLM 结果
- QA 未通过的步骤 → 单独重试该步 LLM + 重新审查 (最多 3 次)
- 3 次后仍不通过 → 使用最后一次结果

**注意**: `is_accepted` 仅用于内部重试决策，**不参与 filter 模块的最终筛选**。筛选由 `filter/` 模块的 `step_pass_ratio` 和 `composite_completion` 独立完成。

---

## 设计原则

| 原则 | 贯彻方式 |
|------|---------|
| **开闭原则 (OCP)** | 新增评判维度只需在 `evaluators/` 加子类并注册 config, 不改 engine |
| **单一职责 (SRP)** | 每个 Evaluator 只负责一个维度, 互不交叉 |
| **策略模式** | Density 和 Conflict 是两个独立策略, 可动态组合/开关 |

## 扩展指南

| 需求 | 操作 |
|------|------|
| 新增评判维度 | `evaluators/` 下继承 `BaseEvaluator`, 在 config 注册名称和权重 |
| 调整阈值 | 修改 `config/qa_settings.yaml` |
| 调整权重 | 修改 `config/qa_settings.yaml` 的 `weights` |

## 配置参考

`config/qa_settings.yaml`:
```yaml
enabled_evaluators:
  - "DensityEvaluator"
  - "ConflictEvaluator"
thresholds:
  min_think_length: 20
  conflict_delta_limit: -0.2
  semantic_score_threshold: 0.8
weights:
  density: 0.4
  conflict: 0.6
accept_threshold: 0.5
```
