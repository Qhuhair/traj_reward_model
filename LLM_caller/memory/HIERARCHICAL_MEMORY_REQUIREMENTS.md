# 分层记忆模块需求文档

## 目标

为 GUI 轨迹评分引入分层记忆机制，使模型在评分当前步骤时同时获得历史压缩信息和最近细节信息。

- 历史步骤信息：距离当前步骤超过 3 步的旧步骤，用摘要形式压缩。
- 最近步骤信息：最近窗口大小为 3 的步骤，保留细致文本描述；当前步继续保留 before/after 图片。
- 模型输入：`历史步骤摘要 + 最近步骤信息 + 当前步骤文本 + 当前步骤图片`。
- 使用方式：记忆模块必须通过参数显式开启或关闭，关闭时保持现有流水线行为不变。

核心目标是减少长轨迹上下文丢失，同时避免把全部历史截图和长文本直接输入模型。

## 设计原则

- 单一职责：摘要、窗口选择、Prompt 构建、模型调用分离。
- 开闭原则：新增分层记忆模块，不破坏现有 `full`、`sliding`、`gs` 流程。
- 可替换策略：历史摘要由大模型生成，摘要器接口保留可替换能力，便于后续切换模型或规则回退。
- 可复现：每一步实际使用的历史摘要和最近窗口必须落盘。
- 成本可控：历史摘要由大模型生成，但应按步骤增量复用，避免对同一历史片段重复摘要。
- 默认安全：除新建的分层记忆流水线外，现有入口默认不启用记忆模块，避免改变历史实验口径。

## 推荐目录结构

```text
LLM_caller/
├── memory/
│   ├── __init__.py
│   ├── schema.py              # StepMemory、HistorySummary 等结构定义
│   ├── history_summarizer.py  # 大模型历史步骤摘要策略
│   ├── recent_window.py       # 最近窗口选择逻辑
│   ├── memory_builder.py      # 汇总历史摘要 + 最近窗口
│   └── prompt_context.py      # 转换为 Prompt 占位符
├── prompts/
│   └── RRM_Qwen_HierMemory.yaml
└── utils/
    └── hierarchical_memory_evaluator.py
```

新增流水线入口：

```text
pipelines/
└── hierarchical_memory.py
```

建议输出目录：

```text
output/<experiment>/<set>/<traj>/
├── standardized.json
├── memory_contexts.json
├── llm_scores.json
├── qa_reports.json
├── prm_scores.json
└── summary.json
```

## 核心数据流

记忆模块开启时，对第 `i` 步评分：

```text
standardized.json
    ↓
MemoryBuilder
    ├── history = steps[0 : i - 3]
    ├── recent = steps[max(0, i - 3) : i + 1]
    ↓
history_summarizer
    ├── 调用 LLM 生成历史摘要
    ├── 缓存或复用相同历史范围摘要
    ↓
prompt_context
    ↓
LLMCaller(model=..., prompt=RRM_Qwen_HierMemory)
    ↓
llm_scores.json
```

记忆模块关闭时，流程必须退化为原有模式：

```text
standardized.json
    ↓
原有 evaluator / prompt_context
    ↓
LLMCaller(model=..., prompt=...)
    ↓
llm_scores.json
```

规则：

- 当前步必须包含 before/after 图片。
- 最近窗口 3 步包含当前步和前 2 步，不包含当前步之后的任何步骤。
- 历史摘要包含第 1 步到第 `i-3` 步。
- 当 `i <= 3` 时，历史摘要为空，只使用最近窗口。
- 禁止引入当前步之后的信息，避免未来信息泄漏。
- `--use-memory` 未开启时，不构建历史摘要，不写入 `memory_contexts.json`，不改变 Prompt 占位符。
- `--use-memory` 开启时，必须写入 `memory_contexts.json`，便于复现每一步模型输入。
- 当前步骤输入为图片 + 文本描述；窗口内其他步骤只使用原始文本描述，不传图片。
- 历史摘要只保留文本摘要概括，不包含图片、不包含未来步骤、不包含当前步骤之后的信息。

## Prompt 输入结构

建议 Prompt 占位符：

```text
【任务】{task_desc}

【当前子目标】{curr_subgoal}

【历史步骤摘要】
{history_summary}

【最近步骤信息】
{recent_steps_detail}

【当前步骤】
步骤 {step_idx}
动作：{curr_action}
元素：{curr_element_id}
执行前状态：{curr_state_before}
执行后状态：{curr_state_after}

请结合：
1. 历史步骤是否已经完成前置目标；
2. 最近 3 步中当前及之前步骤是否形成合理连续路径；
3. 当前 before/after 图片中的 UI 变化；
4. 当前动作是否推进当前子目标。

输出：
<thinking>...</thinking>
<critique>...</critique>
<score>0.0~1.0</score>
```

## 历史摘要生成

历史摘要必须由大模型生成，输入为历史步骤的原始文本描述、子目标、动作、状态摘要和已有评分信息。输出为纯文本摘要，供后续评分 Prompt 使用。

摘要生成要求：

- 只总结历史范围 `steps[0 : i - 3]`。
- 不读取当前步骤之后的信息。
- 不读取历史步骤图片，历史摘要只基于文本。
- 摘要应保留已完成目标、当前状态、关键错误、回退原因、低分或冲突步骤。
- 摘要长度受 `history_summary_max_chars` 控制。
- 同一历史范围摘要可缓存到 `memory_contexts.json` 或中间缓存，避免重复调用。

历史摘要示例：

```text
步骤1-5摘要：
- 已完成：进入社区页；返回首页；进入会员福利、会员商城、会员中心。
- 当前状态：已从会员中心回退到主入口。
- 关键问题：无明显错误；步骤6-8为合理回退。
- 最近历史评分趋势：高分为主，存在弱推进步骤。
```

摘要输入字段优先级：

1. `action_desc`
2. `subgoal_text`
3. `to_node_desc`
4. 已有 `llm_scores.json` 中的 `score` / `critique`，如果该步骤已经评分
5. `is_backtrack`

建议新增摘要 Prompt：

```text
【任务】{task_desc}
【需要摘要的历史步骤】
{history_steps_text}

请生成面向 GUI 轨迹评分的历史摘要，要求：
1. 只概括给定历史步骤，不推测未来。
2. 保留已完成子目标、当前页面状态、关键错误和合理回退。
3. 输出不超过 {history_summary_max_chars} 字。
4. 只输出摘要文本。
```

## 最近窗口格式

最近窗口保留步骤级文本细节，但控制长度。窗口范围为当前步骤和之前 2 步，不允许包含当前步骤之后的内容：

```text
最近步骤：
- 步骤8：Back
  子目标：浏览北京奢华酒店排行
  动作：从会员福利页返回携程主入口
  结果：回到主入口，可继续访问酒店预订

- 步骤9：Tap 酒店预订
  执行前：携程首页
  执行后：酒店预订页，显示北京、日期、查询按钮

- 步骤10：Tap 北京奢华酒店排行榜
  执行前：酒店预订页
  执行后：北京奢华酒店榜
```

图片策略：

- 当前步：必须传 before/after 图片。
- 最近前 2 步：只传原始文本描述，不传图片。
- 不提供当前步骤之后的 next action、next subgoal 或未来状态。
- 不再配置 `recent_image_steps`，避免窗口内图片过多和未来信息误用。

## 配置建议

```yaml
hierarchical_memory:
  enabled: false
  recent_window_size: 3
  history_summary_max_chars: 1200
  recent_step_max_chars: 600
  include_previous_scores: true
  include_current_images: true
  summarizer: "llm"
  summarizer_model: "codex_mm_baseline"
  summarizer_prompt: "RRM_HistorySummary"
```

参数约定：

- `enabled: false`：默认关闭分层记忆，保持现有流水线行为。
- `--use-memory`：命令行显式开启分层记忆，覆盖配置中的 `enabled: false`。
- `--no-memory`：命令行显式关闭分层记忆，优先级高于配置文件。
- `--memory-window-size 3`：控制最近窗口大小，默认 3。
- `--memory-summary-max-chars 1200`：控制历史摘要最大长度。
- `--memory-summarizer llm`：选择大模型摘要策略，默认 `llm`。
- `--memory-summarizer-model codex_mm_baseline`：指定生成历史摘要的大模型配置。
- `--memory-summarizer-prompt RRM_HistorySummary`：指定历史摘要 Prompt。

## 流水线命令目标

后续实现后，全量命令应类似：

```bash
python -m pipelines.hierarchical_memory --output-name crossapp_codex_hier_memory_all --llm-model codex_mm_baseline --llm-prompt RRM_Qwen_HierMemory --use-memory --quiet
```

命令说明：

- `python -m pipelines.hierarchical_memory`：运行分层记忆流水线。
- `--output-name crossapp_codex_hier_memory_all`：结果写入 `output/crossapp_codex_hier_memory_all/`。
- `--llm-model codex_mm_baseline`：使用 Codex Responses 多模态基准模型。
- `--llm-prompt RRM_Qwen_HierMemory`：使用分层记忆 Prompt。
- `--use-memory`：显式开启分层记忆模块。
- `--quiet`：日志写入输出目录下的 `pipeline.log`。

关闭记忆、复用同一入口验证对照组：

```bash
python -m pipelines.hierarchical_memory --output-name crossapp_codex_no_memory_all --llm-model codex_mm_baseline --llm-prompt RRM_Qwen_MM --no-memory --quiet
```

命令说明：

- `--no-memory`：显式关闭分层记忆，模型只接收原有当前步骤输入。
- `--llm-prompt RRM_Qwen_MM`：关闭记忆时使用普通多模态 Prompt，避免 Prompt 占位符缺失。

## 验收标准

- `memory_contexts.json` 每个步骤都有历史摘要和最近窗口。
- 第 1-3 步历史摘要为空或很短。
- 第 4 步开始历史摘要只包含更早步骤，不重复最近窗口。
- 当前步骤图片仍然传入模型。
- 最近窗口内除当前步骤外，其他步骤只能包含原始文本描述，不得传图片。
- 任何步骤的 Prompt 和 `memory_contexts.json` 都不得包含当前步骤之后的信息。
- 历史摘要必须由配置指定的大模型生成，并在 `memory_contexts.json` 中保存摘要文本和摘要输入范围。
- `--no-memory` 运行时不得生成 `memory_contexts.json`，输出结构应与普通多模态流程一致。
- `--use-memory` 与 `--no-memory` 的输出目录可以并行存在，便于做消融实验。
- `llm_scores.json`、`qa_reports.json`、`prm_scores.json`、`summary.json` 正常生成。
- 能与 Codex 多模态单步基准对比 MAE、平均偏差、分类一致率。

## 风险点

- 历史摘要可能引入错误归纳，影响后续评分。
- 最近窗口如果包含当前步之后的信息，会造成未来信息泄漏。
- 图片过多会增加成本和延迟，默认只给当前步图片。
- 如果历史摘要依赖前面 LLM 评分，早期错误可能被传播，需要在摘要中保留“不确定、低分、冲突”信息。
