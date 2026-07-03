# Agent-RRM 轨迹评估系统

基于过程奖励模型 (PRM) 和相对奖励模型 (RRM) 的 GUI 轨迹质量评估流水线。对手机 App GUI 操作轨迹逐步骤分析，产出多维评分及质量审查报告，为 4B 小模型蒸馏训练筛选高质量数据。

---

## 全流水线

```
trajs/<set>/traj_XXX/             原始轨迹集 (metadata + history + per-step perception + 截图)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1  env_parser         环境解析器                          │
│            perception[]  →  state_desc_before + state_desc_after │
│            metadata + history  →  标准化 S-A 对                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ standardized.json
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2  LLM_caller         LLM 评分生成器                      │
│            state_before + action + element + state_after         │
│            →  <think> / <critique> / <score>                     │
│            支持并发 (4 workers) + 多适配器 (DeepSeek/vLLM/Ollama) │
└──────────────────────────────┬───────────────────────────────────┘
                               │ llm_scores.json
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  QA + Retry                  logic_QA 审查 + 重试                │
│            DensityEvaluator + ConflictEvaluator                  │
│            QA 未通过 → 重试该步 LLM (最多 3 次)                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ reviewed llm_scores.json
                    ┌──────────┴──────────┐
                    ▼                     ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│  Stage 3  core_prm       │  │  report                            │
│  过程奖励计算引擎        │  │  报告生成                          │
│                          │  │                                    │
│  Progress / TD-Error (δ) │  │  summary.json (单轨迹)             │
│  GAE 平滑                │  │  master_summary.json (总报告)      │
└──────────────────────────┘  └────────────────────┬───────────────┘
                               prm_scores.json     │
                                                   ▼
                                          ┌────────────────────────┐
                                          │  filter                │
                                          │  轨迹筛选              │
                                          │                        │
                                          │  composite_completion  │
                                          │  step_pass_ratio       │
                                          │  → A/B/C/D 四类        │
                                          └────────────────────────┘
                                                   │
                                          filtered.json
```

---

## 目录结构

```
.
├── trajs/                                    # 原始轨迹数据 (不修改)
│   ├── 20250113_21442_test/                  #   轨迹集 1 (4 条)
│   ├── 20260113_214142_subgoal/              #   轨迹集 2 (10 条)
│   └── ...
│
├── models/                                   # 本地模型服务辅助模块
│   ├── serve.py                              #   vLLM 启动兼容入口
│   ├── test_model.py                         #   OpenAI 兼容接口测试
│   └── vllm/                                 #   vLLM 服务启动与日志
│
├── pipelines/                                # 流水线入口模块
│   ├── full.py                               #   完整流水线
│   ├── multimodal.py                         #   多模态单步
│   ├── sliding.py                            #   多模态滑动窗口
│   ├── text_window.py                        #   纯文本滑动窗口
│   ├── image_only.py                         #   纯图片模式
│   └── gs.py                                 #   GUI-Shepherd 两步评估
│
├── framework_api/                            # 对外 API 抽象层
│   ├── schema.py                             #   请求/响应数据结构
│   ├── interfaces.py                         #   Handler / Service 协议
│   ├── registry.py                           #   能力注册表
│   └── service.py                            #   最小服务分发层
│
├── env_parser/                    # Stage 1 — 环境解析器
│   ├── config/parser_config.yaml
│   ├── core/ (loader, state_builder, formatter, pipeline)
│   └── utils/ (perception_utils, path_utils)
│
├── LLM_caller/                    # Stage 2 — LLM 评分生成器
│   ├── config.yaml                #   模型配置 (DeepSeek / Qwen vLLM / Ollama 兼容)
│   ├── caller.py, utils.py
│   ├── adapters/                  #   适配器 (OpenAI / Qwen vLLM / Ollama 兼容)
│   ├── prompts/                   #   Prompt 模板 (RRM_V1 / RRM_Qwen / RRM_Qwen_MM)
│   └── utils/
│       ├── image_annotator.py     #   截图标注工具 (红叉+绿框)
│       └── annotate_traj_images.py #   批量标注脚本
│
├── logic_QA/                      # QA 审查 + 重试
│   ├── config/qa_settings.yaml
│   ├── evaluators/ (density, conflict)
│   ├── utils/ (aggregator, text_processor)
│   └── engine.py
│
├── core_prm/                      # Stage 3 — 过程奖励计算引擎
│   ├── config/prm_params.yaml
│   ├── data_adapters/
│   ├── estimators/ (progress_calc, td_gae_engine)
│   └── prm_orchestrator.py
│
├── report/                        # 报告生成模块
│   ├── loader.py, builder.py, writer.py
│
├── filter/                        # 轨迹筛选模块
│   ├── config/filter_config.yaml
│   ├── core/ (loader, scorer, classifier, filter_engine)
│   └── utils/ (conditions)
│
├── run_pipeline.py                # 兼容入口 → pipelines.full
├── run_multimodal_pipeline.py     # 兼容入口 → pipelines.multimodal
├── run_sliding_pipeline.py        # 兼容入口 → pipelines.sliding
└── README.md

---

## 输出目录

```
output/                              # 各种模式输出
├── deepseek/                        # DeepSeek 结果
├── qwen3.5-4b/                      # Qwen3.5-4B 纯文本单步
├── qwen-text-window/                # Qwen3.5-4B 纯文本滑动窗口
├── qwen3.5-4b-image-text/          # Qwen3.5-4B 多模态单步
└── qwen3.5-4b-image-text-window/   # Qwen3.5-4B 多模态滑动窗口
```

---

## 快速开始

```bash
pip install pyyaml requests

# 处理指定轨迹集
python -m pipelines.full 20250113_21442_test

# 指定输出目录名
python -m pipelines.full 20250113_21442_test --output-name crossapp_qwen35_4b_test

# 纯文本输入并静默运行
python -m pipelines.full 20250113_21442_test --output-name crossapp_qwen35_4b_test --quiet

# 处理所有轨迹集
python -m pipelines.full

# 处理所有轨迹集，并指定统一实验输出目录
python -m pipelines.full --output-name crossapp_qwen35_4b_all
```

根目录仍保留兼容入口，旧命令 `python run_pipeline.py <set_name>` 可以继续使用。
指定单个轨迹集时，`--output-name` 会将结果写入 `output/<自定义名称>/`；
处理全部轨迹集时，`--output-name` 会作为总实验目录，内部按轨迹集名分子目录保存。
`pipelines.full` 默认使用纯文本输入：`--input-mode text --llm-model qwen_vllm_text --llm-prompt RRM_Qwen`。
该默认配置要求 vLLM OpenAI 兼容服务已监听 `http://localhost:8002/v1`。
`--quiet` 会把控制台日志追加写入输出目录下的 `pipeline.log`。

### 切换 LLM 模型

在 `LLM_caller/config.yaml` 中改 `active_model`:

```yaml
active_model: "deepseek"        # 云端 DeepSeek (质量最高, 需要 API key)
active_model: "qwen_vllm_text"  # 本地 vLLM Qwen3.5-4B 纯文本 OpenAI 兼容接口
active_model: "qwen_vllm_mm"    # 本地 vLLM 多模态 OpenAI 兼容接口（需模型支持图片）
active_model: "qwen_local"      # 历史兼容：本地 Ollama 纯文本
active_model: "qwen_local_mm"   # 历史兼容：本地 Ollama 多模态
active_model: "qwen_remote"     # 远程服务器 Qwen3.5-4B (纯文本, HTTP API)
```

多模态模式入口已整理到 `pipelines/` 模块，例如：
`python -m pipelines.multimodal <set_name>`。兼容旧入口
`python run_multimodal_pipeline.py <set_name>` 仍可使用。

### 不同模式运行命令

以下命令默认处理 `trajs/` 下全部轨迹集。传入轨迹集名时只处理该集合，例如
`python -m pipelines.full 20260113_214142_subgoal ...`。本地 Qwen 模式统一使用
vLLM OpenAI 兼容接口，服务地址为 `http://localhost:8002/v1`。

| 入口 | 用途 |
|------|------|
| `python -m pipelines.full --output-name crossapp_qwen35_4b_vllm_text --input-mode text --llm-model qwen_vllm_text --llm-prompt RRM_Qwen --quiet` | 纯文本单步；输出到 `output/crossapp_qwen35_4b_vllm_text/` |
| `python -m pipelines.full --output-name crossapp_qwen35_4b_multimodal_step --input-mode multimodal --llm-model qwen_vllm_mm --llm-prompt RRM_Qwen_MM --quiet` | 多模态单步；输入文本 + 前后截图；需 vLLM 模型支持图片 |
| `python -m pipelines.text_window --output-name crossapp_qwen35_4b_text_window --quiet` | 纯文本滑动窗口；3 步上下文 |
| `python -m pipelines.sliding --output-name crossapp_qwen35_4b_multimodal_window --quiet` | 多模态滑动窗口；含截图、窗口上下文和 QA 重试 |
| `python -m pipelines.gs --output-name crossapp_qwen35_4b_gs --quiet` | GUI-Shepherd 两步；先视觉匹配，再文本评分 |
| `python -m pipelines.image_only --output-name crossapp_qwen35_4b_image_only --quiet` | 纯图片模式；已废弃，仅保留兼容入口 |

`--output-name` 指定 `output/` 下的实验目录；`--quiet` 将日志写入该目录下的
`pipeline.log`。

### 输出

每次运行在 `output/` 下生成以**轨迹集名**命名的目录：

```text
output/20250113_21442_test/
├── master_summary.json            # 总报告
├── filtered.json                  # 筛选报告 (A/B/C/D 四分类)
├── traj_007/
│   ├── standardized.json          # env_parser: 标准化 S-A 对 (含 before/after 状态)
│   ├── llm_scores.json            # LLM_caller: think/critique/score (经QA审查后)
│   ├── LLMscore.json              # llm_scores.json 的兼容副本
│   ├── qa_reports.json            # logic_QA: 每步 Meta-Score/is_accepted
│   ├── prm_scores.json            # core_prm: Progress/TD-Error/GAE
│   └── summary.json               # report:   该轨迹汇总
└── traj_009/
    └── ...
```

---

## 各模块详解

### Stage 1 — `env_parser/` 环境解析器

将原始 `trajs/<set>/traj_XXX/` 中的多文件轨迹数据解析为标准化的 State-Action 对。

**关键修正**：每个 step JSON (`NN_action.json`) 的 perception 是该步骤**执行后**的屏幕。因此：
- Step i 的 `state_desc_before` = step_file[i] 的 perception (上一步的 after 屏)
- Step i 的 `state_desc_after` = step_file[i+1] 的 perception

**标准化输出** (standardized.json):
```json
{
  "trajectory_id": "traj_007",
  "app": "携程旅行",
  "task": "Navigate from the main interface...",
  "steps": [{
    "step_idx": 1,
    "action": "Tap (538, 2324)",
    "action_desc": "On the main screen, tap...",
    "element_id": "elem_01c06911096c30a9",
    "state_desc_before": "首页 UI 元素...",
    "state_desc_after": "酒店搜索页 UI 元素...",
    "image_before": "/abs/path/00_start.jpg",
    "image_after": "/abs/path/01_Tap_538_2324_.jpg",
    "subgoal_text": "Navigate to hotel search...",
    "is_backtrack": false
  }]
}
```

---

### Stage 2 — `LLM_caller/` LLM 评分生成器

调用大语言模型为每一步生成结构化评分。

**适配器模式**: 三种适配器覆盖不同后端

| 适配器 | 协议 | 用途 |
|--------|------|------|
| `OpenAIStyleAdapter` | `/v1/chat/completions` | DeepSeek, 远程 Qwen |
| `OllamaAdapter` | `/api/chat` | 历史兼容的本地 Ollama 后端 |
| `QwenVLLMAdapter` | `/v1/chat/completions` | Qwen 多模态 (传入 before/after 图片) |

**Prompt 模板**:

| 模板 | 适用 | 特点 |
|------|------|------|
| `RRM_V1.yaml` | DeepSeek | 中文指令, 详细格式 |
| `RRM_Qwen.yaml` | Qwen3.5-4B 纯文本 | 严格标签格式, system_message |
| `RRM_Qwen_MM.yaml` | Qwen3.5-4B 多模态 | 含图片标注说明 + 防循环规则 |

**并发加速**: `ThreadPoolExecutor(max_workers=4)` — 20 步轨迹从 13 分钟降至 4 分钟。

支持 `--indices=1,3,7` 参数仅重试指定步骤（用于 QA 审查失败后的重跑）。

---

### QA + Retry — `logic_QA/` 审查 + 重试

LLM 产出的 `<think>` / `<critique>` / `<score>` 经过两步审查：

| 评判器 | 检查内容 |
|--------|---------|
| `DensityEvaluator` | think 长度 ≥ 20 字、是否引用 UI 元素、是否模板化 |
| `ConflictEvaluator` | Q 骤降但不说有问题、骂了给高分、状态没变却高分 |

**重试机制**：QA 未通过的步骤，单独重新调用 LLM（`--indices` 模式），最多 3 次。

**角色**：QA 的 `is_accepted` 仅用于内部重试决策，**不参与最终筛选**。

---

### Stage 3 — `core_prm/` 过程奖励计算引擎

对 LLM 产生的 Q 值序列进行时序数学加工。

| 指标 | 公式 | 说明 |
|------|------|------|
| Progress | `A_t = Q_t − V_t` | 当前步相对历史均值的增值 |
| TD-Error | `δ_t = r_t + γ·Q_t − Q_{t−1}` | 单步对未来的实际贡献误差 |
| GAE | `A_t^GAE = Σ (γ·λ)^k · δ_{t+k}` | 指数衰减平滑，回溯分配终点红利 |

**超参数**: `γ=0.99`, `λ=0.95`

---

### `report/` 报告生成模块

扫描 `output/<set>/` 下产物 JSON：

| 报告 | 内容 |
|------|------|
| `summary.json` | 单轨迹汇总: avg_q, avg_progress, avg_gae + 每步详情 |
| `master_summary.json` | 总报告: 所有轨迹对比 + overall 统计 |

---

### `filter/` 轨迹筛选模块

基于两条核心指标将轨迹分为四类：

| 指标 | 公式 | 阈值 | 回答 |
|------|------|------|------|
| `composite_completion` | `avg_q × 0.4 + norm_gae × 0.6` | ≥ 0.5 | 是否完成目标 |
| `step_pass_ratio` | `Q ≥ 0.7 的步骤数 / 总步骤数` | 0.85-0.90 | 步骤逻辑是否合理 |

```
               step_pass_ratio ≥ 阈值          step_pass_ratio < 阈值
composite     ┌────────────────────────┬────────────────────────┐
≥ 0.5          │  A: 完成目标_逻辑合理    │  B: 完成目标_逻辑可疑    │
composite     ├────────────────────────┼────────────────────────┤
< 0.5          │  C: 未完成_局部合理      │  D: 未完成_逻辑异常      │
               └────────────────────────┴────────────────────────┘
```

筛选规则全部在 `filter/config/filter_config.yaml` 中配置。

---

## 已知问题和诊断

### 五种方案质量对比（基于 290 步实测）

| 指标 | DeepSeek v4 Pro | Qwen3.5-4B 纯文本(单步) | Qwen3.5-4B 纯文本(窗口) | Qwen3.5-4B 多模态(单步) | Qwen3.5-4B 多模态(窗口) |
|------|:---------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
| **QA 通过率** | **98%** | **100%** | **100%** | 31% | 44% |
| **三项标签完整率** | **99%** | **100%** | **100%** | 31% | ~44% |
| **加权 avg_Q** | **0.551** | 0.503 | 0.419 | 0.849* | 0.248 |
| **单步响应时间** | ~2s | ~4s | ~14s | 3~170s | 38~92s |
| **视觉验证能力** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **本地运行** | ❌ 需 API | ✅ | ✅ | ✅ | ✅ |
| **成本** | API 按量计费 | 免费 | 免费 | 免费 | 免费 |

> *MM 单步 avg_Q 偏高（模糊步 score 缺失未计入）。窗口方案 avg_Q 偏低（上下文使模型更保守）。
> 详见 `project_summary.md` 完整对比报告。

### 常见失败模式

| 问题 | 表现 | 原因 | 状态 |
|------|------|------|------|
| Score 解析失败 | think 有内容但 score=0.0 | 模型在 `<think>` 草稿嵌入 `<score>` 标签 | ✅ 已修复 (Parser 取最后匹配) |
| Chain-of-thought 泄露 | think 包含 "Thinking Process: 1. Analyze..." | 模型输出内部推理过程 | ⚠️ 偶发, `/no_think` 可缓解 |
| 首步预热慢 | 首步 ~55s, 后续 ~3s | GPU 冷启动 | ⚠️ 已用并发弥补 |
| 实体对齐不足 | density_index 常低 | 模型不习惯引用 UI 元素名 | ⚠️ Prompt 已优化 |
| Back 动作评分不一致 | 相同 Back 得分 0.0~0.9 | 模型对导航动作理解不稳定 | ⚠️ 需更大模型 |
| QA 不检查 score 质量 | score=0.0 的步骤仍通过 QA | QA 只检查 think 长度和冲突 | ⚠️ 设计如此 (QA 是重试触发器) |

### 建议运行策略

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 生产数据标注/最终筛选 | **DeepSeek** | 评分最合理、输出完整、QA通过率98% |
| 本地开发/快速调试 | **Qwen3.5-Text单步** | 100%可靠、速度快、零维护 |
| 离线批量处理（无网络） | **Qwen3.5-Text单步** | 不需API，全本地运行，100%可靠 |
| 需完整think的训练数据 | **Qwen3.5-Text窗口** | 100% think完整率，但评分偏低 |
| 点击位置验证/质检抽样 | **Qwen3.5-MM单步** | 唯一能发现点击偏差的方案 |
| 视觉验证+think补全 | ⚠️ **Qwen3.5-MM窗口** | 效果有限，avg_Q过低 |
| 批量处理加速 | 并发模式 (`max_workers=4`) | 3x加速, 已在 `main.py` 内置 |

---

## 设计原则

| 原则 | 贯彻方式 |
|------|---------|
| **开闭原则 (OCP)** | 策略模式：新增 StateBuilder / Estimator / Evaluator / 筛选维度 只需添加子类并注册配置 |
| **单一职责 (SRP)** | 每个文件职责唯一 (loader 管 IO, builder 管转换, evaluator 管评判, scorer 管计算) |
| **高内聚低耦合** | 模块间仅通过 JSON 文件交互，无代码级依赖；pipeline 基于子进程串联 |
| **不修改原始数据** | 所有模块只读取 `trajs/`，产物写入 `output/<set_name>/` |
| **适配器模式** | LLM_caller 支持 OpenAI / Qwen vLLM / Ollama 兼容后端；core_prm 支持多种 Q 值输入 |

## 扩展指南

| 需求 | 操作位置 | 方法 |
|------|---------|------|
| 新增 LLM 模型 | `LLM_caller/adapters/` | 继承 `BaseAdapter`，在 `caller.py` 注册 |
| 新增 PRM 算法 | `core_prm/estimators/` | 继承 `BaseEstimator`，在 config 切换 |
| 新增 QA 评判维度 | `logic_QA/evaluators/` | 继承 `BaseEvaluator`，在 config 注册 |
| 新增 state 描述策略 | `env_parser/core/state_builder.py` | 继承 `BaseStateBuilder` |
| 新增筛选维度 | `filter/utils/conditions.py` | 新增操作符；在 scorer 加指标；在 yaml 加规则 |
| 调整筛选阈值 | `filter/config/filter_config.yaml` | 改 value 即可 |
| 改变 Prompt 模板 | `LLM_caller/prompts/` | 新建 YAML，在 config 切换 |
| 调参 (γ/λ) | `core_prm/config/prm_params.yaml` | 修改 YAML 即可生效 |

## 依赖

- Python 3.10+
- PyYAML
- requests
- DeepSeek API 或本地 vLLM OpenAI 兼容服务
- (可选) lmdeploy, transformers, torch (用于本地模型加载)
