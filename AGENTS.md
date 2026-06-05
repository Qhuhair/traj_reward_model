# 仓库协作指南

## 仓库布局

本仓库是 Python 3.10+ 的 GUI 轨迹奖励评估流水线。`pipelines/` 模块集中管理各类流水线入口；根目录 `run_*.py` 脚本只作为兼容入口保留。

重要目录：

- `trajs/`：原始轨迹数据，只读，不要改写源样本。
- `env_parser/`：Stage 1 环境解析，将轨迹感知、截图、metadata、history 转为标准化 State-Action JSON。
- `LLM_caller/`：Stage 2 LLM 评分，包含适配器、Prompt、图片标注和重试工具。
- `logic_QA/`：LLM 输出质检，用于触发重试，不参与最终筛选。
- `core_prm/`：过程奖励计算，产出 Progress、TD-Error、GAE。
- `report/`：生成 `summary.json` 和 `master_summary.json`。
- `filter/`：基于 `composite_completion` 和 `step_pass_ratio` 进行 A/B/C/D 分类。
- `models/`：本地模型服务辅助模块，当前包含 vLLM 启动与测试工具。
- `pipelines/`：流水线入口模块，包含完整流水线、多模态、滑动窗口、纯图片和 GUI-Shepherd 两步评估入口。
- `Lora/checkpoint-248/`：本地 LoRA/checkpoint 资产，避免随意修改。
- `output/`：生成物目录，包含评分、报告、标注图片和筛选结果。

## 模块 README

- 根目录总览：[README.md](README.md)
- 环境解析模块：[env_parser/README.md](env_parser/README.md)
- LLM 调用模块：[LLM_caller/README.md](LLM_caller/README.md)
- QA 审查模块：[logic_QA/README.md](logic_QA/README.md)
- PRM 计算模块：[core_prm/README.md](core_prm/README.md)
- 报告生成模块：[report/README.md](report/README.md)
- 轨迹筛选模块：[filter/README.md](filter/README.md)
- 本地模型模块：[models/README.md](models/README.md)
- vLLM 服务模块：[models/vllm/README.md](models/vllm/README.md)
- 流水线入口模块：[pipelines/README.md](pipelines/README.md)
- LoRA 检查点说明：[Lora/checkpoint-248/README.md](Lora/checkpoint-248/README.md)

## 如何运行

安装基础依赖：

```bash
pip install pyyaml requests
```

运行指定轨迹集：

```bash
python -m pipelines.full 20260113_214142_subgoal
```

运行全部轨迹集：

```bash
python -m pipelines.full
```

兼容旧入口仍可使用，例如 `python run_pipeline.py 20260113_214142_subgoal`。

常用模块级命令：

```bash
python env_parser/main.py
python LLM_caller/main.py output/<set>/traj_007/standardized.json output/<set>/traj_007/llm_scores.json
python logic_QA/main.py output/<set>/traj_007/standardized.json output/<set>/traj_007/llm_scores.json output/<set>/traj_007/qa_reports.json
python core_prm/main.py output/<set>/traj_007/llm_scores.json output/<set>/traj_007/prm_scores.json
python filter/main.py 20260113_214142_subgoal
python -m pipelines.text_window 20250113_21442_test
```

通过 `LLM_caller/config.yaml` 切换 `active_model` 和 Prompt。生产标注优先 DeepSeek，本地调试优先 Qwen/Ollama 纯文本；需要截图验证时再使用多模态方案。

## 构建、测试和检查

当前仓库没有统一的 package build 文件、pytest 配置或 lint 配置。修改后应运行覆盖变更范围的最小命令：解析器改动运行 `python env_parser/main.py`；筛选逻辑改动运行 `python filter/main.py <output_set>`；本地模型服务改动运行 `python models/test_model.py`。

新增测试时使用 `test_*.py` 命名，优先使用小型合成 JSON fixture，不要让单元测试依赖真实 API。

提供 Python 命令行验证片段时，统一使用 `python -c` 加反斜杠换行，不使用 heredoc。例如：

```bash
python -c "import torch; \
print(torch.__version__)"
```

提供 shell 命令时，必须解释命令用途，并说明重要选项、环境变量、路径、端口和副作用。

## 工程惯例

Python 代码使用 4 空格缩进；函数、变量、模块和 YAML key 使用 `snake_case`。JSON 载荷要保持清晰 schema，优先使用 dataclass 或结构明确的 dict。

遵守已有模块模式：loader 负责 IO，builder/formatter 负责转换，evaluator/estimator 负责单一策略，`pipeline`、`orchestrator`、`engine` 负责编排。新增模型适配器放在 `LLM_caller/adapters/` 并继承 `BaseAdapter`；新增 PRM 算法放在 `core_prm/estimators/`；新增 QA 维度放在 `logic_QA/evaluators/`；新增筛选规则优先修改 `filter/config/filter_config.yaml`，必要时补充 scorer 或 condition。

新增和修改代码必须遵守开闭原则和基本软件工程原则。优先扩展稳定抽象，不要为局部需求大范围改写无关代码；保持职责单一，一个函数只做一件事。

## 约束和禁止规则

不要修改 `trajs/` 原始数据；派生产物写入 `output/`。不要提交 API key、私有服务地址、本机绝对模型路径或大模型权重。除非任务明确要求，不要编辑生成物。

不要把 `logic_QA` 的接受结果作为最终筛选条件；QA 只用于内部重试，最终 A/B/C/D 分类由 `filter/` 负责。

终端命令禁止使用 `sudo`。如果某个操作看起来需要提权，停止并解释限制，不要绕过规则。

完整流水线可能调用付费 API、本地 GPU 服务或长时间多模态推理；除非确实需要端到端验证，优先运行小范围命令。

## 完成标准

任务完成意味着：变更涉及的阶段能在代表性输入上成功运行，预期 JSON 或日志产物已生成，下游模块仍能识别当前 schema。

常见验证产物包括：`standardized.json`、`llm_scores.json`、`qa_reports.json`、`prm_scores.json`、`summary.json`、`master_summary.json`、`filtered.json` 和 `models/vllm/log/vllm_qwen35_4b.log`。

PR 或变更说明应包含：影响的模块、运行过的命令、配置变更、使用的模型/后端，以及行为变化前后的输出路径或关键指标。当前工作区无法可靠读取 Git 历史，因此提交信息使用简洁祈使句，例如 `Fix PRM score parsing fallback` 或 `Add QA retry threshold config`。
