# Pipelines 流水线入口模块

`pipelines/` 集中管理根目录原有的 `run_*.py` 流水线脚本。根目录仍保留
短兼容入口，因此旧命令可以继续使用；新代码应优先通过 `python -m pipelines.<模块名>`
运行。

## 目录结构

```text
pipelines/
├── full.py          # 完整流水线：env_parser -> LLM -> QA -> PRM -> report -> filter
├── multimodal.py    # 多模态单步评估
├── sliding.py       # 多模态滑动窗口评估
├── text_window.py   # 纯文本滑动窗口评估
├── image_only.py    # 纯图片评估模式
├── gs.py            # GUI-Shepherd 两步评估
└── paths.py         # 共享项目根目录和 Python 解释器路径
```

## 常用命令

完整流水线：

```bash
python -m pipelines.full 20260113_214142_subgoal
```

`python -m pipelines.full` 从模块入口运行完整流水线；后面的参数是轨迹集名称。
默认使用纯文本输入，相当于 `--input-mode text --llm-model qwen_vllm_text --llm-prompt RRM_Qwen`，
要求 vLLM OpenAI 兼容服务已监听 `http://localhost:8002/v1`。

指定输出目录名：

```bash
python -m pipelines.full 20260113_214142_subgoal --output-name crossapp_qwen35_4b_test
```

`--output-name` 指定 `output/` 下的子目录名。上面的命令会将结果写入
`output/crossapp_qwen35_4b_test/`，而不是默认的
`output/20260113_214142_subgoal/`。

静默运行：

```bash
python -m pipelines.full 20260113_214142_subgoal --output-name crossapp_qwen35_4b_test --quiet
```

`--quiet` 会把流水线日志追加写入输出目录下的 `pipeline.log`，控制台不再打印阶段输出。

指定完整输出路径：

```bash
python -m pipelines.full 20260113_214142_subgoal --output-root /tmp/crossapp_eval
```

`--output-root` 指定完整输出目录路径。

遍历全部轨迹集并指定统一实验目录：

```bash
python -m pipelines.full --output-name crossapp_qwen35_4b_all
```

`--output-name` 在未指定轨迹集时会作为总实验目录，内部按轨迹集名分子目录输出，
例如 `output/crossapp_qwen35_4b_all/20260113_214142_subgoal/`。

完整流水线每条轨迹会生成 `llm_scores.json`，并同步生成兼容文件
`LLMscore.json`。

多模态单步流水线：

```bash
python -m pipelines.multimodal 20250113_21442_test
```

`pipelines.multimodal` 会进行截图标注，并使用 `qwen_vllm_mm` 配合
`RRM_Qwen_MM` 进行逐步评分。需确认 `http://localhost:8002/v1`
加载的是支持图片输入的模型。

多模态滑动窗口：

```bash
python -m pipelines.sliding --output-name crossapp_qwen35_4b_multimodal_window --quiet
```

`pipelines.sliding` 使用多模态滑动窗口上下文，并在 QA 失败时执行上下文重试。
省略轨迹集名时会处理全部轨迹集；指定轨迹集名时只处理该集合。

纯文本滑动窗口：

```bash
python -m pipelines.text_window --output-name crossapp_qwen35_4b_text_window --quiet
```

`pipelines.text_window` 使用纯文本 Qwen 模型，额外生成轨迹级总结。
同样支持 `--output-name`、`--output-root` 和 `--quiet`。

GUI-Shepherd 两步：

```bash
python -m pipelines.gs --output-name crossapp_qwen35_4b_gs --quiet
```

`pipelines.gs` 先做视觉识别和动作匹配，再基于文本结论评分。

兼容旧入口：

```bash
python run_pipeline.py 20260113_214142_subgoal
python run_multimodal_pipeline.py 20250113_21442_test
python run_sliding_pipeline.py 20250113_21442_test
python run_text_window_pipeline.py 20250113_21442_test
```

这些脚本只负责转发到 `pipelines/` 中的对应模块。

## 输出目录

各流水线输出仍写入 `output/` 下对应子目录，例如：

- `output/<set>/`：完整流水线
- `output/qwen3.5-4b-image-text/<set>/`：多模态单步
- `output/qwen3.5-4b-image-text-window/<set>/`：多模态滑动窗口
- `output/qwen-text-window/<set>/`：纯文本滑动窗口
- `output/qwen3.5-4b-text-image-gs/<set>/`：GUI-Shepherd 两步

## 维护规则

新增流水线入口应放入 `pipelines/`，根目录只保留短兼容脚本。共享路径逻辑放在
`paths.py`，不要在各脚本中重复推导项目根目录。流水线内部仍应通过 JSON 文件衔接模块，
避免隐藏的跨模块状态。
