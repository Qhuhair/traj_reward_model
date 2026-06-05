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

多模态单步流水线：

```bash
python -m pipelines.multimodal 20250113_21442_test
```

`pipelines.multimodal` 会进行截图标注，并使用 `qwen_local_mm` 和
`RRM_Qwen_MM` 进行逐步评分。

多模态滑动窗口：

```bash
python -m pipelines.sliding 20250113_21442_test
```

`pipelines.sliding` 使用多模态滑动窗口上下文，并在 QA 失败时执行上下文重试。

纯文本滑动窗口：

```bash
python -m pipelines.text_window 20250113_21442_test
```

`pipelines.text_window` 使用纯文本 Qwen 模型，额外生成轨迹级总结。

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

## 维护规则

新增流水线入口应放入 `pipelines/`，根目录只保留短兼容脚本。共享路径逻辑放在
`paths.py`，不要在各脚本中重复推导项目根目录。流水线内部仍应通过 JSON 文件衔接模块，
避免隐藏的跨模块状态。
