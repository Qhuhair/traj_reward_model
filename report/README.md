# Report — 报告生成模块

扫描流水线产物 JSON，生成单轨迹摘要和总报告。

## 目录结构

```
report/
├── __init__.py      # 导出公共 API
├── loader.py        # 加载 output/ 下的 summary.json
├── builder.py       # 构建汇总 dict
└── writer.py        # 写 JSON 文件 + 控制台打印
```

## 各文件职责

### `loader.py` — 数据加载

| 类/函数 | 职责 |
|---------|------|
| `StepRow` (dataclass) | 单步数据: `step_idx`, `action`, `q`, `progress`, `gae`, `qa_accepted` |
| `TrajReport` (dataclass) | 单轨迹数据: `traj_id`, `app`, `task`, `n_steps`, `avg_q/avg_progress/avg_gae`, `qa_pass/qa_total`, `steps: list[StepRow]` |
| `safe_load(path)` | 安全读取 JSON，文件不存在返回 None |
| `load_traj_output(out_dir)` | 读取一条轨迹的全部产物 (summary.json + qa_reports.json + prm_scores.json) → `TrajReport` |
| `load_all_outputs(output_root)` | 扫描 `output/<run>/` 下所有 `traj_*/` 子目录 |

### `builder.py` — 报告构建

| 函数 | 职责 |
|------|------|
| `build_traj_summary(r: TrajReport)` | 单轨迹汇总 dict (输出到 `summary.json`) |
| `build_master_summary(reports, run_dir, ts)` | 总报告 dict，含 overall 统计 (输出到 `master_summary.json`) |

### `writer.py` — 输出

| 函数 | 职责 |
|------|------|
| `save_json(path, data)` | 写入 UTF-8 JSON |
| `print_traj_table(r: TrajReport)` | 控制台打印单轨迹表格 |
| `print_master_table(reports)` | 控制台打印总报告表格 |
| `generate(run_dir)` | 主入口: 加载 → 构建 → 写 JSON → 打印表格 |

## 使用方式

```python
from report.writer import generate

generate("output/20250113_21442_test")
# → output/20250113_21442_test/summary.json (每条轨迹)
# → output/20250113_21442_test/master_summary.json
```

自动被 `run_pipeline.py` 在流水线末尾调用。
