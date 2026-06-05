import math
from .loader import TrajData


def compute_all(traj: TrajData, step_pass_threshold: float = 0.6) -> dict:
    """从 TrajData 计算全部衍生指标，返回 dict"""

    n = traj.n_steps
    if n == 0:
        return {}
    steps = traj.steps

    q_seq = [s.q for s in steps]
    prog_seq = [s.progress for s in steps]
    gae_seq = [s.gae for s in steps]
    meta_seq = [s.meta_score for s in steps]
    density_seq = [s.density_index for s in steps]
    conflict_seq = [s.conflict_index for s in steps]

    qa_accepted = [s.qa_accepted for s in steps]
    avg_q = _mean(q_seq)
    avg_gae = _mean(gae_seq)
    avg_progress = _mean(prog_seq)

    # ---- 新指标: 步骤通过率 ----
    step_pass_count = sum(1 for s in steps if s.q >= step_pass_threshold)
    step_pass_ratio = step_pass_count / n

    # ---- 新指标: GAE 归一化到 [0,1] ----
    gae_min = min(gae_seq)
    gae_max = max(gae_seq)
    gae_range = gae_max - gae_min
    if gae_range > 1e-9:
        norm_gae = (_clamp((avg_gae - gae_min) / gae_range, 0.0, 1.0)
                    if gae_max > 0 else 0.0)
    else:
        norm_gae = 0.5

    # ---- 新指标: 综合完成度 (LLM评分 40% + core_prm GAE 60%) ----
    composite_completion = avg_q * 0.4 + norm_gae * 0.6

    last = steps[-1]

    return {
        "avg_q": avg_q,
        "avg_progress": avg_progress,
        "avg_gae": avg_gae,
        "qa_pass_ratio": sum(1 for a in qa_accepted if a) / n,
        "last_step_q": last.q,
        "backtrack_ratio": 0.0,
        "q_increasing_ratio": sum(1 for i in range(1, n) if q_seq[i] > q_seq[i - 1]) / max(n - 1, 1),
        "avg_meta_score": _mean(meta_seq),
        "avg_density_index": _mean(density_seq),
        "avg_conflict_index": _mean(conflict_seq),
        "terminal_gae": last.gae,
        "score_volatility": _std(q_seq),
        "max_q": max(q_seq),
        "min_q": min(q_seq),
        "q_range": max(q_seq) - min(q_seq),
        "positive_progress_ratio": sum(1 for p in prog_seq if p > 0) / n,
        "high_score_ratio": sum(1 for q in q_seq if q >= 0.7) / n,
        # 新指标
        "step_pass_ratio": step_pass_ratio,
        "step_pass_count": step_pass_count,
        "composite_completion": composite_completion,
        "norm_avg_gae": norm_gae,
    }


def _mean(seq: list) -> float:
    if not seq:
        return 0.0
    return sum(seq) / len(seq)


def _std(seq: list) -> float:
    if len(seq) < 2:
        return 0.0
    avg = _mean(seq)
    return math.sqrt(sum((x - avg) ** 2 for x in seq) / len(seq))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
