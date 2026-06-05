from .base_estimator import BaseEstimator, EstimationResult


class ProgressCalculator(BaseEstimator):
    """
    进展估计器
    A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
    V(s_t) = 滑动平均值 / 前一时刻 Q / 零基线
    """

    def evaluate(self, q_sequence: list, rewards: list, config: dict) -> EstimationResult:
        n = len(q_sequence)
        if n == 0:
            return EstimationResult(progress=[], td_errors=[], advantages=[],
                                    method=self.__class__.__name__)

        progress_cfg = config.get("progress", {})
        method = progress_cfg.get("baseline_method", "running_mean")

        progress = []
        running_sum = 0.0

        for t in range(n):
            qt = q_sequence[t]

            if method == "running_mean":
                vt = running_sum / t if t > 0 else 0.0
            elif method == "prev_q":
                vt = q_sequence[t - 1] if t > 0 else 0.0
            elif method == "zero":
                vt = 0.0
            else:
                vt = 0.0

            at = qt - vt
            progress.append(round(at, 6))
            running_sum += qt

        return EstimationResult(
            progress=progress,
            td_errors=[],
            advantages=[],
            method=f"{self.__class__.__name__}(baseline={method})",
        )
