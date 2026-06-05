from .base_estimator import BaseEstimator, EstimationResult


class TDGAEEngine(BaseEstimator):
    """
    TD-Error + GAE 引擎
    δ_t = r_t + γ·Q_t - Q_{t-1}
    A_t^GAE = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}
    """

    def evaluate(self, q_sequence: list, rewards: list, config: dict) -> EstimationResult:
        n = len(q_sequence)
        if n == 0:
            return EstimationResult(progress=[], td_errors=[], advantages=[],
                                    method=self.__class__.__name__)

        gamma = config.get("gamma", 0.99)
        lam = config.get("lambda_gae", 0.95)
        td_cfg = config.get("td_gae", {})
        terminal_reward = td_cfg.get("terminal_reward", 1.0)
        non_terminal_reward = td_cfg.get("non_terminal_reward", 0.0)

        td_errors = self._calc_td_errors(q_sequence, rewards, gamma,
                                         terminal_reward, non_terminal_reward)
        advantages = self._calc_gae(td_errors, gamma, lam)

        progress = [round(q_sequence[t] - (q_sequence[t-1] if t > 0 else 0.0), 6)
                    for t in range(n)]

        return EstimationResult(
            progress=progress,
            td_errors=td_errors,
            advantages=advantages,
            method=f"{self.__class__.__name__}(γ={gamma}, λ={lam})",
        )

    @staticmethod
    def _calc_td_errors(q_seq: list, rewards: list, gamma: float,
                        terminal_r: float, non_terminal_r: float) -> list:
        """计算每一步的 TD-Error: δ_t = r_t + γ·Q_t - Q_{t-1}"""
        n = len(q_seq)
        td = []

        for t in range(n):
            qt = q_seq[t]
            qt_prev = q_seq[t - 1] if t > 0 else 0.0
            is_terminal = (t == n - 1)
            rt = terminal_r if is_terminal else non_terminal_r
            rt += rewards[t] if t < len(rewards) else 0.0

            delta = rt + gamma * qt - qt_prev
            td.append(round(delta, 6))

        return td

    @staticmethod
    def _calc_gae(td_errors: list, gamma: float, lam: float) -> list:
        """广义优势估计: A_t = Σ_{k} (γλ)^k · δ_{t+k}"""
        n = len(td_errors)
        gae = [0.0] * n
        running = 0.0

        for t in range(n - 1, -1, -1):
            running = td_errors[t] + gamma * lam * running
            gae[t] = round(running, 6)

        return gae
