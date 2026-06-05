import re
from .base_adapter import BaseAdapter, AdaptedData


SCORE_PATTERN = re.compile(r"<score>(.*?)</score>", re.DOTALL | re.IGNORECASE)
NUMERIC = re.compile(r"[\d.]+")


class LLMPromptAdapter(BaseAdapter):
    def parse(self, raw_data) -> AdaptedData:
        errors = []

        if isinstance(raw_data, str):
            q_seq, r_seq, errs = self._parse_text(raw_data)
            errors.extend(errs)
        elif isinstance(raw_data, dict):
            q_seq, r_seq, errs = self._parse_json(raw_data)
            errors.extend(errs)
        elif isinstance(raw_data, list):
            q_seq, r_seq, errs = self._parse_steps(raw_data)
            errors.extend(errs)
        else:
            errors.append(f"Unsupported raw_data type: {type(raw_data)}")
            return AdaptedData(q_sequence=[], rewards=[], errors=errors)

        return AdaptedData(q_sequence=q_seq, rewards=r_seq, errors=errors)

    def _parse_text(self, text: str) -> tuple:
        errors = []
        q_seq = []
        matches = SCORE_PATTERN.findall(text)
        for i, m in enumerate(matches):
            val = self._extract_float(m)
            if val is None:
                errors.append(f"Step {i}: cannot parse score from '{m.strip()[:50]}'")
                val = 0.5
            q_seq.append(val)
        r_seq = [0.0] * len(q_seq)
        if r_seq:
            r_seq[-1] = 1.0
        return q_seq, r_seq, errors

    def _parse_json(self, data: dict) -> tuple:
        steps = data.get("steps", [])
        return self._parse_steps(steps)

    def _parse_steps(self, steps: list) -> tuple:
        errors = []
        q_seq = []
        r_seq = []

        for step in steps:
            promise = self._extract_promise(step, errors)
            progress = self._extract_progress(step)

            q_seq.append(round(promise, 4))
            r_seq.append(round(progress, 4))

        return q_seq, r_seq, errors

    def _extract_promise(self, step: dict, errors: list) -> float:
        """尝试多种路径提取 promise_score (Q值)"""
        lg = step.get("logic_augmentation", {})

        # 路径1: mock 格式 logic_augmentation.promise_score
        raw = lg.get("promise_score")
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

        # 路径2: LLM 输出格式 score (env_parser + LLM_caller 产出)
        score = step.get("score")
        if score is not None:
            try:
                return float(score)
            except (ValueError, TypeError):
                pass

        # 路径3: 降级估计
        fallback = self._fallback_estimate(step)
        errors.append(
            f"Step {step.get('step_idx', '?')}: missing score, fallback={fallback}"
        )
        return fallback

    def _extract_progress(self, step: dict) -> float:
        """尝试多种路径提取 progress_score"""
        lg = step.get("logic_augmentation", {})

        raw = lg.get("progress_score")
        if raw is not None:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

        pw = step.get("progress_score")
        if pw is not None:
            try:
                return float(pw)
            except (ValueError, TypeError):
                pass

        return 0.0

    def _extract_float(self, text: str):
        match = NUMERIC.search(text)
        if match:
            val = float(match.group())
            return min(max(val, 0.0), 1.0)
        return None

    @staticmethod
    def _fallback_estimate(step: dict) -> float:
        action = step.get("action", "")

        if isinstance(action, dict):
            action_type = action.get("type", "")
            description = action.get("description", "")
            if action_type == "VERIFY_SUCCESS":
                return 1.0
            if action_type == "CLICK" and ("进入" in description or "详情" in description):
                return 0.5
            if action_type == "TYPE":
                return 0.3
            if action_type == "CLICK" and "搜索" in description:
                return 0.2
            return 0.3

        # 格式A: action 是纯字符串 "Tap (x, y)" / "Back"
        action_str = str(action).lower()
        if "back" in action_str or step.get("is_backtrack"):
            return 0.0
        if "tap" in action_str:
            return 0.3
        return 0.3
