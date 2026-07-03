import yaml
import re
import logging
import time

class ConfigLoader:
    @staticmethod
    def load(path="config.yaml"):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

class ResponseParser:
    @staticmethod
    def _normalize_score(value) -> float | None:
        """将模型文本中的分数字符串归一化到 0.0-1.0，失败时返回 None。"""
        try:
            score = float(str(value).strip())
        except (ValueError, TypeError):
            return None
        if 0.0 <= score <= 1.0:
            return score
        return None

    @staticmethod
    def _extract_score_fallback(raw_text: str) -> float | None:
        """从非标准输出中兜底提取评分，优先取靠近评分关键词的最后一个数值。"""
        patterns = [
            r"(?:最终(?:评分|得分|决定)|评分|得分|分数|score)\s*(?:为|是|[:：=])?\s*([01](?:\.\d+)?)\s*(?:分)?",
            r"(?:给|判为|打)\s*([01](?:\.\d+)?)\s*分",
            r"([01](?:\.\d+)?)\s*分\s*(?:。|\.|；|;|$)",
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, raw_text, re.IGNORECASE))
        for candidate in reversed(candidates):
            score = ResponseParser._normalize_score(candidate)
            if score is not None:
                return score
        return None

    @staticmethod
    def _parse_gs(raw_text: str) -> dict:
        """解析 GUI-Shepherd 的【视觉扫描】→【动作匹配】→【最终得分】格式"""
        result = {"think": "N/A", "critique": "N/A", "score": 0.0}

        # 提取三段
        scan_m = re.search(r"【视觉扫描[】:：](.*?)(?=【|$)", raw_text, re.DOTALL)
        match_m = re.search(r"【动作匹配[】:：](.*?)(?=【|$)", raw_text, re.DOTALL)

        lines = []
        if scan_m:
            lines.append(f"视觉扫描: {scan_m.group(1).strip()}")
        if match_m:
            lines.append(f"动作匹配: {match_m.group(1).strip()}")

        result["think"] = "\n".join(lines) if lines else raw_text.strip()[:200]
        result["critique"] = match_m.group(1).strip() if match_m else "N/A"

        # 用 findall 取最后一个【最终得分】（模型思考中会引用格式说明导致第一个是假的）
        score_matches = re.findall(r"【最终得分[】:：]\s*(\d+)", raw_text)
        if score_matches:
            val = int(score_matches[-1])
            result["score"] = 1.0 if val >= 1 else 0.0
        else:
            # 回退：从文本末尾提取孤立的 0 或 1
            nums = re.findall(r"(\d+)", raw_text)
            if nums:
                result["score"] = 1.0 if int(nums[-1]) >= 1 else 0.0

        return result

    @staticmethod
    def parse(raw_text):
        # 检测是否为 GUI-Shepherd 【视觉扫描】/【最终得分】格式
        if "【视觉扫描" in raw_text or "【最终得分" in raw_text:
            return ResponseParser._parse_gs(raw_text)

        """提取 <think>/<thinking>, <critique>, <score> 标签内容"""
        result = {}

        # 1. 提取 think — 兼容多种格式
        # 1a. 优先匹配完整 <think>...</think> 或 <thinking>...</thinking>
        think_matches = re.findall(r"<think(?:ing)?>(.*?)</think(?:ing)?>", raw_text, re.DOTALL)
        if think_matches:
            result["think"] = think_matches[-1].strip()
        else:
            # 1b. 兼容只有 </think> 闭合标签没有开标签的情况
            #     取 </think> 之前的内容直到上一个标签结束或开头
            m = re.search(r"(.*?)</think(?:ing)?>", raw_text, re.DOTALL)
            if m:
                # 去掉 critique/score 等后续标签的干扰
                before = m.group(1).strip()
                # 如果 before 以 <critique> 等开头，说明没有 think 内容
                if before and not before.startswith("<"):
                    result["think"] = before
                else:
                    result["think"] = "N/A"
            else:
                # 1c. 无 XML 标签：将原始文本作为 think（GS 填空格式等）
                result["think"] = raw_text.strip() if raw_text.strip() else "N/A"

        # 2. critique / score: 取最后一个
        for key in ("critique", "score"):
            matches = re.findall(f"<{key}>(.*?)</{key}>", raw_text, re.DOTALL)
            result[key] = matches[-1].strip() if matches else "N/A"
        
        # 3. 尝试将 score 转为 float；如果标签缺失，再从长推理文本中兜底寻找“评分 0.x”等表达。
        score = ResponseParser._normalize_score(result["score"])
        if score is None:
            score = ResponseParser._extract_score_fallback(raw_text)
        result["score"] = score if score is not None else 0.0
        return result

class DebugLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def log_call(self, model_name, prompt, response, duration):
        if not self.enabled: return
        logging.info(f"--- Model: {model_name} | Duration: {duration:.2f}s ---")
        logging.info(f"Input Prompt Snippet: {prompt[:50]}...")
        logging.info(f"Raw Response: {response}")
