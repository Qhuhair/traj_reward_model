import re

TEMPLATE_PATTERNS = [
    r"点击以继续", r"进入下一页", r"等待加载完成", r"如图所示",
    r"根据页面提示", r"继续执行", r"操作成功", r"如上所述",
    r"显而易见", r"毫无疑问",
    r"click to continue", r"go to next page", r"as shown above",
    r"it is obvious that", r"without a doubt",
]

NEGATIVE_KEYWORDS = [
    "错误", "失败", "无效", "不正确", "误导", "冗余", "多余",
    "不应该", "没必要", "循环", "死循环", "卡住", "重复",
    "wrong", "failed", "invalid", "incorrect", "misleading",
    "redundant", "unnecessary", "loop", "stuck", "repeated",
    "not needed", "no progress", "error", "mistake",
]

ZH_WORD = re.compile(r"[\u4e00-\u9fff]{2,}")


def extract_entities(text: str) -> set:
    entities = set()
    entities.update(re.findall(r'\[([^\]]+)\]', text))
    entities.update(re.findall(r'\b([a-zA-Z_][\w]*)\b', text))
    return entities


def count_template_phrases(text: str) -> int:
    count = 0
    for pattern in TEMPLATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


def count_negative_keywords(text: str) -> int:
    count = 0
    for kw in NEGATIVE_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            count += 1
    return count


def extract_ui_entities(state) -> set:
    """
    从 state 中提取关键实体。
    支持: str (state_desc 文本), dict (mock 格式), 或 None
    """
    if state is None:
        return set()
    if isinstance(state, str):
        return set(ZH_WORD.findall(state))
    entities = set()
    for el in state.get("key_elements", []):
        eid = el.get("id", "")
        if eid:
            entities.add(eid)
        text = el.get("text", "")
        if text:
            entities.add(text)
    return entities


def has_state_changed(prev_state, curr_state) -> bool:
    """
    检测两个 state 是否不同。
    支持: str (state_desc), dict (含 to_state 哈希或 mock key_elements)
    """
    if not prev_state or not curr_state:
        return True

    if isinstance(prev_state, dict) and isinstance(curr_state, dict):
        prev_hash = prev_state.get("to_state") or prev_state.get("to")
        curr_hash = curr_state.get("to_state") or curr_state.get("to")
        if prev_hash and curr_hash:
            return prev_hash != curr_hash
        if prev_state.get("current_page") != curr_state.get("current_page"):
            return True
        prev_ids = sorted(
            [el.get("id", "") for el in prev_state.get("key_elements", [])]
        )
        curr_ids = sorted(
            [el.get("id", "") for el in curr_state.get("key_elements", [])]
        )
        return prev_ids != curr_ids

    return prev_state != curr_state
