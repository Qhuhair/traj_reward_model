import re

ICON_PATTERN = re.compile(r"^icon:\s*")
TEXT_PATTERN = re.compile(r"^text:\s*")


def clean_text(raw: str) -> str:
    """去除 'text:' / 'icon:' 前缀，返回纯净文本"""
    raw = raw.strip()
    raw = TEXT_PATTERN.sub("", raw)
    raw = ICON_PATTERN.sub("", raw)
    return raw


def is_icon(elem: dict) -> bool:
    """判断感知元素是否为 icon 类型"""
    text = elem.get("text", "")
    return bool(ICON_PATTERN.match(text))


def is_text(elem: dict) -> bool:
    """判断感知元素是否为 text 类型"""
    text = elem.get("text", "")
    return bool(TEXT_PATTERN.match(text))


def truncate(elements: list, max_n: int) -> list:
    """截断元素列表到最大数量"""
    if len(elements) <= max_n:
        return list(elements)
    return list(elements[:max_n])


def cluster_by_area(elements: list, bins: int = 3) -> list:
    """
    按 y 坐标将元素分组为 top/mid/bottom 区域
    返回 [[top_elems], [mid_elems], [bottom_elems]]
    """
    if not elements or bins <= 0:
        return [list(elements)]

    sorted_elems = sorted(
        elements,
        key=lambda e: e.get("center", [0, 0])[1]
    )

    n = len(sorted_elems)
    per_bin = max(1, n // bins)
    clusters = []
    for i in range(bins):
        start = i * per_bin
        if i == bins - 1:
            end = n
        else:
            end = start + per_bin
        clusters.append(sorted_elems[start:end])
    return clusters


def format_element(elem: dict, include_coordinates: bool = False) -> str:
    """将单个感知元素格式化为一行描述"""
    text = clean_text(elem.get("text", ""))
    if not text:
        return ""

    if include_coordinates:
        coord = elem.get("center", [])
        if len(coord) == 2:
            return f"- {text} @({coord[0]},{coord[1]})"
    return f"- {text}"


def build_state_text(perception: list, config: dict) -> str:
    """
    将完整 perception[] 数组转化为自然语言 state_desc
    这是 state_builder 使用的核心纯函数
    """
    max_n = config.get("max_perception_elements", 30)
    use_cluster = config.get("cluster_by_area", True)
    use_coords = config.get("include_coordinates", False)
    bins = config.get("area_bins", 3)

    elements = list(perception)

    if use_cluster and bins > 1:
        clusters = cluster_by_area(elements, bins)
        labels = ["[Top]", "[Middle]", "[Bottom]"]
        lines = []
        for i, cluster in enumerate(clusters):
            truncated = truncate(cluster, max_n // bins)
            if not truncated:
                continue
            if len(clusters) > 1:
                lines.append(labels[i])
            for elem in truncated:
                line = format_element(elem, include_coordinates=use_coords)
                if line:
                    lines.append(line)
        return "\n".join(lines)

    truncated = truncate(elements, max_n)
    lines = []
    for elem in truncated:
        line = format_element(elem, include_coordinates=use_coords)
        if line:
            lines.append(line)
    return "\n".join(lines)
