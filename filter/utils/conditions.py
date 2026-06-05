EPSILON = 1e-7

OPERATORS = {
    "ge": lambda a, b: a >= b - EPSILON,
    "gt": lambda a, b: a > b + EPSILON,
    "le": lambda a, b: a <= b + EPSILON,
    "lt": lambda a, b: a < b - EPSILON,
    "eq": lambda a, b: abs(a - b) < 1e-9,
    "ne": lambda a, b: abs(a - b) >= 1e-9,
}


def evaluate_condition(metric_value: float, op: str, threshold: float) -> bool:
    """对单个条件求值"""
    fn = OPERATORS.get(op)
    if fn is None:
        raise ValueError(f"Unknown operator: {op}")
    return fn(metric_value, threshold)


def describe_condition(metric: str, op: str, value: float, actual: float) -> str:
    """生成条件的可读描述"""
    op_label = {
        "ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "==", "ne": "!="
    }
    sym = op_label.get(op, op)
    result = "PASS" if evaluate_condition(actual, op, value) else "FAIL"
    return f"{metric}({actual:.4f}) {sym} {value} -> {result}"


def evaluate_conditions(metrics: dict, conditions: dict) -> tuple:
    """
    conditions = {"all_of": [...], "any_of": [...]}
    返回 (passed: bool, descriptions: list[str])
    """
    all_of = conditions.get("all_of", [])
    any_of = conditions.get("any_of", [])
    descriptions = []
    passed = True

    for c in all_of:
        actual = metrics.get(c["metric"], 0)
        ok = evaluate_condition(actual, c["op"], c["value"])
        descriptions.append(describe_condition(c["metric"], c["op"], c["value"], actual))
        if not ok:
            passed = False

    if any_of and not any(
        evaluate_condition(metrics.get(c["metric"], 0), c["op"], c["value"])
        for c in any_of
    ):
        passed = False
        for c in any_of:
            actual = metrics.get(c["metric"], 0)
            descriptions.append(
                describe_condition(c["metric"], c["op"], c["value"], actual)
            )

    return passed, descriptions
