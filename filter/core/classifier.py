from ..utils.conditions import evaluate_conditions


class RuleClassifier:
    """规则引擎: 按 config 中 category_rules 顺序匹配，第一命中即返回"""

    def __init__(self, rules: list):
        self.rules = rules

    def classify(self, metrics: dict) -> tuple:
        """返回 (category_id, category_label, [condition_descriptions])"""
        for rule in self.rules:
            conditions = rule.get("conditions", {})
            if not conditions:
                continue
            passed, descriptions = evaluate_conditions(metrics, conditions)
            if passed:
                return rule.get("id", "?"), rule.get("label", "?"), descriptions

        return "?", "未分类", ["No category rule matched"]
