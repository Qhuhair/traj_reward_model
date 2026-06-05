import os
import json
import yaml
from datetime import datetime

from .loader import load_run_dir
from .scorer import compute_all
from .classifier import RuleClassifier


class FilterEngine:
    """筛选主引擎 — Facade"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config", "filter_config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.classifier = RuleClassifier(self.config.get("category_rules", []))

    def run(self, run_dir: str) -> dict:
        trajs = load_run_dir(run_dir)
        if not trajs:
            print("[FILTER] No trajectory reports found")
            return {}

        step_threshold = self.config.get("step_pass_threshold", 0.6)

        results = []
        counters = {}
        for t in trajs:
            metrics = compute_all(t, step_pass_threshold=step_threshold)
            cat_id, cat_label, conditions = self.classifier.classify(metrics)
            counters[cat_id] = counters.get(cat_id, 0) + 1

            results.append({
                "trajectory_id": t.traj_id,
                "app": t.app,
                "task": t.task,
                "n_steps": t.n_steps,
                "category": cat_id,
                "category_label": cat_label,
                "matched_rules": conditions,
                "raw_scores": metrics,
                "step_details": [
                    {
                        "step_idx": s.step_idx,
                        "action": s.action,
                        "q": s.q,
                        "progress": s.progress,
                        "gae": s.gae,
                        "qa_accepted": s.qa_accepted,
                        "meta_score": s.meta_score,
                        "density_index": s.density_index,
                        "conflict_index": s.conflict_index,
                        "step_pass": s.q >= step_threshold,
                    }
                    for s in t.steps
                ],
            })

        filtered_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        report = {
            "run_dir": os.path.basename(run_dir),
            "filtered_at": filtered_at,
            "config_snapshot": {
                "category_count": len(self.config.get("category_rules", [])),
                "categories": [
                    f"{r['id']}:{r['label']}"
                    for r in self.config.get("category_rules", [])
                ],
            },
            "summary": {
                "total_trajectories": len(results),
                "category_counts": counters,
            },
            "trajectories": results,
        }

        out_path = os.path.join(run_dir, "filtered.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report
