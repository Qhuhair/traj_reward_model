import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from filter.core.filter_engine import FilterEngine


def _print_report(report: dict):
    s = report["summary"]
    print(f"Run: {report['run_dir']}")
    print(f"Filtered at: {report['filtered_at']}")
    print(f"Total trajectories: {s['total_trajectories']}")
    print()

    cat_labels = {
        "A": "A-完成目标_逻辑合理",
        "B": "B-完成目标_逻辑可疑",
        "C": "C-未完成_局部合理",
        "D": "D-未完成_逻辑异常",
    }

    print(f"  {'Category':<30} {'Count':<8}")
    print("-" * 40)
    for cid, label in cat_labels.items():
        cnt = s["category_counts"].get(cid, 0)
        print(f"  {label:<30} {cnt:<8}")
    print()

    for t in report["trajectories"]:
        m = t["raw_scores"]
        print(
            f"  [{t['category']}] {t['trajectory_id']:<10} "
            f"avg_q={m['avg_q']:.3f}  "
            f"qa={t['n_steps']}/{t['n_steps']}  "
            f"GAE={m['avg_gae']:+.3f}"
        )
        for rule in t["matched_rules"]:
            print(f"      {rule}")

    out = os.path.join(
        PROJECT_ROOT, "output", report["run_dir"], "filtered.json"
    )
    print(f"\nReport saved: {out}")


def main():
    output_root = os.path.join(PROJECT_ROOT, "output")

    # 自动选最新的 run_dir, 除非手动指定
    if len(sys.argv) > 1:
        run_dir = os.path.join(output_root, sys.argv[1])
    else:
        dirs = sorted(os.listdir(output_root), reverse=True)
        run_dir = os.path.join(output_root, dirs[0]) if dirs else None

    if not run_dir or not os.path.isdir(run_dir):
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)

    engine = FilterEngine()
    report = engine.run(run_dir)
    _print_report(report)


if __name__ == "__main__":
    main()
