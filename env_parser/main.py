import os
import sys
from core.pipeline import EnvParserPipeline

TRAJS_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "trajs"
))


def main():
    if len(sys.argv) > 1:
        traj_id = sys.argv[1]
        traj_dir = os.path.join(TRAJS_ROOT, traj_id)
        if not os.path.isdir(traj_dir):
            print(f"轨迹目录不存在: {traj_dir}")
            sys.exit(1)
        pipeline = EnvParserPipeline()
        output = pipeline.run(traj_dir)
        _print_result(output)
    else:
        pipeline = EnvParserPipeline()
        results = pipeline.run_all(TRAJS_ROOT)
        for r in results:
            _print_result(r)
            print()


def _print_result(output: dict):
    print(f"轨迹: {output['trajectory_id']}")
    print(f"App: {output['app']}")
    print(f"任务: {output['task']}")
    print(f"子目标: {len(output['subgoals'])} 个")
    print(f"步骤: {len(output['steps'])} 步")
    print("-" * 60)
    for step in output["steps"]:
        backtrack_tag = " [BACK]" if step["is_backtrack"] else ""
        print(f"  Step {step['step_idx']:>2}{backtrack_tag}: {step['action']}")
        print(f"    动作: {step['action_desc'][:60]}...")
        state_preview = step["state_desc"][:100].replace("\n", " | ")
        print(f"    状态: {state_preview}...")
        print()


if __name__ == "__main__":
    main()
