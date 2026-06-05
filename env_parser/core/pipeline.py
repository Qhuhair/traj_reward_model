import os
import json
import yaml

from .loader import TrajectoryLoader
from .state_builder import create_state_builder
from .formatter import TrajectoryFormatter
from ..utils.path_utils import list_traj_dirs


class EnvParserPipeline:
    """
    Facade — 环境解析流水线唯⼀入口。
    串联 loader → builder → formatter。
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config", "parser_config.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._loader = TrajectoryLoader()
        self._builder = create_state_builder(self.config)
        self._formatter = TrajectoryFormatter()

    def run(self, traj_dir: str) -> dict:
        """
        处理单条轨迹。
        输入: trajs/<traj_id>/ 目录路径
        输出: 标准化 dict {trajectory_id, app, task, subgoals, steps}
        """
        traj_id = os.path.basename(os.path.normpath(traj_dir))
        metadata = self._loader.load(traj_dir)
        aligned = self._loader.align(traj_dir)
        task_desc = metadata.get("overall_task", metadata.get("instruction", ""))

        state_before_list = [
            self._builder.build(step.perception_before, self.config)
            for step in aligned
        ]
        state_after_list = [
            self._builder.build(step.perception_after, self.config)
            if step.perception_after else step.to_node_desc
            for step in aligned
        ]

        standardized = self._formatter.format_trajectory(
            aligned, state_before_list, state_after_list, task_desc
        )

        return self._formatter.format_output(traj_id, metadata, standardized)

    def run_all(self, trajs_root: str, save_output: bool = False) -> list:
        """
        批量处理 trajs/ 下所有轨迹。
        save_output=True 时写入 output/ 目录，不修改原始数据。
        """
        results = []
        for tid in list_traj_dirs(trajs_root):
            traj_dir = os.path.join(trajs_root, tid)
            output = self.run(traj_dir)
            results.append(output)

            if save_output:
                output_dir = os.path.join(
                    trajs_root, self.config.get("output_dir", "output")
                )
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{tid}_standard.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

        return results
