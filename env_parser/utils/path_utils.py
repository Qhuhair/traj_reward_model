import os
import re

STEP_FILE_RE = re.compile(r"^(\d{2})_(.+)\.json$")


def list_traj_dirs(trajs_root: str) -> list:
    """扫描 trajs/ 下直接将 traj_XXX 作为子目录的扁平结构 (旧格式)"""
    entries = []
    for name in sorted(os.listdir(trajs_root)):
        full = os.path.join(trajs_root, name)
        if os.path.isdir(full) and name.startswith("traj_"):
            entries.append(name)
    return entries


def list_traj_sets(trajs_root: str) -> list:
    """返回所有轨迹集目录名 (如 ['20260113_214142_subgoal'])"""
    entries = []
    for name in sorted(os.listdir(trajs_root)):
        full = os.path.join(trajs_root, name)
        if not os.path.isdir(full):
            continue
        # 检查是否包含 traj_* 子目录
        for sub in os.listdir(full):
            if sub.startswith("traj_") and os.path.isdir(os.path.join(full, sub)):
                entries.append(name)
                break
    return entries


def list_traj_dirs_in_set(trajs_root: str, set_name: str) -> list:
    """扫描某轨迹集下所有 traj_XXX 目录 (如 ['traj_004', 'traj_005', ...])"""
    set_dir = os.path.join(trajs_root, set_name)
    if not os.path.isdir(set_dir):
        return []
    entries = []
    for name in sorted(os.listdir(set_dir)):
        full = os.path.join(set_dir, name)
        if os.path.isdir(full) and name.startswith("traj_"):
            entries.append(name)
    return entries


def discover_step_files(traj_dir: str) -> list:
    """
    返回 [(filename, index, action_tag), ...]
    例如: ('00_start.json', 0, 'start'), ('01_Tap_129_348_.json', 1, 'Tap_129_348_')
    包含 00_start.json — 它是步骤 1 执行前的初始状态感知源。
    """
    results = []
    for fname in os.listdir(traj_dir):
        if fname in ("history.json", "metadata.json"):
            continue
        m = STEP_FILE_RE.match(fname)
        if not m:
            continue
        results.append((fname, int(m.group(1)), m.group(2)))
    results.sort(key=lambda x: x[1])
    return results


def resolve_data_path(base_dir: str, rel_path: str) -> str:
    """将 config/prompt 中的相对路径解析为绝对路径"""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(base_dir, rel_path))
