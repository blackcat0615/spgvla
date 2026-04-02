from pathlib import Path
import time
import json
import numpy as np

# Try to import OmegaConf for handling ListConfig/DictConfig
try:
    from omegaconf import ListConfig, DictConfig, OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False
    OmegaConf = None


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def get_next_experiment_path_with_gap(base_path: Path) -> Path:
    """Find the first available number (including open positions)"""
    base_path.mkdir(parents=True, exist_ok=True)

    # collect existing indices
    indices = set()
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                indices.add(int(folder.name))
            except ValueError:
                continue

    # find the first available index
    folder_index = 1
    while folder_index in indices:
        folder_index += 1

    return base_path / f"{folder_index:03d}"


def _ndarray_to_list(obj):
    """Recursively convert numpy arrays, OmegaConf objects to JSON-serializable types."""
    # Handle OmegaConf types first (before checking for dict/list)
    if HAS_OMEGACONF and isinstance(obj, (ListConfig, DictConfig)):
        # Convert OmegaConf objects to native Python types
        obj = OmegaConf.to_container(obj, resolve=True)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    else:
        return obj


def append_episode_initial_pose(
    json_path, 
    episode_idx, 
    object_initial_pose,
    garment_name=None,
    scale=None,
):
    """
    Append initial pose information to the JSON file with hierarchical structure.
    
    This function saves the initial pose of objects after each environment reset.
    The format uses nested dictionaries for easy alignment and lookup.
    
    Format:
    {
      "Top_Long_Unseen_0": {
        "0": {
          "object_initial_pose": [...],
          "scale": [...]  // optional
        },
        "1": {
          "object_initial_pose": [...],
          "scale": [...]
        }
      }
    }
    
    Args:
        json_path: Path to the JSON file
        episode_idx: Episode index
        object_initial_pose: Initial pose dictionary from env.get_all_pose() after reset
                          (e.g., {"Garment": [x, y, z, roll, pitch, yaw]})
        garment_name: Optional garment name (e.g., "Top_Long_Seen_0")
        scale: Optional scale information (list or array)
    """
    if garment_name is None:
        garment_name = "unknown"
    
    # Extract pose list from dictionary if needed
    # object_initial_pose is typically {"Garment": [x, y, z, roll, pitch, yaw]}
    if isinstance(object_initial_pose, dict):
        # Extract the pose list from the dictionary (typically "Garment" key)
        if "Garment" in object_initial_pose:
            pose_list = object_initial_pose["Garment"]
        else:
            # If no "Garment" key, try to get the first value
            pose_list = list(object_initial_pose.values())[0] if object_initial_pose else None
    else:
        # If already a list, use it directly
        pose_list = object_initial_pose
    
    # Convert to list format (handle numpy arrays, etc.)
    pose_list = _ndarray_to_list(pose_list)
    scale_list = _ndarray_to_list(scale) if scale is not None else None
    
    # Create episode record
    episode_rec = {
        "object_initial_pose": pose_list,
    }
    if scale_list is not None:
        episode_rec["scale"] = scale_list
    
    # Read existing data
    json_path = Path(json_path)
    data = {}
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fin:
                data = json.load(fin)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is invalid or doesn't exist, start with empty dict
            data = {}
    
    # Update or create garment entry
    if garment_name not in data:
        data[garment_name] = {}
    
    # Use string key for episode index for better alignment
    episode_key = str(episode_idx)
    data[garment_name][episode_key] = episode_rec
    
    # Write back to file with indentation for readability
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)
