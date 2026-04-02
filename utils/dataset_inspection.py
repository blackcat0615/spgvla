"""Dataset inspection utilities for LeRobot datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
import pyarrow.parquet as pq


def print_separator(char: str = "=", width: int = 80) -> None:
    """Print a separator line for better output formatting."""
    print(char * width)


def print_meta_info(dataset_root: Path) -> Optional[Dict[str, Any]]:
    """Display dataset metadata information including garment_info.json."""
    print_separator()
    print("📊 Dataset Metadata")
    print_separator()

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"

    if not info_path.exists():
        print(f"❌ meta/info.json not found: {info_path}")
        return None

    try:
        with info_path.open("r") as f:
            info = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading info.json: {e}")
        return None

    print(f"📁 Dataset path: {dataset_root}")
    print(f"🎬 Total episodes: {info.get('total_episodes', 'N/A')}")
    print(f"🎞️  Total frames: {info.get('total_frames', 'N/A')}")
    print(f"🎥 FPS: {info.get('fps', 'N/A')}")
    print(f"📦 Chunk size: {info.get('chunks_size', 'N/A')}")

    garment_info_path = meta_dir / "garment_info.json"
    if garment_info_path.exists():
        try:
            with garment_info_path.open("r", encoding="utf-8") as f:
                garment_info = json.load(f)

            print_separator("-")
            print("👕 Garment Information")
            print_separator("-")

            total_episodes_in_garment_info = 0
            for garment_name, episodes in garment_info.items():
                episode_count = len(episodes)
                total_episodes_in_garment_info += episode_count
                print(f"  • {garment_name}: {episode_count} episode(s)")

                if episodes:
                    first_ep_key = sorted(episodes.keys(), key=int)[0]
                    first_ep = episodes[first_ep_key]
                    info_parts = []

                    if "object_initial_pose" in first_ep:
                        pose = first_ep["object_initial_pose"]
                        if isinstance(pose, dict) and "translation" in pose:
                            trans = pose["translation"]
                            info_parts.append(
                                f"pose: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]"
                            )
                        else:
                            info_parts.append("pose: available")

                    if "scale" in first_ep:
                        scale = first_ep["scale"]
                        if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                            info_parts.append(
                                f"scale: [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]"
                            )
                        else:
                            info_parts.append("scale: available")

                    if info_parts:
                        print(
                            f"    Sample (episode {first_ep_key}): {', '.join(info_parts)}"
                        )

            print(
                f"\n  Total episodes in garment_info: {total_episodes_in_garment_info}"
            )

            total_episodes_info = info.get("total_episodes", 0)
            if total_episodes_in_garment_info != total_episodes_info:
                print(f"  ⚠️  Warning: Episode count mismatch!")
                print(
                    f"     info.json: {total_episodes_info}, garment_info.json: {total_episodes_in_garment_info}"
                )
            else:
                print(f"  ✅ Episode count matches info.json")

        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse garment_info.json: {e}")
        except Exception as e:
            print(f"  ❌ Error reading garment_info.json: {e}")
    else:
        print_separator("-")
        print("⚠️  garment_info.json not found (no initial pose information)")

    return info


def print_features(info: Dict[str, Any]) -> None:
    """Display dataset features grouped by type."""
    print_separator()
    print("🔍 Dataset Features")
    print_separator()

    if not isinstance(info, dict):
        print("❌ Invalid info dictionary")
        return

    features = info.get("features", {})
    if not features:
        print("⚠️  No features found in dataset metadata")
        return

    observation_feats: List[Tuple[str, Dict[str, Any]]] = []
    action_feats: List[Tuple[str, Dict[str, Any]]] = []
    system_feats: List[Tuple[str, Dict[str, Any]]] = []

    for name, feat in features.items():
        if not isinstance(feat, dict):
            continue
        if name.startswith("observation."):
            observation_feats.append((name, feat))
        elif name.startswith("action"):
            action_feats.append((name, feat))
        else:
            system_feats.append((name, feat))

    def print_feature(name: str, feat: Dict[str, Any]) -> None:
        """Print a single feature with formatted information."""
        dtype = feat.get("dtype", "N/A")
        shape = feat.get("shape", [])
        names = feat.get("names", None)

        shape_str = f"shape={tuple(shape)}" if shape else "scalar"
        names_str = (
            f", names={names[:3]}..."
            if names and len(names) > 3
            else f", names={names}" if names else ""
        )

        print(f"  • {name:30s} [{dtype:8s}] {shape_str}{names_str}")

    if observation_feats:
        print("📷 Observation Features:")
        for name, feat in observation_feats:
            print_feature(name, feat)

    if action_feats:
        print("\n🎮 Action Features:")
        for name, feat in action_feats:
            print_feature(name, feat)

    if system_feats:
        print("\n⚙️  System Features:")
        for name, feat in system_feats:
            print_feature(name, feat)

    has_obs_ee = "observation.ee_pose" in features
    has_act_ee = "action.ee_pose" in features

    print_separator("-")
    if has_obs_ee and has_act_ee:
        print("✅ Dataset contains both observation.ee_pose and action.ee_pose")
    elif has_obs_ee:
        print("⚠️  Dataset only contains observation.ee_pose, missing action.ee_pose")
    elif has_act_ee:
        print("⚠️  Dataset only contains action.ee_pose, missing observation.ee_pose")
    else:
        print("❌ Dataset does not contain end-effector pose information (ee_pose)")


def print_sample_frames(dataset_root: Path, num_frames: int = 3) -> None:
    """Display sample frames from the dataset."""
    if num_frames <= 0:
        print("⚠️  Invalid num_frames: must be positive. Using default value 3.")
        num_frames = 3

    print_separator()
    print(f"📋 Sample Data (first {num_frames} frames)")
    print_separator()

    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        print("❌ No parquet files found")
        return

    first_file = parquet_files[0]
    print(f"📄 File: {first_file.relative_to(dataset_root)}")

    try:
        table = pq.read_table(first_file)
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        return

    if table.num_rows == 0:
        print("⚠️  Parquet file is empty")
        return

    key_columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "action",
        "observation.ee_pose",
        "action.ee_pose",
    ]
    available_cols = [col for col in key_columns if col in table.column_names]

    if not available_cols:
        print("⚠️  No key columns found in the dataset")
        return

    print(f"\nColumns: {', '.join(available_cols)}\n")

    try:
        for i in range(min(num_frames, table.num_rows)):
            print(f"--- Frame {i} ---")
            for col in available_cols:
                try:
                    value = table[col][i].as_py()
                    if isinstance(value, list):
                        if len(value) <= 8:
                            value_str = (
                                "["
                                + ", ".join(
                                    f"{v:.4f}" if isinstance(v, float) else str(v)
                                    for v in value
                                )
                                + "]"
                            )
                        else:
                            value_str = f"[{len(value)} elements]"
                    else:
                        value_str = str(value)
                    print(f"  {col:25s}: {value_str}")
                except Exception as e:
                    print(f"  {col:25s}: <Error reading value: {e}>")
            print()
    except Exception as e:
        print(f"❌ Error displaying frames: {e}")


def print_column_stats(dataset_root: Path) -> None:
    """Display statistical information for numeric columns.

    Note: Currently only reads the first parquet file for performance.
    """
    print_separator()
    print("📈 Numeric Column Statistics")
    print_separator()

    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        print("❌ No parquet files found")
        return

    first_file = parquet_files[0]
    try:
        table = pq.read_table(first_file)
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        return

    if table.num_rows == 0:
        print("⚠️  Parquet file is empty")
        return

    numeric_cols = [
        "observation.state",
        "action",
        "observation.ee_pose",
        "action.ee_pose",
    ]
    available_cols = [col for col in numeric_cols if col in table.column_names]

    if not available_cols:
        print("⚠️  No numeric columns found in the dataset")
        return

    for col in available_cols:
        try:
            data = table[col].to_pylist()
            if not data:
                print(f"\n{col}: <Empty column>")
                continue

            try:
                arr = np.array(data, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"\n{col}: <Cannot convert to numeric array: {e}>")
                continue

            if arr.size == 0:
                print(f"\n{col}: <Empty array>")
                continue

            print(f"\n{col}:")
            print(f"  Shape: {arr.shape}")
            print(f"  Mean:  {np.mean(arr, axis=0)}")
            print(f"  Std:   {np.std(arr, axis=0)}")
            print(f"  Min:   {np.min(arr, axis=0)}")
            print(f"  Max:   {np.max(arr, axis=0)}")
        except Exception as e:
            print(f"\n{col}: <Error processing column: {e}>")


def inspect(
    dataset_root: Path, show_frames: Optional[int] = None, show_stats: bool = False
) -> None:
    """Main inspection function."""
    dataset_root = Path(dataset_root).resolve()

    if not dataset_root.exists():
        print(f"❌ Dataset path does not exist: {dataset_root}")
        return

    info = print_meta_info(dataset_root)

    if info:
        print_features(info)

        if show_frames:
            print_sample_frames(dataset_root, show_frames)

        if show_stats:
            print_column_stats(dataset_root)

        print_separator()
        print("✅ Inspection completed")


def load_dataset_info(dataset_root: Path) -> Optional[Dict[str, Any]]:
    """Load dataset metadata information."""
    meta_path = dataset_root / "meta" / "info.json"
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r") as f:
            info = json.load(f)
        return info
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON file: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error reading info.json: {e}")
        return None


def load_parquet_data(
    dataset_root: Path, episode_idx: Optional[int] = None
) -> Dict[str, List[Any]]:
    """Load parquet data from dataset."""
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"Parquet files not found: {data_root}")

    all_data: Dict[str, List[Any]] = {
        "episode_index": [],
        "frame_index": [],
        "observation.state": [],
        "action": [],
        "observation.ee_pose": [],
        "action.ee_pose": [],
    }

    for pf in parquet_files:
        try:
            table = pq.read_table(pf)

            if episode_idx is not None and "episode_index" in table.column_names:
                episode_indices = table["episode_index"].to_pylist()
                mask = [idx == episode_idx for idx in episode_indices]
                if not any(mask):
                    continue
                indices = [i for i, m in enumerate(mask) if m]
                table = table.take(indices)

            for key in all_data.keys():
                if key in table.column_names:
                    all_data[key].extend(table[key].to_pylist())

        except Exception as e:
            print(f"⚠️  Skipping corrupted file {pf}: {e}")
            continue

    return all_data


def print_frame_data(
    data: Dict[str, List[Any]], start_idx: int = 0, num_frames: int = 5
) -> None:
    """Display frame data with formatted output."""
    total = len(data["observation.state"])
    if total == 0:
        print("⚠️  No data available to display")
        return

    end_idx = min(start_idx + num_frames, total)

    print(f"\nDisplaying frames {start_idx} - {end_idx - 1} (total: {total} frames)\n")
    print("=" * 100)

    for i in range(start_idx, end_idx):
        print(f"Frame {i}:")

        if i < len(data["episode_index"]) and i < len(data["frame_index"]):
            print(
                f"  Episode: {data['episode_index'][i]}, Frame: {data['frame_index'][i]}"
            )

        if i < len(data["observation.state"]):
            try:
                obs_state = np.array(data["observation.state"][i])
                if len(obs_state) > 0:
                    print(
                        f"  observation.state [{len(obs_state)}]: {obs_state[:3]} ... {obs_state[-1]}"
                    )
                else:
                    print(f"  observation.state: <empty>")
            except Exception as e:
                print(f"  observation.state: <Error: {e}>")

        if i < len(data["action"]):
            try:
                action = np.array(data["action"][i])
                if len(action) > 0:
                    print(
                        f"  action           [{len(action)}]: {action[:3]} ... {action[-1]}"
                    )
                else:
                    print(f"  action: <empty>")
            except Exception as e:
                print(f"  action: <Error: {e}>")

        if (
            i < len(data["observation.ee_pose"])
            and data["observation.ee_pose"][i] is not None
        ):
            try:
                obs_ee = np.array(data["observation.ee_pose"][i])
                if len(obs_ee) == 8:
                    print(
                        f"  obs.ee_pose [8]: pos=[{obs_ee[0]:.3f},{obs_ee[1]:.3f},{obs_ee[2]:.3f}], grip={obs_ee[7]:.3f}"
                    )
                elif len(obs_ee) == 16:
                    print(f"  obs.ee_pose [16]:")
                    print(
                        f"    Left arm: pos=[{obs_ee[0]:.3f},{obs_ee[1]:.3f},{obs_ee[2]:.3f}], grip={obs_ee[7]:.3f}"
                    )
                    print(
                        f"    Right arm: pos=[{obs_ee[8]:.3f},{obs_ee[9]:.3f},{obs_ee[10]:.3f}], grip={obs_ee[15]:.3f}"
                    )
                else:
                    print(f"  obs.ee_pose [{len(obs_ee)}]: <unexpected dimension>")
            except Exception as e:
                print(f"  obs.ee_pose: <Error: {e}>")
        else:
            print(f"  obs.ee_pose: Not recorded")

        if i < len(data["action.ee_pose"]) and data["action.ee_pose"][i] is not None:
            try:
                act_ee = np.array(data["action.ee_pose"][i])
                if len(act_ee) == 8:
                    print(
                        f"  act.ee_pose [8]: pos=[{act_ee[0]:.3f},{act_ee[1]:.3f},{act_ee[2]:.3f}], grip={act_ee[7]:.3f}"
                    )
                elif len(act_ee) == 16:
                    print(f"  act.ee_pose [16]:")
                    print(
                        f"    Left arm: pos=[{act_ee[0]:.3f},{act_ee[1]:.3f},{act_ee[2]:.3f}], grip={act_ee[7]:.3f}"
                    )
                    print(
                        f"    Right arm: pos=[{act_ee[8]:.3f},{act_ee[9]:.3f},{act_ee[10]:.3f}], grip={act_ee[15]:.3f}"
                    )
                else:
                    print(f"  act.ee_pose [{len(act_ee)}]: <unexpected dimension>")
            except Exception as e:
                print(f"  act.ee_pose: <Error: {e}>")
        else:
            print(f"  act.ee_pose: Not recorded")

        print()


def print_statistics(data: Dict[str, List[Any]]) -> None:
    """Display statistical information for the dataset."""
    print("\n" + "=" * 100)
    print("Dataset Statistics")
    print("=" * 100)

    total_frames = len(data["observation.state"])
    if total_frames == 0:
        print("⚠️  No data available for statistics")
        return

    unique_episodes = len(set(data["episode_index"])) if data["episode_index"] else 1
    if unique_episodes == 0:
        unique_episodes = 1

    print(f"\nBasic Information:")
    print(f"  Total frames: {total_frames}")
    print(f"  Number of episodes: {unique_episodes}")
    print(f"  Average frames per episode: {total_frames / unique_episodes:.1f}")

    try:
        obs_state_arr = np.array(data["observation.state"])
        if obs_state_arr.size > 0:
            print(f"\nobservation.state [{obs_state_arr.shape[1]} dimensions]:")
            print(f"  Mean: {np.mean(obs_state_arr, axis=0)[:3]} ...")
            print(f"  Std:  {np.std(obs_state_arr, axis=0)[:3]} ...")
            print(f"  Min:  {np.min(obs_state_arr, axis=0)[:3]} ...")
            print(f"  Max:  {np.max(obs_state_arr, axis=0)[:3]} ...")
    except Exception as e:
        print(f"\nobservation.state: <Error computing statistics: {e}>")

    try:
        action_arr = np.array(data["action"])
        if action_arr.size > 0:
            print(f"\naction [{action_arr.shape[1]} dimensions]:")
            print(f"  Mean: {np.mean(action_arr, axis=0)[:3]} ...")
            print(f"  Std:  {np.std(action_arr, axis=0)[:3]} ...")
    except Exception as e:
        print(f"\naction: <Error computing statistics: {e}>")

    if (
        data["observation.ee_pose"]
        and len(data["observation.ee_pose"]) > 0
        and data["observation.ee_pose"][0] is not None
    ):
        try:
            obs_ee_arr = np.array(data["observation.ee_pose"])
            if obs_ee_arr.size > 0:
                print(f"\nobservation.ee_pose [{obs_ee_arr.shape[1]} dimensions]:")
                if obs_ee_arr.shape[1] == 8:
                    print(
                        f"  Position range: X=[{obs_ee_arr[:, 0].min():.3f}, {obs_ee_arr[:, 0].max():.3f}] m"
                    )
                    print(
                        f"                  Y=[{obs_ee_arr[:, 1].min():.3f}, {obs_ee_arr[:, 1].max():.3f}] m"
                    )
                    print(
                        f"                  Z=[{obs_ee_arr[:, 2].min():.3f}, {obs_ee_arr[:, 2].max():.3f}] m"
                    )
                    print(
                        f"  Gripper range: [{obs_ee_arr[:, 7].min():.3f}, {obs_ee_arr[:, 7].max():.3f}]"
                    )
                elif obs_ee_arr.shape[1] == 16:
                    print(
                        f"  Left arm position: X=[{obs_ee_arr[:, 0].min():.3f}, {obs_ee_arr[:, 0].max():.3f}] m"
                    )
                    print(
                        f"                     Y=[{obs_ee_arr[:, 1].min():.3f}, {obs_ee_arr[:, 1].max():.3f}] m"
                    )
                    print(
                        f"                     Z=[{obs_ee_arr[:, 2].min():.3f}, {obs_ee_arr[:, 2].max():.3f}] m"
                    )
                    print(
                        f"  Right arm position: X=[{obs_ee_arr[:, 8].min():.3f}, {obs_ee_arr[:, 8].max():.3f}] m"
                    )
                    print(
                        f"                      Y=[{obs_ee_arr[:, 9].min():.3f}, {obs_ee_arr[:, 9].max():.3f}] m"
                    )
                    print(
                        f"                      Z=[{obs_ee_arr[:, 10].min():.3f}, {obs_ee_arr[:, 10].max():.3f}] m"
                    )
        except Exception as e:
            print(f"\nobservation.ee_pose: <Error computing statistics: {e}>")
    else:
        print(f"\nobservation.ee_pose: Not recorded")


def export_to_csv(data: Dict[str, List[Any]], output_path: str) -> None:
    """Export dataset to CSV file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for CSV export: pip install pandas")

    rows = []
    for i in range(len(data["observation.state"])):
        row: Dict[str, Any] = {}

        if i < len(data["episode_index"]):
            row["episode_index"] = data["episode_index"][i]
            row["frame_index"] = data["frame_index"][i]

        obs_state = data["observation.state"][i]
        for j, val in enumerate(obs_state):
            row[f"obs_state_{j}"] = val

        action = data["action"][i]
        for j, val in enumerate(action):
            row[f"action_{j}"] = val

        if (
            i < len(data["observation.ee_pose"])
            and data["observation.ee_pose"][i] is not None
        ):
            obs_ee = data["observation.ee_pose"][i]
            for j, val in enumerate(obs_ee):
                row[f"obs_ee_{j}"] = val

        if i < len(data["action.ee_pose"]) and data["action.ee_pose"][i] is not None:
            act_ee = data["action.ee_pose"][i]
            for j, val in enumerate(act_ee):
                row[f"act_ee_{j}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data exported to: {output_path}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")


def read_states(
    dataset_root: Path,
    num_frames: Optional[int] = None,
    episode: Optional[int] = None,
    output_csv: Optional[str] = None,
    show_stats: bool = False,
) -> None:
    """Read and analyze dataset observation/action data."""
    dataset_root = Path(dataset_root).resolve()

    if not dataset_root.exists():
        print(f"❌ Dataset path does not exist: {dataset_root}")
        return

    print(f"📂 Dataset: {dataset_root}")

    info = load_dataset_info(dataset_root)
    if info:
        print(
            f"✓ Total episodes: {info.get('total_episodes', 'N/A')}, Total frames: {info.get('total_frames', 'N/A')}"
        )

        has_ee = "observation.ee_pose" in info.get("features", {})
        if has_ee:
            ee_shape = info["features"]["observation.ee_pose"]["shape"]
            print(f"✓ Contains end-effector pose data ({ee_shape[0]} dimensions)")
        else:
            print(f"⚠️  Does not contain end-effector pose data")
    else:
        print(f"⚠️  Failed to read metadata or metadata file not found")

    print(f"\nLoading data...")
    try:
        data = load_parquet_data(dataset_root, episode)
        total_loaded = len(data["observation.state"])
        if total_loaded == 0:
            print("⚠️  No data loaded from dataset")
            return
        print(f"✓ Loaded {total_loaded} frames")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    if num_frames and num_frames > 0:
        print_frame_data(data, 0, num_frames)

    if show_stats:
        print_statistics(data)

    if output_csv:
        try:
            export_to_csv(data, output_csv)
        except ImportError:
            print("❌ pandas required for CSV export: pip install pandas")
        except Exception as e:
            print(f"❌ Export failed: {e}")
