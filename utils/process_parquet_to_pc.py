import sys
import os
import argparse
import glob
import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))


def get_args():
    parser = argparse.ArgumentParser(description="Process Parquet depth and MP4 RGB into Pointclouds.")
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        required=True, 
        help="Path to the dataset root, e.g., /path/to/lehome/Datasets/record/001"
    )
    parser.add_argument(
        "--num_points", 
        type=int, 
        default=4096, 
        help="Number of points per pointcloud"
    )
    return parser.parse_args()

def main():
    args = get_args()
    dataset_path = Path(args.dataset_root)

    # ==========================================
    # 2. Locate RGB Video File
    # ==========================================
    # Assumes this video contains frames for ALL episodes sequentially
    video_path = dataset_path / "videos" / "observation.images.top_rgb" / "chunk-000" / "file-000.mp4"
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Found Master Video: {video_path}")

    # ==========================================
    # 3. Locate Parquet Files
    # ==========================================
    parquet_dir = dataset_path / "data" / "chunk-000"
    if not parquet_dir.exists():
        print(f"Error: Parquet directory not found at {parquet_dir}")
        return

    parquet_files = sorted(glob.glob(str(parquet_dir / "file-*.parquet")))
    
    if not parquet_files:
        print("No parquet files found.")
        return

    print(f"Found {len(parquet_files)} episode parquet files.")

    # ==========================================
    # 4. Initialize Video Stream Iterator (using imiter)
    # ==========================================
    video_reader = iio.imiter(str(video_path), plugin="pyav")
    video_iterator = iter(video_reader)

    # ==========================================
    # 5. Processing Loop
    # ==========================================
    total_frames_processed = 0

    for ep_idx, pq_file in enumerate(tqdm(parquet_files, desc="Processing Episodes")):
        
        # --- Read Depth ---
        try:
            table = pq.read_table(pq_file)
            pydict = table.to_pydict()
            
            if "observation.top_depth" not in pydict:
                print(f"Skipping {pq_file}: 'observation.top_depth' key not found.")
                continue
                
            depth_list = pydict["observation.top_depth"]
            num_frames_in_episode = len(depth_list)
            
        except Exception as e:
            print(f"Error reading parquet {pq_file}: {e}")
            continue

        # --- Prepare Output Path ---
        output_dir = dataset_path / "pointclouds" / f"episode_{ep_idx:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Process by Frame ---
        for frame_idx in range(num_frames_in_episode):
            try:
                rgb_frame = next(video_iterator)
                
                depth_data = depth_list[frame_idx]
                
                if isinstance(depth_data, list):
                    depth_frame = np.array(depth_data, dtype=np.float32)
                else:
                    depth_frame = depth_data
                
                # Make Sure Shape is (480, 640)
                if depth_frame.ndim == 1:
                    if depth_frame.size == 480 * 640:
                        depth_frame = depth_frame.reshape((480, 640))
                    else:
                        print(f"Error: Depth frame size {depth_frame.size} does not match 480x640")
                        continue
                
                # Since the depth information was stored in uint16 and in mm
                depth_frame = depth_frame.astype(np.float32) / 1000.0

                pointclouds_with_color = generate_pointcloud_from_data(
                    rgb_image=rgb_frame,
                    depth_image=depth_frame,
                    num_points=args.num_points,
                    use_fps=True
                )

                save_path = output_dir / f"frame_{frame_idx:06d}.npz"
                np.savez_compressed(str(save_path), pointcloud=pointclouds_with_color)
                
                total_frames_processed += 1

            except StopIteration:
                print(f"Error: Video ended unexpectedly at Episode {ep_idx}, Frame {frame_idx}!")
                break
            except Exception as e:
                print(f"Error processing Ep {ep_idx} Frame {frame_idx}: {e}")
                continue

    print(f"\nAll done! Processed {total_frames_processed} frames across {len(parquet_files)} episodes.")

if __name__ == "__main__":
    main()
    