import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objs as go

# ==========================================
# 1. FPS Sampling 
# ==========================================
def farthest_point_sampling_with_color(points, colors, n_samples):
    N, D = points.shape
    if N < n_samples:
        indices = np.random.choice(N, n_samples, replace=True)
        return points[indices], colors[indices]

    xyz = points
    centroids = np.zeros((n_samples,), dtype=int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)

    return points[centroids], colors[centroids]

# ==========================================
# 2. Remove Outliers
# ==========================================
def remove_outliers_statistical(points, colors, nb_neighbors=20, std_ratio=2.0):
    from scipy.spatial import cKDTree
    
    if len(points) == 0:
        return points, colors
        
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors)
    mean_dists = np.mean(dists, axis=1)
    
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    
    return points[mask], colors[mask]

# ==========================================
# 3. Pointcloud Generation
# ==========================================
def generate_pointcloud_from_data(
    rgb_image: np.ndarray, 
    depth_image: np.ndarray, 
    num_points: int = 2048,
    use_fps: bool = True
):
    """
    Inputs:
        rgb_image: (H, W, 3) np.uint8
        depth_image: (H, W) np.float32 (meters, M)
        num_points: number of final output points
        use_fps: whether to use FPS
    Outputs:
        pointcloud: (N, 6) XYZRGB
    """
    
    # Camera Intrinsics
    fx, fy = 482.0, 482.0
    cx, cy = 320.0, 240.0
    
    # =========================================================
    # Transform 1: Camera -> Right Robot Base (Local)
    # =========================================================
    quat_wxyz = [0.1650476, -0.9862856, 0.0, 0.0]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    translation_cam_to_robot = np.array([0.225, -0.5, 0.6])

    img_h, img_w = depth_image.shape
    
    # 1. Acquire valid depth index
    valid_mask = (depth_image > 0) & np.isfinite(depth_image)
    v_idx, u_idx = np.nonzero(valid_mask)
    total_valid = len(v_idx)
    
    if total_valid == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # 2. Pre-sampling
    pre_sample_target = max(8192, num_points * 2)

    if total_valid > pre_sample_target:
        sample_indices = np.random.choice(total_valid, pre_sample_target, replace=False)
        v_sample = v_idx[sample_indices]
        u_sample = u_idx[sample_indices]
    else:
        v_sample = v_idx
        u_sample = u_idx

    # 3. Calculate XYZ
    Z_sample = depth_image[v_sample, u_sample]
    X_sample = (u_sample - cx) * Z_sample / fx
    Y_sample = (v_sample - cy) * Z_sample / fy
    
    points_cam = np.stack([X_sample, Y_sample, Z_sample], axis=1)
    
    # 4. Retrive Color
    if rgb_image.shape[-1] == 4:
        colors_sample = rgb_image[v_sample, u_sample, :3]
    else:
        colors_sample = rgb_image[v_sample, u_sample]

    # 5. Camera -> World
    r_usd_to_base = R.from_quat(quat_xyzw).as_matrix()
    r_optical_to_usd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    r_mix = np.dot(r_usd_to_base, r_optical_to_usd)

    points_local = np.dot(points_cam, r_mix.T) + translation_cam_to_robot
    
    # =========================================================
    # [NEW] Transform 2: Right Robot Base -> World
    # =========================================================
    t_robot_to_world = np.array([0.23, -0.25, 0.5])
    r_robot_to_world = R.from_euler('z', 180, degrees=True).as_matrix()
    
    points_world = np.dot(points_local, r_robot_to_world.T) + t_robot_to_world

    # 6. Statistic Filter Denoising
    points_world, colors_sample = remove_outliers_statistical(
        points_world, colors_sample, nb_neighbors=50, std_ratio=1.0
    )

    # 7. FPS or Random
    if points_world.shape[0] > num_points:
        if use_fps:
            points_world, colors_sample = farthest_point_sampling_with_color(
                points_world, colors_sample, num_points
            )
        else:
            indices = np.random.choice(points_world.shape[0], num_points, replace=False)
            points_world = points_world[indices]
            colors_sample = colors_sample[indices]
            
    colors_norm = colors_sample.astype(np.float32) / 255.0
    pointcloud = np.concatenate([points_world, colors_norm], axis=1)
    
    return pointcloud
