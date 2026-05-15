import numpy as np

def analyze_pose():
    # 把 pose.txt 路径改成你实际路径
    pose_path = '/home/liu/workspace/recorded_scene/recorded_scene/20260515_170340/vggt_res/poses.txt'
    # 云端用: pose_path = '/root/liu_ws/holoagent_data/0515_v1/20260515_120546/pose.txt'

    poses = np.loadtxt(pose_path)
    ts, tx, ty, tz = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]

    print(f"总帧数: {len(poses)}")
    print(f"tx range: {tx.min():.4f} ~ {tx.max():.4f}, std: {tx.std():.4f}")
    print(f"ty range: {ty.min():.4f} ~ {ty.max():.4f}, std: {ty.std():.4f}")
    print(f"tz range: {tz.min():.4f} ~ {tz.max():.4f}, std: {tz.std():.4f}")

    variances = {'x': tx.std(), 'y': ty.std(), 'z': tz.std()}
    min_axis = min(variances, key=variances.get)
    max_axis = max(variances, key=variances.get)
    print(f"\n最小位移轴 (高度): {min_axis} (std={variances[min_axis]:.4f})")
    print(f"最大位移轴 (水平): {max_axis} (std={variances[max_axis]:.4f})")

def check_pcd_range():
    import open3d as o3d
    pcd = o3d.io.read_point_cloud('/home/liu/workspace/recorded_scene/icra_ic3f/fsr/full_pcd.ply')
    pts = np.asarray(pcd.points)
    print('X range:', pts[:,0].min(), pts[:,0].max())
    print('Y range:', pts[:,1].min(), pts[:,1].max())
    print('Z range:', pts[:,2].min(), pts[:,2].max())
