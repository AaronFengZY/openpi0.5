#!/usr/bin/env python3
"""
test_fk_direct_indexed.py
──────────────────────
结合了 "Index 查找" 和 "Direct 读取" 的优点。
1. 读取 index.npy 获取元数据。
2. 动态处理视频分块 (Chunking) 逻辑 (每500帧一个文件)。
3. 直接读取 action.npy (全量)，绕过 Dataset 类进行 FK 验证。
"""

import argparse
import cv2
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy

# 引入项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..")) 

# 尝试导入 AgibotActionState 和 FK
try:
    from openpi.training.action import AgibotActionState
    from g1_fk import G1_FK
    print("✅ Successfully imported Modules")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# ================= 配置 =================
HARDCODED_DIM_LIST = [2, 14, 2, 2, 2, 8, 6, 2, 14, 2, 2]
TOTAL_DIM = sum(HARDCODED_DIM_LIST) 
BYTES_PER_ROW = TOTAL_DIM * 4       
VIDEO_CHUNK_SIZE = 500  # 每个视频文件的帧数

def load_camera_params(npy_path, lookup_key):
    if not os.path.exists(npy_path): return None, None, None, None
    data = np.load(npy_path, allow_pickle=True).item()
    if lookup_key not in data['camera2intrinsic']: return None, None, None, None

    intr_raw = data['camera2intrinsic'][lookup_key]
    fx, fy, cx, cy = intr_raw
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((1, 5)) 

    ext_raw = data['camera2extrinsic'][lookup_key]
    vals = np.array(ext_raw).flatten()
    t_vec = vals[:3]
    r_obj = R_scipy.from_euler('xyz', vals[3:], degrees=False) 
    R_wc = r_obj.as_matrix()
    
    T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = t_vec
    T_cw = np.linalg.inv(T_wc)
    return K, dist, T_cw[:3, :3], T_cw[:3, 3]

def project_bbox(frame, bbox_3d, Rcw, tcw, K, dist, color, label=""):
    if bbox_3d is None: return
    corners_w = np.array([[bbox_3d[0], bbox_3d[1], bbox_3d[2]], [bbox_3d[0], bbox_3d[1], bbox_3d[5]], 
                          [bbox_3d[0], bbox_3d[4], bbox_3d[2]], [bbox_3d[0], bbox_3d[4], bbox_3d[5]],
                          [bbox_3d[3], bbox_3d[1], bbox_3d[2]], [bbox_3d[3], bbox_3d[1], bbox_3d[5]], 
                          [bbox_3d[3], bbox_3d[4], bbox_3d[2]], [bbox_3d[3], bbox_3d[4], bbox_3d[5]]])
    corners_c = (Rcw @ corners_w.T).T + tcw
    if (corners_c[:, 2] <= 0.01).all(): return 
    pix, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), K, dist)
    pix = pix.reshape(-1, 2)
    x_min, y_min = pix.min(axis=0).astype(int)
    x_max, y_max = pix.max(axis=0).astype(int)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def calculate_projection_debug(bbox_3d, Rcw, tcw, K, dist):
    if bbox_3d is None: return "None"
    corners_w = np.array([[bbox_3d[0], bbox_3d[1], bbox_3d[2]], [bbox_3d[0], bbox_3d[1], bbox_3d[5]], 
                          [bbox_3d[0], bbox_3d[4], bbox_3d[2]], [bbox_3d[0], bbox_3d[4], bbox_3d[5]],
                          [bbox_3d[3], bbox_3d[1], bbox_3d[2]], [bbox_3d[3], bbox_3d[1], bbox_3d[5]], 
                          [bbox_3d[3], bbox_3d[4], bbox_3d[2]], [bbox_3d[3], bbox_3d[4], bbox_3d[5]]])
    corners_c = (Rcw @ corners_w.T).T + tcw
    if (corners_c[:, 2] <= 0.01).all(): return "Behind Camera"
    pix, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), K, dist)
    pix = pix.reshape(-1, 2)
    return f"x[{pix[:,0].min():.1f}, {pix[:,0].max():.1f}], y[{pix[:,1].min():.1f}, {pix[:,1].max():.1f}]"

def main(args):
    # 1. 加载 Index 文件
    print(f"📂 Loading Index File: {args.index_file}")
    if not os.path.exists(args.index_file):
        print("❌ Index file not found!"); return
    
    meta_data = np.load(args.index_file, allow_pickle=True).item()
    video_paths = meta_data["video_path"]
    start_ends = meta_data["start_end"]
    total_samples = len(video_paths)

    # 2. 确定 Index (指定 or 随机)
    if args.index is not None:
        selected_idx = args.index
        if selected_idx < 0 or selected_idx >= total_samples:
            print(f"❌ Index {selected_idx} out of bounds."); return
        print(f"🎯 Mode: Specific Index {selected_idx}")
    else:
        selected_idx = random.randint(0, total_samples - 1)
        print(f"🎲 Mode: Random Index {selected_idx}")

    # 3. 解析元数据
    rel_path = video_paths[selected_idx]  # e.g., "327/648642.mp4"
    rel_path_no_ext = os.path.splitext(rel_path)[0] # "327/648642"
    path_parts = rel_path_no_ext.split('/')
    
    if len(path_parts) >= 2:
        scene_id, episode_id = path_parts[-2], path_parts[-1]
    else:
        scene_id, episode_id = "unknown", os.path.basename(rel_path_no_ext)

    start_frame = start_ends[selected_idx][0] 
    end_frame = start_ends[selected_idx][1]   
    
    # 构建绝对路径
    action_path = os.path.join(args.dataset_root, "actions_gaussian", rel_path_no_ext, "action.npy")
    video_dir = os.path.join(args.dataset_root, "videos_h264", rel_path_no_ext, "videos")

    print("\n" + "="*50)
    print(f"🎬 Meta Info Parsed:")
    print(f"   Scene: {scene_id} | Ep: {episode_id}")
    print(f"   Start Frame: {start_frame} | End Frame: {end_frame}")
    print(f"   Action Path: {action_path}")
    print(f"   Video Dir:   {video_dir}")
    print("="*50 + "\n")

    if not os.path.exists(action_path):
        print("❌ Action file missing!"); return

    # 4. 加载 Action Data (Direct Binary Read)
    try:
        file_size = os.path.getsize(action_path)
        total_steps = file_size // BYTES_PER_ROW
        action_obj = AgibotActionState.load_range_from_path(action_path, HARDCODED_DIM_LIST, 0, total_steps)
        print(f"✅ Action Data Loaded ({total_steps} steps).")
    except Exception as e:
        print(f"❌ Action Load Error: {e}"); return

    # 5. 加载 Camera
    camera_key = f"{scene_id}/{episode_id}_head"
    K, dist, Rcw, tcw = load_camera_params(args.camera_npy, camera_key)
    if K is None: print(f"❌ Camera params missing for {camera_key}"); return

    # 6. 初始化 FK
    fk_solver = G1_FK(args.urdf, package_dirs=os.path.dirname(args.urdf)) if G1_FK else None
    
    # 7. 准备输出
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"direct_viz_{scene_id}_{episode_id}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    # 8. 循环处理 (Dynamic Video Loading)
    # 处理完整的片段长度 (不再截断到240)
    loop_len = end_frame - start_frame
    print(f"🚀 Processing {loop_len} frames (Start: {start_frame} -> End: {end_frame})...")

    # 状态变量
    current_cap = None
    current_chunk_idx = -1

    for i in tqdm(range(loop_len)):
        current_abs_frame = start_frame + i 
        
        # ▼▼▼▼▼▼▼▼▼▼ 视频分块处理逻辑 ▼▼▼▼▼▼▼▼▼▼
        chunk_idx = current_abs_frame // VIDEO_CHUNK_SIZE
        frame_in_chunk = current_abs_frame % VIDEO_CHUNK_SIZE
        
        # 如果切换了 chunk，需要重新打开视频文件
        if chunk_idx != current_chunk_idx:
            if current_cap is not None:
                current_cap.release()
            
            video_name = f"head_color_{chunk_idx}.mp4"
            video_path = os.path.join(video_dir, video_name)
            
            if not os.path.exists(video_path):
                print(f"⚠️ Video chunk not found: {video_path}. Stopping.")
                break
                
            current_cap = cv2.VideoCapture(video_path)
            current_chunk_idx = chunk_idx
            # print(f"🔄 Switched to chunk: {video_name}")

        # 设置帧位置
        current_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_chunk)
        ret, frame = current_cap.read()
        
        if not ret: 
            print(f"⚠️ Failed to read frame {frame_in_chunk} in chunk {chunk_idx}. Stopping.")
            break
            
        frame = cv2.resize(frame, (640, 480))
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # Action Seek
        if current_abs_frame >= total_steps: break
        
        q_arm = action_obj.state_joint_position[current_abs_frame].tolist()
        q_head = action_obj.state_head_position[current_abs_frame].tolist()
        q_waist = action_obj.state_waist_position[current_abs_frame].tolist()

        if fk_solver:
            res = fk_solver.fk(head_position=q_head, waist_position=q_waist, joint_position=q_arm)
            if res:
                # Left -> Left Label, Right -> Right Label
                project_bbox(frame, res.get("left_bbox"), Rcw, tcw, K, dist, (255, 255, 0), "Left")
                project_bbox(frame, res.get("right_bbox"), Rcw, tcw, K, dist, (0, 255, 255), "Right")

                if i == 0:
                    print(f"\n🔎 Frame {current_abs_frame} Check (Chunk {chunk_idx}, Local {frame_in_chunk}):")
                    print(f"   Head: {q_head}")
                    print(f"   Arm[0:3]: {q_arm[:3]}...")
                    print(f"   L_Proj: {calculate_projection_debug(res.get('left_bbox'), Rcw, tcw, K, dist)}")

        writer.write(frame)

    if current_cap: current_cap.release()
    writer.release()
    print(f"✨ Done! Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--camera_npy", type=str, required=True)
    parser.add_argument("--urdf", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_fk_vis")
    parser.add_argument("--index", type=int, default=None, help="Specific index from npy file")
    args = parser.parse_args()
    main(args)