import argparse
import asyncio
import json
import os
import sys
import re
import cv2
import msgpack
import msgpack_numpy as m
import numpy as np
import websockets
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt

# ==========================================
# 0. 环境配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "src"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入 AgibotActionState
try:
    from openpi.training.action import AgibotActionState
    print("✅ Successfully imported AgibotActionState")
except ImportError:
    print("❌ ImportError: Cannot find 'openpi.training.action'.")
    sys.exit(1)

# 导入 FK 模块 (G1_FK)
try:
    from g1_fk import G1_FK
    print("✅ Successfully imported G1_FK Module")
except ImportError:
    G1_FK = None
    print("⚠️ Warning: G1_FK module not found. Visualization will skip FK.")

m.patch()

# ==========================================
# 1. 辅助函数
# ==========================================

def get_gradient_color(step_idx, total_steps):
    """
    Pred 专用渐变：从 浅黄色 (近期) -> 深红色 (远期)
    """
    if total_steps <= 1: return (0, 0, 255) 
    ratio = step_idx / (total_steps - 1)
    start_bgr = np.array([0, 255, 255]) 
    end_bgr   = np.array([0, 0, 255])
    color = (1 - ratio) * start_bgr + ratio * end_bgr
    return tuple(map(int, color))

def load_camera_params(npy_path, lookup_key):
    if not os.path.exists(npy_path): return None, None, None, None
    data = np.load(npy_path, allow_pickle=True).item()
    
    target_data = None
    if lookup_key in data.get('camera2intrinsic', {}):
        target_data = lookup_key
    else:
        base_key = lookup_key.replace("_head", "")
        for k in data.get('camera2intrinsic', {}):
            if base_key in k and "head" in k:
                target_data = k
                break
    
    if not target_data: return None, None, None, None

    intr_raw = data['camera2intrinsic'][target_data]
    fx, fy, cx, cy = intr_raw
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((1, 5)) 

    ext_raw = data['camera2extrinsic'][target_data]
    vals = np.array(ext_raw).flatten()
    r_obj = R_scipy.from_euler('xyz', vals[3:], degrees=False) 
    R_wc = r_obj.as_matrix()
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc; T_wc[:3, 3] = vals[:3]
    T_cw = np.linalg.inv(T_wc)
    
    return K, dist, T_cw[:3, :3], T_cw[:3, 3]

def project_points_3d(frame, points_3d, Rcw, tcw, K, dist, color, radius=3):
    """
    投影一组 3D 点并在图像上画圆点
    points_3d: (N, 3) numpy array
    Returns: frame, center_of_mass(x,y)
    """
    if points_3d is None or len(points_3d) == 0: return frame, None
    
    # 投影
    corners_c = (Rcw @ points_3d.T).T + tcw
    
    # 简单的 Z-clip (过滤掉相机背后的点)
    if (corners_c[:, 2] <= 0.01).all(): return frame, None
    
    pix, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), K, dist)
    pix = pix.reshape(-1, 2).astype(int)
    
    h, w = frame.shape[:2]
    
    valid_points = []
    for p in pix:
        x, y = p
        # 只画在画面内的点
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), radius, color, -1) # -1 filled circle
            valid_points.append(p)
    
    # 计算重心作为连线的中心点
    if valid_points:
        valid_points = np.array(valid_points)
        center_x = int(np.mean(valid_points[:, 0]))
        center_y = int(np.mean(valid_points[:, 1]))
        return frame, (center_x, center_y)
    
    return frame, None

def project_bbox(frame, bbox_3d, Rcw, tcw, K, dist, color, thickness=2):
    """ 投影 3D BBox 到 2D 图像 """
    if bbox_3d is None: return frame, None
    
    corners_w = np.array([
        [bbox_3d[0], bbox_3d[1], bbox_3d[2]], [bbox_3d[0], bbox_3d[1], bbox_3d[5]], 
        [bbox_3d[0], bbox_3d[4], bbox_3d[2]], [bbox_3d[0], bbox_3d[4], bbox_3d[5]],
        [bbox_3d[3], bbox_3d[1], bbox_3d[2]], [bbox_3d[3], bbox_3d[1], bbox_3d[5]], 
        [bbox_3d[3], bbox_3d[4], bbox_3d[2]], [bbox_3d[3], bbox_3d[4], bbox_3d[5]]
    ])
    corners_c = (Rcw @ corners_w.T).T + tcw
    
    if (corners_c[:, 2] <= 0.01).all(): return frame, None
    
    pix, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), K, dist)
    pix = pix.reshape(-1, 2)
    
    x_min, y_min = pix.min(axis=0).astype(int)
    x_max, y_max = pix.max(axis=0).astype(int)
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (max(0, x_min), max(0, y_min)), (min(w, x_max), min(h, y_max)), color, thickness)
    
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    
    return frame, (center_x, center_y)

def get_bbox_center_3d(bbox):
    """ 计算 3D BBox 的中心点 """
    if bbox is None: return None
    # bbox assumed to be [x_min, y_min, z_min, x_max, y_max, z_max]
    return np.array([
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0
    ])

# ==========================================
# 2. 多视角视频加载器
# ==========================================
class MultiViewVideoLoader:
    def __init__(self, data_root, episode_rel_path):
        self.video_dir = os.path.join(data_root, episode_rel_path, "videos") 
        if not os.path.exists(self.video_dir):
            self.video_dir = os.path.join(data_root, episode_rel_path)

        # 必须包含 3 个视角，否则 Server 报错
        self.views_map = {
            "base_0_rgb": "head_color",
            "left_wrist_0_rgb": "hand_left_color",
            "right_wrist_0_rgb": "hand_right_color"
        }
        self.caps = {} 

    def get_images(self, global_frame_idx):
        segment_idx = global_frame_idx // 500
        local_frame_idx = global_frame_idx % 500
        images = {}
        for server_key, file_prefix in self.views_map.items():
            filename = f"{file_prefix}_{segment_idx}.mp4"
            filepath = os.path.join(self.video_dir, filename)
            cap_key = f"{server_key}_{segment_idx}"
            
            if cap_key not in self.caps:
                if os.path.exists(filepath):
                    cap = cv2.VideoCapture(filepath)
                    if cap.isOpened():
                        self.caps[cap_key] = cap
                    else: self.caps[cap_key] = None
                else: self.caps[cap_key] = None

            cap = self.caps[cap_key]
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
                ret, frame = cap.read()
                if ret: images[server_key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else: images[server_key] = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # 缺失视角填黑图，保证不报错
                images[server_key] = np.zeros((224, 224, 3), dtype=np.uint8)
        return images

    def close(self):
        for cap in self.caps.values():
            if cap: cap.release()
        self.caps.clear()

# ==========================================
# 3. 主逻辑
# ==========================================
async def run_client(args):
    print(f"📂 Loading Index File: {args.data_path}")
    
    # --- A. 加载 Index ---
    try:
        raw_data = np.load(args.data_path, allow_pickle=True)
        data = raw_data.item() if raw_data.ndim == 0 else raw_data
    except Exception as e:
        print(f"❌ Failed to load index file: {e}")
        return

    episode_rel_path = str(args.index)
    episode_idx_int = -1
    
    if 'video_path' in data:
        for i, path in enumerate(data['video_path']):
            path_str = path.decode('utf-8') if isinstance(path, bytes) else str(path)
            if str(args.index) in path_str:
                episode_rel_path = path_str 
                episode_idx_int = i
                print(f"✅ Found path fragment: {episode_rel_path}")
                break

    instruction_text = ""
    if 'instructions' in data and episode_idx_int != -1:
        instr_raw = data['instructions'][episode_idx_int]
        if isinstance(instr_raw, bytes): instruction_text = instr_raw.decode('utf-8')
        else: instruction_text = str(instr_raw)
        print(f"📝 Instruction: \"{instruction_text}\"")

    # --- B. 加载 AgiBot State 数据 ---
    meta_path = os.path.join(args.action_root, "meta_data.json")
    dim_list = []
    
    try:
        with open(meta_path, 'r') as f:
            all_meta_data = json.load(f)
            if episode_rel_path in all_meta_data:
                meta_info = all_meta_data[episode_rel_path]
                dim_list = meta_info["dim_list"]
            else:
                print(f"❌ Path {episode_rel_path} not found in meta_data.json!")
                return 
    except Exception as e:
        print(f"❌ Failed to load meta_data.json: {e}")
        return

    data_folder = os.path.join(args.action_root, episode_rel_path)
    action_npy_path = os.path.join(data_folder, "action.npy")

    episode_joint = None
    episode_gripper = None
    episode_head = None
    episode_waist = None

    if os.path.exists(action_npy_path):
        try:
            total_dim = sum(dim_list)
            file_size = os.path.getsize(action_npy_path)
            total_frames = file_size // (total_dim * 4)
            
            agibot_data = AgibotActionState.load_range_from_path(
                path=action_npy_path, dim_list=dim_list, start=0, end=total_frames
            )
            
            if agibot_data.state_joint_position is not None:
                episode_joint = agibot_data.state_joint_position     # (T, 14)
                episode_gripper = agibot_data.action_effector_position # (T, 2)
                episode_head = agibot_data.state_head_position # (T, 2)
                episode_waist = agibot_data.state_waist_position # (T, 2)
                print(f"✅ State Loaded! Frames: {total_frames}")
            else:
                print("❌ state_joint_position is None.")
        except Exception as e:
            print(f"❌ Error parsing action.npy: {e}")
            return
    else:
        print(f"❌ action.npy not found in {data_folder}")
        return

    if episode_joint is None:
        episode_joint = np.zeros((1000, 14)); episode_gripper = np.zeros((1000, 2))
        episode_head = np.zeros((1000, 2)); episode_waist = np.zeros((1000, 2))

    # --- C. 初始化 FK Solver & Camera Params ---
    fk_solver = None
    K, dist, Rcw, tcw = None, None, None, None

    if G1_FK and args.urdf and os.path.exists(args.urdf):
        print("🤖 Initializing FK Solver...")
        try:
            fk_solver = G1_FK(args.urdf, package_dirs=os.path.dirname(args.urdf))
        except Exception as e:
            print(f"❌ Failed to init FK: {e}")

    if args.camera_npy and os.path.exists(args.camera_npy):
        print("📷 Loading Camera Params...")
        cam_key = f"{episode_rel_path}_head"
        K, dist, Rcw, tcw = load_camera_params(args.camera_npy, cam_key)
        
        if K is not None:
            print(f"✅ Camera Params Loaded for {cam_key}")
        else:
            print(f"⚠️ Warning: Camera key {cam_key} not found.")

    # --- D. 准备循环 ---
    start_frame = args.start_frame
    end_frame = args.end_frame
    interval = args.interval
    target_frames = range(start_frame, end_frame, interval)
    
    loader = MultiViewVideoLoader(data_root=args.video_folder, episode_rel_path=episode_rel_path)
    uri = f"ws://{args.host}:{args.port}"
    print(f"🚀 Connecting to {uri}...")

    video_writer = None

    sanitized_instr = re.sub(r'[^a-zA-Z0-9]', '_', instruction_text)
    sanitized_instr = re.sub(r'_+', '_', sanitized_instr).strip('_')[:50]
    if not sanitized_instr: sanitized_instr = "no_instr"
    safe_id = episode_rel_path.replace(os.path.sep, '_').replace('/', '_')

    # ==========================================
    # 🆕 文件命名: {ID}_{Range}_{Type} (方便检索)
    # ==========================================
    file_prefix = f"{safe_id}_S{start_frame}_E{end_frame}"
    
    # 1. 视频文件
    filename_video = f"{file_prefix}_compare_SxS_growth_{sanitized_instr}.mp4"
    output_path = os.path.join(args.output_dir, filename_video)
    
    # 2. State Plot 文件
    filename_plot_state = f"{file_prefix}_error_plot_state.png"
    output_path_plot_state = os.path.join(args.output_dir, filename_plot_state)

    # 3. 3D Plot 文件
    filename_plot_fk = f"{file_prefix}_error_plot_3d.png"
    output_path_plot_fk = os.path.join(args.output_dir, filename_plot_fk)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    error_records_state = [] 
    error_records_fk = []    

    async with websockets.connect(uri, max_size=None) as websocket:
        for i, global_frame_idx in enumerate(tqdm(target_frames)):
            
            # 1. Image & State
            images_dict = loader.get_images(global_frame_idx)
            idx = global_frame_idx if global_frame_idx < len(episode_joint) else -1
            curr_joint = episode_joint[idx]   
            curr_gripper = episode_gripper[idx] 
            
            state_16 = np.concatenate([curr_joint, curr_gripper])
            state_32 = np.zeros(32, dtype=np.float32)
            state_32[:16] = state_16

            payload = {
                "image": images_dict,
                "image_mask": {k: True for k in images_dict},
                "state": state_32,
                "prompt": instruction_text
            }

            actions = None
            try:
                await websocket.send(msgpack.packb(payload, default=m.encode))
                resp_raw = await websocket.recv()
                
                if isinstance(resp_raw, str):
                    print(f"\n[Server Error]: {resp_raw}")
                    continue

                resp = msgpack.unpackb(resp_raw)
                actions = resp.get('actions', None)
                if actions is not None:
                    if actions.ndim == 3: actions = actions[0]
            except Exception as e:
                print(f"\n❌ Communication Error: {e}")
                break

            # ==========================================
            # 2. 计算误差 & 收集数据
            # ==========================================
            if actions is not None:
                check_horizon = min(interval, len(actions))
                temp_state_errors = []
                temp_fk_errors = [] 
                
                curr_head = episode_head[idx].tolist()
                curr_waist = episode_waist[idx].tolist()

                for t in range(check_horizon):
                    future_idx = idx + t
                    if future_idx >= len(episode_joint): break
                    
                    # A. State Error
                    gt_joint_t = episode_joint[future_idx]
                    gt_gripper_t = episode_gripper[future_idx]
                    gt_state_16 = np.concatenate([gt_joint_t, gt_gripper_t])
                    
                    pred_delta = actions[t, :16]
                    pred_state_16 = state_16 + pred_delta
                    
                    # Convert to Degrees for Joints
                    gt_deg = np.zeros(16)
                    gt_deg[:14] = np.rad2deg(gt_state_16[:14])
                    gt_deg[14:] = gt_state_16[14:]
                    
                    pred_deg = np.zeros(16)
                    pred_deg[:14] = np.rad2deg(pred_state_16[:14])
                    pred_deg[14:] = pred_state_16[14:]
                    
                    temp_state_errors.append(np.abs(gt_deg - pred_deg))

                    # B. FK Error
                    if fk_solver:
                        res_gt = fk_solver.fk(head_position=curr_head, waist_position=curr_waist, joint_position=gt_joint_t.tolist())
                        pred_joint_t_rad = pred_state_16[:14]
                        res_pred = fk_solver.fk(head_position=curr_head, waist_position=curr_waist, joint_position=pred_joint_t_rad.tolist())
                        
                        if res_gt and res_pred:
                            # 注意：这里我们尝试获取 'left_points'，如果你的 g1_fk 没改好，这里会报错
                            # 如果报错，请确保 g1_fk.py 已经更新
                            gt_pts_l = res_gt.get("left_points") 
                            pred_pts_l = res_pred.get("left_points")
                            
                            # 使用点的重心计算距离
                            gt_c_l = np.mean(gt_pts_l, axis=0) if gt_pts_l is not None else None
                            pred_c_l = np.mean(pred_pts_l, axis=0) if pred_pts_l is not None else None
                            
                            dist_l = np.linalg.norm(gt_c_l - pred_c_l) if (gt_c_l is not None and pred_c_l is not None) else 0.0
                            
                            # Right
                            gt_pts_r = res_gt.get("right_points")
                            pred_pts_r = res_pred.get("right_points")
                            gt_c_r = np.mean(gt_pts_r, axis=0) if gt_pts_r is not None else None
                            pred_c_r = np.mean(pred_pts_r, axis=0) if pred_pts_r is not None else None
                            dist_r = np.linalg.norm(gt_c_r - pred_c_r) if (gt_c_r is not None and pred_c_r is not None) else 0.0
                            
                            temp_fk_errors.append([dist_l, dist_r])
                        else:
                            temp_fk_errors.append([0.0, 0.0])
                
                if temp_state_errors:
                    error_records_state.append((global_frame_idx, np.mean(np.stack(temp_state_errors), axis=0)))
                if temp_fk_errors:
                    error_records_fk.append((global_frame_idx, np.mean(np.stack(temp_fk_errors), axis=0)))

            # ==========================================
            # 3. 视频生成 (Side-by-Side Growth)
            # ==========================================
            # ==========================================
            # 3. 视频生成 (使用 Points)
            # ==========================================
            if 'base_0_rgb' in images_dict and fk_solver and K is not None and actions is not None:
                
                base_img_orig = images_dict['base_0_rgb'].copy()
                base_img_orig = cv2.resize(base_img_orig, (640, 480))
                
                if video_writer is None:
                    h_vid, w_vid = base_img_orig.shape[:2]
                    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_vid * 2, h_vid))

                curr_head = episode_head[idx].tolist()
                curr_waist = episode_waist[idx].tolist()
                chunk_len = min(interval, len(actions))
                
                # 预计算 Points List
                pred_pts_l_list, pred_pts_r_list = [], []
                gt_pts_l_list, gt_pts_r_list = [], []
                
                for t in range(chunk_len):
                    # Pred FK
                    pred_joint_abs = curr_joint + actions[t, :14]
                    res = fk_solver.fk(head_position=curr_head, waist_position=curr_waist, joint_position=pred_joint_abs.tolist())
                    if res:
                        # 获取 Points
                        pred_pts_l_list.append(res.get("left_points"))
                        pred_pts_r_list.append(res.get("right_points"))
                    else:
                        pred_pts_l_list.append(None); pred_pts_r_list.append(None)
                    
                    # GT FK
                    f_idx = idx + t
                    if f_idx < len(episode_joint):
                        gt_j = episode_joint[f_idx]
                        res_gt = fk_solver.fk(head_position=curr_head, waist_position=curr_waist, joint_position=gt_j.tolist())
                        if res_gt:
                            gt_pts_l_list.append(res_gt.get("left_points"))
                            gt_pts_r_list.append(res_gt.get("right_points"))
                        else:
                            gt_pts_l_list.append(None); gt_pts_r_list.append(None)
                    else:
                        gt_pts_l_list.append(None); gt_pts_r_list.append(None)

                # 动画循环
                for t in range(chunk_len):
                    canvas_gt = base_img_orig.copy()
                    canvas_pred = base_img_orig.copy()
                    
                    # --- Draw GT (Left) ---
                    prev_c_gt_l, prev_c_gt_r = None, None
                    for k in range(t + 1):
                        color = (0, 255, 0) # Green
                        # 点的大小：当前帧大一点(5)，历史帧小一点(2)
                        radius = 5 if k == t else 2 
                        
                        # 改用 project_points_3d
                        canvas_gt, c = project_points_3d(canvas_gt, gt_pts_l_list[k], Rcw, tcw, K, dist, color, radius)
                        if k>0 and prev_c_gt_l and c: cv2.line(canvas_gt, prev_c_gt_l, c, color, 1) # 连线更细一点
                        prev_c_gt_l = c
                        
                        canvas_gt, c = project_points_3d(canvas_gt, gt_pts_r_list[k], Rcw, tcw, K, dist, color, radius)
                        if k>0 and prev_c_gt_r and c: cv2.line(canvas_gt, prev_c_gt_r, c, color, 1)
                        prev_c_gt_r = c
                    
                    # --- Draw Pred (Right) ---
                    prev_c_p_l, prev_c_p_r = None, None
                    for k in range(t + 1):
                        color = get_gradient_color(k, chunk_len)
                        radius = 5 if k == t else 2
                        
                        canvas_pred, c = project_points_3d(canvas_pred, pred_pts_l_list[k], Rcw, tcw, K, dist, color, radius)
                        if k>0 and prev_c_p_l and c: cv2.line(canvas_pred, prev_c_p_l, c, color, 1)
                        prev_c_p_l = c
                        
                        canvas_pred, c = project_points_3d(canvas_pred, pred_pts_r_list[k], Rcw, tcw, K, dist, color, radius)
                        if k>0 and prev_c_p_r and c: cv2.line(canvas_pred, prev_c_p_r, c, color, 1)
                        prev_c_p_r = c

                    cv2.putText(canvas_gt, "GT (Points)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(canvas_pred, "Pred (Points)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
                    combined = np.hstack((canvas_gt, canvas_pred))
                    # video_writer.write(combined)

                    # =========== 修改开始 ===========
                    # OpenCV VideoWriter 期望 BGR 格式，而你的 combined 是 RGB 格式
                    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                    video_writer.write(combined_bgr)
                    # =========== 修改结束 ===========

    # 4. 保存视频
    if video_writer:
        video_writer.release()
        print(f"\n✅ Video saved to: {output_path}")
    
    loader.close()

    # ==========================================
    # 5. 绘图 1: State Error (Unified Blue Style)
    # ==========================================
    if error_records_state:
        print(f"📊 Generating State Error Plot (Degrees)...")
        frames, errors_list = zip(*error_records_state)
        errors_matrix = np.stack(errors_list) 
        
        fig, axes = plt.subplots(4, 4, figsize=(24, 18))
        fig.suptitle(f"State Error Analysis\n(Blue Line = Prediction Error, Red Dashed = Threshold)", fontsize=20, y=0.98)
        
        for dim in range(16):
            ax = axes.flat[dim]
            y_vals = errors_matrix[:, dim]
            
            # 统一蓝色实线
            ax.plot(frames, y_vals, marker='.', linestyle='-', color='tab:blue', linewidth=1.5, label='L1 Error')
            
            if dim < 14: # Joints
                ax.set_ylabel("Error (Degrees)")
                ax.set_title(f"Joint {dim}", fontweight='bold')
                # 5度 阈值
                ax.axhline(y=5.0, color='tab:red', linestyle='--', alpha=0.6, linewidth=1.5, label='5° Threshold')
                current_max = np.max(y_vals)
                ax.set_ylim(0, max(current_max * 1.2, 6.0))
            else: # Gripper
                ax.set_ylabel("Error (Linear)")
                ax.set_title(f"Gripper {'Left' if dim==14 else 'Right'}", fontweight='bold')
                # 0.05 阈值
                ax.axhline(y=0.05, color='tab:red', linestyle='--', alpha=0.6, linewidth=1.5, label='0.05 Threshold')
                current_max = np.max(y_vals)
                ax.set_ylim(0, max(current_max * 1.2, 0.06))

            ax.grid(True, alpha=0.3)
            if dim == 0: ax.legend(loc='upper right', frameon=True)
            if dim >= 12: ax.set_xlabel("Frame Index")
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path_plot_state)
        plt.close()
        print(f"✅ Saved State Error Plot: {output_path_plot_state}")

    # ==========================================
    # 6. 绘图 2: 3D FK Error
    # ==========================================
    if error_records_fk:
        print(f"📊 Generating 3D Position Error Plot...")
        frames, fk_errors_list = zip(*error_records_fk)
        fk_errors_matrix = np.stack(fk_errors_list) 
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"End-Effector 3D Position Error (Mean over {interval} steps)", fontsize=16)
        
        ax1.plot(frames, fk_errors_matrix[:, 0], marker='o', markersize=4, color='tab:green')
        ax1.set_ylabel("Error (Meters)"); ax1.set_title("Left Arm End-Effector Error", fontweight='bold')
        ax1.grid(True, alpha=0.3); ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5cm Threshold')
        ax1.legend()
        
        ax2.plot(frames, fk_errors_matrix[:, 1], marker='o', markersize=4, color='tab:purple')
        ax2.set_ylabel("Error (Meters)"); ax2.set_xlabel("Frame Index"); ax2.set_title("Right Arm End-Effector Error", fontweight='bold')
        ax2.grid(True, alpha=0.3); ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5cm Threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path_plot_fk)
        plt.close()
        print(f"✅ Saved FK Error Plot: {output_path_plot_fk}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--action_root", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    
    parser.add_argument("--camera_npy", type=str, default="", help="Path to camera_param.npy")
    parser.add_argument("--urdf", type=str, default="", help="Path to robot URDF")
    parser.add_argument("--action_horizon", type=int, default=30)

    args = parser.parse_args()
    asyncio.run(run_client(args))

if __name__ == "__main__":
    main()