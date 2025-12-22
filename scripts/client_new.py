#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import os
import random
import numpy as np
import torch
import websockets
import cv2
import msgpack
import msgpack_numpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

msgpack_numpy.patch() 
from openpi.training.agibot_dataset import AgiBotDataset

# 尝试导入 FK 模块
try:
    from g1_fk import G1_FK
    print("✅ Successfully imported G1_FK Module")
except ImportError:
    G1_FK = None
    print("⚠️ Warning: G1_FK module not found.")

# ============================
# 1. 辅助绘图与工具函数
# ============================
def plot_action_comparison(gt_actions, pred_actions, prompt, idx, save_path):
    dim = gt_actions.shape[1]
    horizon = min(gt_actions.shape[0], pred_actions.shape[0])
    cols = 4
    rows = (dim + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()
    fig.suptitle(f"Action Comparison | Index: {idx}\nPrompt: {prompt}", fontsize=14)

    for i in range(dim):
        ax = axes[i]
        ax.plot(gt_actions[:horizon, i], label="GT", color='green', linestyle='--', linewidth=2)
        ax.plot(pred_actions[:horizon, i], label="Pred", color='red', alpha=0.8)
        mse = np.mean((gt_actions[:horizon, i] - pred_actions[:horizon, i])**2)
        ax.set_title(f"Dim {i} (MSE: {mse:.4f})", fontsize=10)
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(dim, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def load_camera_params(npy_path, lookup_key):
    if not os.path.exists(npy_path): return None, None, None, None
    data = np.load(npy_path, allow_pickle=True).item()
    if lookup_key not in data.get('camera2intrinsic', {}): return None, None, None, None

    intr_raw = data['camera2intrinsic'][lookup_key]
    fx, fy, cx, cy = intr_raw
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((1, 5)) 

    ext_raw = data['camera2extrinsic'][lookup_key]
    vals = np.array(ext_raw).flatten()
    r_obj = R_scipy.from_euler('xyz', vals[3:], degrees=False) 
    R_wc = r_obj.as_matrix()
    T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = vals[:3]
    T_cw = np.linalg.inv(T_wc)
    
    return K, dist, T_cw[:3, :3], T_cw[:3, 3]

def project_bbox(frame, bbox_3d, Rcw, tcw, K, dist, color, label=""):
    if bbox_3d is None: return frame
    corners_w = np.array([
        [bbox_3d[0], bbox_3d[1], bbox_3d[2]], [bbox_3d[0], bbox_3d[1], bbox_3d[5]], 
        [bbox_3d[0], bbox_3d[4], bbox_3d[2]], [bbox_3d[0], bbox_3d[4], bbox_3d[5]],
        [bbox_3d[3], bbox_3d[1], bbox_3d[2]], [bbox_3d[3], bbox_3d[1], bbox_3d[5]], 
        [bbox_3d[3], bbox_3d[4], bbox_3d[2]], [bbox_3d[3], bbox_3d[4], bbox_3d[5]]
    ])
    corners_c = (Rcw @ corners_w.T).T + tcw
    if (corners_c[:, 2] <= 0.01).all(): return frame
    pix, _ = cv2.projectPoints(corners_c, np.zeros(3), np.zeros(3), K, dist)
    pix = pix.reshape(-1, 2)
    x_min, y_min = pix.min(axis=0).astype(int); x_max, y_max = pix.max(axis=0).astype(int)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (max(0, x_min), max(0, y_min)), (min(w, x_max), min(h, y_max)), color, 2)
    cv2.putText(frame, label, (max(0, x_min), max(20, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

# ============================
# 2. Payload 构建
# ============================
def build_payload(sample):
    def _process_img(img):
        arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)
        if arr.dtype.kind == 'f' and arr.max() <= 1.05: arr = (arr * 255).astype(np.uint8)
        return arr
    return {
        "image": {
            "base_0_rgb": _process_img(sample["head"]),
            "left_wrist_0_rgb": _process_img(sample["left_gripper"]),
            "right_wrist_0_rgb": _process_img(sample["right_gripper"]),
        },
        "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": True},
        "state": sample["states"].numpy() if isinstance(sample["states"], torch.Tensor) else sample["states"],
        "prompt": sample.get("prompt", "")
    }

# ============================
# 3. 主流程
# ============================
async def run_visual_test(args):
    # --- 创建可视化文件夹 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "visualization_fk_compare")
    os.makedirs(output_folder, exist_ok=True)

    os.environ["NORM_STATS_FILE"] = args.norm_stats_file
    os.environ["AGIBOT_INDEX_FILE"] = args.index_file

    ds = AgiBotDataset(root_dir=args.dataset_root, action_horizon=args.action_horizon, normalization=False)
    idx = random.randint(0, len(ds) - 1)
    sample = ds[idx]
    rel_path = sample["rel_path"]
    scene_id, ep_id = rel_path.split('/')
    camera_key = f"{scene_id}/{ep_id}_head"

    # 文件前缀统一
    file_prefix = f"idx{idx}_{scene_id}_{ep_id}"

    # --- 相机参数 Log ---
    print("\n📷 " + "="*25 + " CAMERA PARAMETERS LOG " + "="*25)
    K, dist, Rcw, tcw = load_camera_params(args.camera_npy, camera_key)
    print(f"Source File: {args.camera_npy}")
    print(f"Lookup Key:  {camera_key}")
    print(f"Intrinsic K: \n{K}")
    print(f"Extrinsic (R_cw): \n{Rcw}")
    print(f"Extrinsic (t_cw): \n{tcw}")
    print("="*77 + "\n")

    async with websockets.connect(args.ws, max_size=2**28) as ws:
        try: await asyncio.wait_for(ws.recv(), timeout=1.0)
        except: pass

        payload = build_payload(sample)
        print("📤 " + "="*25 + " CLIENT PAYLOAD (INPUT) " + "="*25)
        print(f"Prompt: {payload['prompt']}")
        for k, v in payload['image'].items():
            print(f"  - Key: {k:<20} | Shape: {v.shape} | Range: [{v.min()}, {v.max()}]")
        print("="*77 + "\n")

        await ws.send(msgpack.packb(payload))
        print("⏳ Waiting for server response...")
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
        resp = msgpack.unpackb(resp_raw)

        # --- Server 响应 Log ---
        print("\n📥 " + "="*25 + " SERVER RESPONSE (OUTPUT) " + "="*25)
        print(f"Type: {type(resp)}")
        if isinstance(resp, dict):
            print(f"Keys: {list(resp.keys())}")
            for k, v in resp.items():
                if hasattr(v, 'shape'): print(f"  - '{k}': Shape {v.shape}, Dtype {v.dtype}")
                else: print(f"  - '{k}': {v}")
        print("="*77 + "\n")
        
        pred_actions = np.array(resp["actions"]) if "actions" in resp else np.array(resp)
        if pred_actions.ndim == 3: pred_actions = pred_actions[0]

    # --- 数值对比 Log ---
    gt_actions = sample["actions"].numpy() if isinstance(sample["actions"], torch.Tensor) else sample["actions"]
    print("\n🔢 " + "="*25 + " NUMERIC COMPARISON (ALL STEPS) " + "="*25)
    for t in range(len(pred_actions)):
        gt_str = ", ".join([f"{x: .4f}" for x in gt_actions[t, :16]])
        pd_str = ", ".join([f"{x: .4f}" for x in pred_actions[t, :16]])
        print(f"[Step {t:02d}] GT  : [{gt_str}]")
        print(f"          Pred: [{pd_str}]")
    print("="*77 + "\n")

    # ==========================================
    # 4. FK 全轨迹投影 (30帧) 与双图拼接
    # ==========================================
    fk_solver = G1_FK(args.urdf, package_dirs=os.path.dirname(args.urdf)) if G1_FK else None
    
    # 统一 Resize 到 640x480
    head_img_rgb = (sample["head"].numpy() * 255).astype(np.uint8)
    head_img_640 = cv2.resize(head_img_rgb, (640, 480))
    
    img_gt_viz = cv2.cvtColor(head_img_640.copy(), cv2.COLOR_RGB2BGR)
    img_pred_viz = cv2.cvtColor(head_img_640.copy(), cv2.COLOR_RGB2BGR)

    if fk_solver and K is not None:
        current_joints_abs = sample["states"][:14].numpy()
        head_pos = sample["state_head"].tolist()
        waist_pos = sample["state_waist"].tolist()
        
        # 每 5 帧画一个框，形成轨迹感
        steps_to_draw = range(0, len(pred_actions), 5)

        # --- A. 投影 GT 30帧轨迹 (绿色) ---
        for t in range(len(gt_actions)):
            abs_joints = current_joints_abs + gt_actions[t, :14]
            res = fk_solver.fk(head_position=head_pos, waist_position=waist_pos, joint_position=abs_joints.tolist())
            if res and (t in steps_to_draw or t == len(gt_actions)-1):
                alpha = 1.0 - (t / len(gt_actions)) * 0.6 # 随时间变淡
                color = (0, int(255 * alpha), 0)
                img_gt_viz = project_bbox(img_gt_viz, res.get("left_bbox"), Rcw, tcw, K, dist, color, f"GT_{t}")
                img_gt_viz = project_bbox(img_gt_viz, res.get("right_bbox"), Rcw, tcw, K, dist, color, f"GT_{t}")

        # --- B. 投影 Pred 30帧轨迹 (红色) ---
        for t in range(len(pred_actions)):
            abs_joints = current_joints_abs + pred_actions[t, :14]
            res = fk_solver.fk(head_position=head_pos, waist_position=waist_pos, joint_position=abs_joints.tolist())
            if res and (t in steps_to_draw or t == len(pred_actions)-1):
                alpha = 1.0 - (t / len(pred_actions)) * 0.6
                color = (0, 0, int(255 * alpha))
                img_pred_viz = project_bbox(img_pred_viz, res.get("left_bbox"), Rcw, tcw, K, dist, color, f"P_{t}")
                img_pred_viz = project_bbox(img_pred_viz, res.get("right_bbox"), Rcw, tcw, K, dist, color, f"P_{t}")

    # 垂直拼接 2x1
    cv2.putText(img_gt_viz, "TOP: GROUND TRUTH 30-FRAME TRAJECTORY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_pred_viz, "BOTTOM: PREDICTION 30-FRAME TRAJECTORY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    combined_img = np.vstack((img_gt_viz, img_pred_viz))

    # --- 保存文件 (统一目录与前缀) ---
    fk_save_path = os.path.join(output_folder, f"{file_prefix}_fk.png")
    compare_save_path = os.path.join(output_folder, f"{file_prefix}_compare.png")
    
    cv2.imwrite(fk_save_path, combined_img)
    plot_action_comparison(gt_actions, pred_actions, sample['prompt'], idx, compare_save_path)
    
    print(f"✅ Visualization saved to: {output_folder}")
    print(f"   - FK Trajectory: {os.path.basename(fk_save_path)}")
    print(f"   - Numeric Plot:  {os.path.basename(compare_save_path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=str, default="ws://127.0.0.1:8001")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--norm_stats_file", type=str, required=True)
    ap.add_argument("--index_file", type=str, required=True)
    ap.add_argument("--camera_npy", type=str, required=True)
    ap.add_argument("--urdf", type=str, required=True)
    ap.add_argument("--action_horizon", type=int, default=30)
    args = ap.parse_args()
    asyncio.run(run_visual_test(args))

if __name__ == "__main__":
    main()