#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import os
import random
import numpy as np
import torch
import websockets
import matplotlib.pyplot as plt

# 📦 引入 MsgPack
import msgpack
import msgpack_numpy
msgpack_numpy.patch() 

from openpi.training.agibot_dataset import AgiBotDataset

# ============================
# 1. 辅助工具
# ============================
def load_stats(stats_path):
    print(f"📊 Loading stats manually from: {stats_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        full_stats = json.load(f)
    
    if "norm_stats" in full_stats:
        stats = full_stats["norm_stats"]
    else:
        stats = full_stats

    if "actions" not in stats:
        raise ValueError("JSON file does not contain 'actions' key.")

    action_q01 = np.array(stats["actions"]["q01"], dtype=np.float32)
    action_q99 = np.array(stats["actions"]["q99"], dtype=np.float32)
    
    return action_q01, action_q99

def unnormalize_actions(pred_actions, q01, q99):
    if isinstance(pred_actions, torch.Tensor):
        pred_actions = pred_actions.detach().cpu().numpy()
        
    VALID_DIM = 14 
    q01_joint = q01[:VALID_DIM]
    q99_joint = q99[:VALID_DIM]
    
    denom = q99_joint - q01_joint
    denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)

    pred_physical = pred_actions.copy()

    # 反归一化前 14 维
    pred_joint_norm = pred_actions[:, :VALID_DIM]
    pred_joint_phys = (pred_joint_norm + 1.0) * denom / 2.0 + q01_joint
    
    pred_physical[:, :VALID_DIM] = pred_joint_phys
    
    return pred_physical

# ============================
# 2. 绘图函数
# ============================
def plot_comparison(gt_actions, pred_actions, prompt, step_idx, mode_name, save_path):
    min_len = min(len(gt_actions), len(pred_actions))
    gt = gt_actions[:min_len]
    pred = pred_actions[:min_len]
    dim = gt.shape[1]
    
    cols = 4
    rows = (dim // cols) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    axes = axes.flatten()
    
    fig.suptitle(f"[{mode_name}] Prompt: {prompt}\nSample Index: {step_idx}", fontsize=16)

    total_mse = 0
    for i in range(dim):
        ax = axes[i]
        
        # GT (Physical)
        ax.plot(gt[:, i], label="GT (Physical)", color='green', linewidth=2, linestyle='--')
        
        # Pred
        ax.plot(pred[:, i], label=f"Pred ({mode_name})", color='red', linewidth=2, alpha=0.8)
        
        mse = np.mean((gt[:, i] - pred[:, i]) ** 2)
        total_mse += mse
        
        title_suffix = ""
        if i < 14: title_suffix = " (Joint)"
        elif i < 16: title_suffix = " (Gripper)"
        else: title_suffix = " (Pad)"

        ax.set_title(f"Dim {i}{title_suffix}\nMSE: {mse:.4f}", fontsize=10)
        
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i in range(dim, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"📊 [{mode_name}] Mean MSE: {total_mse / dim:.6f}")
    print(f"✅ Saved plot to: {os.path.abspath(save_path)}")
    plt.close()

# ============================
# 3. Payload 构建
# ============================
def build_payload(sample):
    def _process_img(img):
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.array(img)
            
        # 0-1 转 0-255 uint8 (关键修复)
        if arr.dtype.kind == 'f' and arr.max() <= 1.05:
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
            
        return arr

    states = sample["states"]
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    
    payload = {
        "image": {
            "base_0_rgb": _process_img(sample["head"]),
            "left_wrist_0_rgb": _process_img(sample["left_gripper"]),
            "right_wrist_0_rgb": _process_img(sample["right_gripper"]),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": True,
        },
        "state": states,
        "prompt": sample.get("prompt", "")
    }
    return payload

# ============================
# 4. 主流程
# ============================
async def run_random_test(args):
    # 环境变量设置
    if args.norm_stats_file: os.environ["NORM_STATS_FILE"] = args.norm_stats_file
    if args.index_file: os.environ["AGIBOT_INDEX_FILE"] = args.index_file

    # [1] 根据参数决定是否加载 Stats
    q01, q99 = None, None
    if args.unnormalize:
        q01, q99 = load_stats(args.norm_stats_file)
    else:
        print("ℹ️  Manual un-normalization is DISABLED (Default). Assuming Server returns physical units.")

    print(f"📂 Loading Dataset...")
    ds = AgiBotDataset(root_dir=args.dataset_root, action_horizon=args.action_horizon, normalization=False)

    random_idx = random.randint(0, len(ds) - 1)
    print(f"🎲 Index: {random_idx}")
    
    sample = ds[random_idx]
    prompt = sample['prompt']
    
    gt_actions = sample["actions"]
    if isinstance(gt_actions, torch.Tensor): gt_actions = gt_actions.numpy()

    print(f"🚀 Connecting to {args.ws} ...")
    async with websockets.connect(args.ws, max_size=2**28) as ws:
        try:
            await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

        print("📤 Sending payload...")
        payload = build_payload(sample)
        await ws.send(msgpack.packb(payload))
        
        print("⏳ Waiting for response...")
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
        
        if isinstance(resp_raw, str):
            print(f"❌ [Server Error]:\n{resp_raw}")
            return
        
        resp = msgpack.unpackb(resp_raw)
        
        if isinstance(resp, dict) and "actions" in resp:
            pred_actions = np.array(resp["actions"])
        elif isinstance(resp, (np.ndarray, list)): 
            pred_actions = np.array(resp)
        
        if pred_actions.ndim == 3: pred_actions = pred_actions[0]

    # [2] 处理结果
    final_pred = pred_actions
    mode_name = "Server_Output"

    # 如果开启了参数，则执行反归一化
    if args.unnormalize:
        print("🔄 Applying manual un-normalization...")
        final_pred = unnormalize_actions(pred_actions, q01, q99)
        mode_name = "Manually_Unnormalized"

    # [3] 打印数值检查
    print("\n" + "="*50)
    print(f"📏 Range Check (First 14 dims):")
    print(f"   🟩 GT Max (Physical):  {np.max(np.abs(gt_actions[:, :14])):.4f}")
    print(f"   🟥 Final Pred Max:     {np.max(np.abs(final_pred[:, :14])):.4f}")
    print("="*50 + "\n")

    # [4] 绘图
    plot_comparison(
        gt_actions, 
        final_pred, 
        prompt, 
        random_idx, 
        mode_name=mode_name,
        save_path=f"check_{random_idx}.png"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=str, default="ws://127.0.0.1:8001")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--norm_stats_file", type=str, required=True)
    ap.add_argument("--index_file", type=str, default="episodic_dataset_fixed_static.npy")
    ap.add_argument("--action_horizon", type=int, default=30)
    
    # ✅ 新增参数：默认不开启反归一化
    ap.add_argument("--unnormalize", action="store_true", help="Enable manual un-normalization of server output.")
    
    args = ap.parse_args()
    asyncio.run(run_random_test(args))

if __name__ == "__main__":
    main()