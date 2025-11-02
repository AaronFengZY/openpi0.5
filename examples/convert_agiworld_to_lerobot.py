#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert AgiBotWorld (3-view videos + H5 state + JSON segments) -> LeRobot-style dataset.

- Each action_text segment becomes ONE episode (trajectory).
- Save per-episode frames for 3 cameras: head, left_gripper, right_gripper
- Save states (joint positions) and actions (delta joint positions) as .npy
- Save instruction.txt with action_text
- Save timestamps (if available in H5) for the same frame slice

Input (hard-coded for quick test; edit if needed):
  Videos:
    /home/v-zhifeng/HPE/openpi/agibotworld/videos/head_color.mp4
    /home/v-zhifeng/HPE/openpi/agibotworld/videos/hand_left_color.mp4
    /home/v-zhifeng/HPE/openpi/agibotworld/videos/hand_right_color.mp4
  Language JSON (segments):
    /home/v-zhifeng/HPE/openpi/agibotworld/task_info/task_327.json
  State H5:
    /home/v-zhifeng/HPE/openpi/agibotworld/state/episode_648642/proprio_stats.h5

Output (default):
  /home/v-zhifeng/HPE/openpi/agibotworld_lerobot_test/
    metadata.json
    episodes/
      000000/  # 1st action_text unit
        images/head/000000.jpg ...
        images/left_gripper/000000.jpg ...
        images/right_gripper/000000.jpg ...
        states.npy
        actions.npy
        timestamps.npy (if available)
        instruction.txt
      000001/
        ...

Requirements:
  pip/uv install: opencv-python, h5py, numpy, pillow, tqdm, tyro (optional)
"""

import os
import json
import math
import h5py
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import argparse

# ---------- Hard-coded inputs for your quick test ----------
VIDEO_DIR = Path("/home/v-zhifeng/HPE/openpi/agibotworld/videos")
HEAD_VIDEO = VIDEO_DIR / "head_color.mp4"
LEFT_VIDEO = VIDEO_DIR / "hand_left_color.mp4"
RIGHT_VIDEO = VIDEO_DIR / "hand_right_color.mp4"

LANG_JSON = Path("/home/v-zhifeng/HPE/openpi/agibotworld/task_info/task_327.json")
H5_PATH   = Path("/home/v-zhifeng/HPE/openpi/agibotworld/state/episode_648642/proprio_stats.h5")

# Output root
OUT_ROOT = Path("/home/v-zhifeng/HPE/openpi/agibotworld_lerobot_test")

# Image save settings
SAVE_SIZE = (256, 256)  # (width, height)
IMG_QUALITY = 90

# Cameras + mapping
CAM_MAP = {
    "head": HEAD_VIDEO,
    "left_gripper": LEFT_VIDEO,
    "right_gripper": RIGHT_VIDEO,
}


# ----------------------- Helpers -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Random-access read: set frame index then read."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame  # BGR


def num_frames(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_jpg(path: Path, rgb: np.ndarray, size: Tuple[int,int]=(256,256), quality: int=90):
    """rgb: HxWx3 (uint8)"""
    h, w, _ = rgb.shape
    if (w, h) != size:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    # OpenCV expects BGR to encode; convert back
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def load_segments_from_json(json_path: Path, target_episode_id: Optional[int]=None) -> List[Dict]:
    """
    JSON structure example (list of dicts). We'll pick the first item if target_episode_id is None,
    otherwise match by 'episode_id'.
    Returns a list of segments: [{"start_frame": s, "end_frame": e, "action_text": "...", "skill": "..."}]
    """
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError("task json is not a list")

    if target_episode_id is None:
        entry = data[0]
    else:
        matches = [x for x in data if int(x.get("episode_id", -1)) == int(target_episode_id)]
        if not matches:
            raise ValueError(f"episode_id={target_episode_id} not found in {json_path}")
        entry = matches[0]

    cfg = entry["label_info"]["action_config"]
    segments = []
    for seg in cfg:
        segments.append({
            "start": int(seg["start_frame"]),
            "end": int(seg["end_frame"]),
            "text": seg.get("action_text", ""),
            "skill": seg.get("skill", ""),
        })
    return segments, entry.get("task_name", ""), entry.get("init_scene_text", "")


def try_read_h5(h5: h5py.File, keys: List[str]) -> Optional[np.ndarray]:
    """Try multiple candidate keys in order, return first that exists (as numpy)."""
    for k in keys:
        if k in h5:
            return np.array(h5[k])
    # keys may be nested paths
    for k in keys:
        try:
            d = h5[k]
            return np.array(d)
        except Exception:
            pass
    return None


def load_state_and_timestamps(h5_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Try to read joint positions (prefer state/joint/position, fallback to joint/position),
    and timestamps (state/timestamp).
    Returns:
      positions: (T, D_joint)
      timestamps: (T,) or None
    """
    with h5py.File(str(h5_path), "r") as f:
        # joint positions
        pos = try_read_h5(f, [
            "state/joint/position",
            "joint/position",
        ])
        if pos is None:
            raise RuntimeError("Cannot find joint positions in H5 (tried state/joint/position and joint/position).")

        # timestamps (optional)
        ts = try_read_h5(f, [
            "state/timestamp",
            "timestamp",
        ])
    return pos, ts


def compute_actions_from_states(states: np.ndarray) -> np.ndarray:
    """
    Simple finite-difference as actions: Δstate (t) = state(t) - state(t-1), with zeros at t=0.
    states: (T, D)
    returns: (T, D)
    """
    if len(states) == 0:
        return states
    deltas = np.zeros_like(states, dtype=np.float32)
    deltas[1:] = (states[1:] - states[:-1]).astype(np.float32)
    return deltas


def write_metadata(out_root: Path, cameras: List[str], action_dim: int, state_dim: int, fps: Optional[float]):
    meta = {
        "format": "lerobot-minimal-0.1",
        "cameras": cameras,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "fps": fps,
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2))


# ----------------------- Main conversion -----------------------
def convert(
    out_root: Path = OUT_ROOT,
    episode_id: Optional[int] = 648642,
    resize_size: Tuple[int,int] = SAVE_SIZE,
    jpeg_quality: int = IMG_QUALITY,
):
    # Prepare output
    episodes_dir = out_root / "episodes"
    ensure_dir(episodes_dir)

    # Open videos
    caps: Dict[str, cv2.VideoCapture] = {}
    total_frames: Dict[str, int] = {}
    fps: Dict[str, float] = {}

    for cam, path in CAM_MAP.items():
        cap = open_video(path)
        caps[cam] = cap
        total_frames[cam] = num_frames(cap)
        fps[cam] = cap.get(cv2.CAP_PROP_FPS)

    # Segments & texts
    segments, task_name, scene_text = load_segments_from_json(LANG_JSON, target_episode_id=episode_id)

    # State & timestamps
    joint_pos, timestamps = load_state_and_timestamps(H5_PATH)   # (1295, 14), (1295,)
    T_all, state_dim = joint_pos.shape

    # Write metadata once (fps 取 head 相机的，或三者平均）
    fps_val = fps.get("head") or np.mean([v for v in fps.values() if v > 0]) if len(fps) else None
    write_metadata(out_root, list(CAM_MAP.keys()), action_dim=state_dim, state_dim=state_dim, fps=fps_val)

    # Iterate segments; each becomes ONE episode
    ep_counter = 0
    for seg in segments:
        s = int(seg["start"])
        e = int(seg["end"])
        if e < s:
            print(f"[WARN] skip invalid segment: start={s}, end={e}")
            continue

        # Clamp to available frames across all videos & state
        max_valid = min(min(total_frames.values()), T_all) - 1
        s_clamped = max(0, min(s, max_valid))
        e_clamped = max(0, min(e, max_valid))
        if e_clamped < s_clamped:
            print(f"[WARN] segment after clamp empty: {seg}")
            continue

        # Episode folder
        ep_dir = episodes_dir / f"{ep_counter:06d}"
        img_root = ep_dir / "images"
        ensure_dir(img_root)
        for cam in CAM_MAP.keys():
            ensure_dir(img_root / cam)

        # Slice states & actions
        states_slice = joint_pos[s_clamped:e_clamped+1].astype(np.float32)    # (T_seg, D)
        actions_slice = compute_actions_from_states(states_slice)             # (T_seg, D)
        # Slice timestamps (optional)
        ts_slice = None
        if timestamps is not None and len(timestamps) > e_clamped:
            ts_slice = timestamps[s_clamped:e_clamped+1].astype(np.int64)

        # Save frames for each camera
        T_seg = states_slice.shape[0]
        for t in tqdm(range(T_seg), desc=f"Episode {ep_counter:06d} frames", leave=False):
            global_idx = s_clamped + t
            for cam, cap in caps.items():
                frame = read_frame(cap, global_idx)
                if frame is None:
                    # If failed to read a specific frame, try to skip or break
                    # Here we skip saving this frame for this camera
                    continue
                rgb = bgr_to_rgb(frame)
                save_jpg(img_root / cam / f"{t:06d}.jpg", rgb, size=resize_size, quality=jpeg_quality)

        # Save npy + instruction
        np.save(ep_dir / "states.npy", states_slice)
        np.save(ep_dir / "actions.npy", actions_slice)
        if ts_slice is not None:
            np.save(ep_dir / "timestamps.npy", ts_slice)

        instr_lines = []
        if seg.get("text"):
            instr_lines.append(seg["text"])
        if seg.get("skill"):
            instr_lines.append(f"[skill] {seg['skill']}")
        if task_name:
            instr_lines.append(f"[task] {task_name}")
        if scene_text:
            instr_lines.append(f"[scene] {scene_text}")

        (ep_dir / "instruction.txt").write_text("\n".join(instr_lines) if instr_lines else "")

        ep_counter += 1

    # Release videos
    for cap in caps.values():
        cap.release()

    print(f"✅ Done. Saved {ep_counter} episodes to: {episodes_dir}")
    print(f"   Cameras: {list(CAM_MAP.keys())}")
    print(f"   State dim: {state_dim} | Actions dim (Δstate): {state_dim}")
    print(f"   Example episode: {episodes_dir / '000000'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default=str(OUT_ROOT))
    parser.add_argument("--episode_id", type=int, default=648642, help="episode_id to pick from JSON")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    # 这里不需要也不应使用 global，直接赋值即可
    OUT_ROOT = Path(args.out_root)
    SAVE_SIZE = (args.width, args.height)
    IMG_QUALITY = args.quality

    convert(out_root=OUT_ROOT, episode_id=args.episode_id,
            resize_size=SAVE_SIZE, jpeg_quality=IMG_QUALITY)
