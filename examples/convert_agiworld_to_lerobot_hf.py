#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgiBotWorld -> LeRobot (official HF format)
- Each action_text segment becomes ONE episode
- Cameras: head, left_gripper, right_gripper
- states: joint positions (T, 14) float32
- actions: delta joint (T, 14) float32
- prompt: action_text + [skill]/[task]/[scene] (string feature, added per-frame)
- task: task_name (episode-level metadata, passed to save_episode())

输出目录：$HF_LEROBOT_HOME/<REPO_ID>
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# === 输入（按你给的绝对路径）===
VIDEO_DIR = Path("/home/v-zhifeng/HPE/openpi/agibotworld/videos")
HEAD_VIDEO = VIDEO_DIR / "head_color.mp4"
LEFT_VIDEO = VIDEO_DIR / "hand_left_color.mp4"
RIGHT_VIDEO = VIDEO_DIR / "hand_right_color.mp4"

LANG_JSON = Path("/home/v-zhifeng/HPE/openpi/agibotworld/task_info/task_327.json")
H5_PATH   = Path("/home/v-zhifeng/HPE/openpi/agibotworld/state/episode_648642/proprio_stats.h5")

# LeRobot 数据集名字（相对于 $HF_LEROBOT_HOME）
REPO_ID = "agibotworld_lerobot_test"

# 输出图像分辨率
SAVE_W, SAVE_H = 256, 256

# === LeRobot API ===
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

def open_cap(p: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {p}")
    return cap

def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {idx}")
    # BGR->RGB, resize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape[1] != SAVE_W or frame.shape[0] != SAVE_H:
        frame = cv2.resize(frame, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)  # HxWx3

def load_segments(json_path: Path, target_episode_id: Optional[int] = None):
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError("task json must be a list")
    if target_episode_id is None:
        entry = data[0]
    else:
        cand = [x for x in data if int(x.get("episode_id", -1)) == int(target_episode_id)]
        if not cand:
            raise ValueError(f"episode_id={target_episode_id} not found")
        entry = cand[0]
    segs = []
    for s in entry["label_info"]["action_config"]:
        segs.append(
            {
                "start": int(s["start_frame"]),
                "end": int(s["end_frame"]),
                "text": s.get("action_text", ""),
                "skill": s.get("skill", ""),
            }
        )
    return segs, entry.get("task_name", "") or "", entry.get("init_scene_text", "") or ""

def try_read_h5(f: h5py.File, keys):
    for k in keys:
        if k in f:
            return np.array(f[k])
    for k in keys:
        try:
            return np.array(f[k])
        except Exception:
            pass
    return None

def load_states_and_timestamps(h5_path: Path):
    with h5py.File(str(h5_path), "r") as f:
        joints = try_read_h5(f, ["state/joint/position", "joint/position"])
        if joints is None:
            raise RuntimeError("Cannot find joint positions in H5")
        ts = try_read_h5(f, ["state/timestamp", "timestamp"])
    joints = joints.astype(np.float32)
    return joints, ts

def delta_from_states(states: np.ndarray):
    acts = np.zeros_like(states, dtype=np.float32)
    if len(states) > 1:
        acts[1:] = states[1:] - states[:-1]
    return acts

def main():
    print(f"HF_LEROBOT_HOME = {HF_LEROBOT_HOME}")

    # 1) 创建 LeRobot 数据集（会生成 meta/info.json 等）
    # 注意：字符串特征 dtype 用 "string"，并提供 shape 键（LeRobot → HF 会映射成标量字符串）
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,              # 建议换新名字以避免旧 meta 干扰
        robot_type="agibot",
        fps=10,
        features={
            "head":          {"dtype": "image",   "shape": (SAVE_H, SAVE_W, 3), "names": ["h","w","c"]},
            "left_gripper":  {"dtype": "image",   "shape": (SAVE_H, SAVE_W, 3), "names": ["h","w","c"]},
            "right_gripper": {"dtype": "image",   "shape": (SAVE_H, SAVE_W, 3), "names": ["h","w","c"]},
            "states":        {"dtype": "float32", "shape": (14,),               "names": ["state"]},
            "actions":       {"dtype": "float32", "shape": (14,),               "names": ["action"]},
            "prompt":        {"dtype": "string",  "shape": (1,),                "names": ["text"]},  # ← 用(1,)
            # ⚠️ 不要在 features 里声明 "task"
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # 2) 打开视频、读取 H5、读取分段
    caps = {
        "head": open_cap(HEAD_VIDEO),
        "left_gripper": open_cap(LEFT_VIDEO),
        "right_gripper": open_cap(RIGHT_VIDEO),
    }
    nframes = int(min(*(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps.values())))
    joints, timestamps = load_states_and_timestamps(H5_PATH)
    T_all = min(nframes, joints.shape[0])

    segs, task_name, scene_text = load_segments(LANG_JSON, target_episode_id=648642)

    # 3) 逐段写入（每个 action_text = 一个 episode）
    ep_count = 0
    for seg in segs:
        s, e = int(seg["start"]), int(seg["end"])
        if e < s:
            continue
        s = max(0, min(s, T_all - 1))
        e = max(0, min(e, T_all - 1))
        if e < s:
            continue

        states_slice  = joints[s : e + 1]                # (T, 14)
        actions_slice = delta_from_states(states_slice)  # (T, 14)
        T_seg = states_slice.shape[0]

        # 固定本段的文本
        base_prompt = seg.get("text", "") or ""
        if seg.get("skill"):
            base_prompt += f" [skill] {seg['skill']}"
        if task_name:
            base_prompt += f" [task] {task_name}"
        if scene_text:
            base_prompt += f" [scene] {scene_text}"

        # 每帧写入
        for t in tqdm(range(T_seg), desc=f"episode {ep_count:06d}", leave=False):
            gidx = s + t
            frame_dict = {
                "head":          read_frame(caps["head"], gidx),
                "left_gripper":  read_frame(caps["left_gripper"], gidx),
                "right_gripper": read_frame(caps["right_gripper"], gidx),
                "states":        states_slice[t],
                "actions":       actions_slice[t],
                "prompt":        str(base_prompt),  # ← 与(1,)一致
                "task":          str(task_name or ""),                        # ← 特殊列：帧里写，features不写
            }
            required = {"head","left_gripper","right_gripper","states","actions","prompt"}
            missing = required - set(frame_dict.keys())
            assert not missing, f"t={t}, gidx={gidx}, missing keys: {missing}"
            print("SCHEMA keys:", ds.features.keys())                     # 1) 确认 features 里真的有 'task'
            print("FIRST frame_dict keys:", frame_dict.keys())             # 2) 确认确实传了 'task'
            # 暂时先只 add 一帧试试
            ds.add_frame(frame_dict)
            print("BUFFER keys after 1 add:", ds.episode_buffer.keys())    # 3) 这里必须包含 'task'

        print("Dataset features:", ds.features.keys())

        # Pass task as metadata to save_episode() instead of adding to each frame
        # ds.save_episode(task=task_name or "")
        if ds.episode_buffer["size"] <= 0:
            print(f"[skip] episode {ep_count}: empty buffer")
            continue
        ds.save_episode()
        ep_count += 1

    for cap in caps.values():
        cap.release()

    print(f"✅ Wrote {ep_count} episodes to: {HF_LEROBOT_HOME / REPO_ID}")
    print("   (This path should contain meta/info.json now.)")

if __name__ == "__main__":
    main()
