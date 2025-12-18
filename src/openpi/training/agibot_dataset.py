import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

from .action import AgibotActionState

try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge('torch')
except ImportError:
    pass

class AgiBotDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        action_horizon: int = 30,
        index_filename: str = "episodic_dataset_fixed.npy",
        stats_filename: str = "dataset_stats_32dim.json",
        meta_filename: str = "actions_gaussian/meta_data.json",
        normalization: bool = True,
    ):
        self.root_dir = root_dir
        self.action_horizon = action_horizon
        self.normalization = normalization
        self.CHUNK_SIZE = 500

        self.JOINT_DIM = 14
        self.TOTAL_OUTPUT_DIM = 32
        
        env_index_file = os.getenv("AGIBOT_INDEX_FILE")
        if env_index_file:
            logging.info(f"🚩 [AgiBotDataset] Overriding index filename from ENV: {env_index_file}")
            index_filename = env_index_file
        # =========================================================

        # 1. 加载索引
        index_path = os.path.join(root_dir, index_filename)
        if not os.path.exists(index_path):
            index_path = os.path.join(root_dir, "episodic_dataset_fixed.npy")
        
        logging.info(f"[AgiBotDataset] Loading index from: {index_path}")
        meta_data = np.load(index_path, allow_pickle=True).item()
        
        self.video_paths = meta_data["video_path"]
        self.instructions = meta_data["instructions"]
        self.start_end = meta_data["start_end"]
        
        self.episode_lengths = self.start_end[:, 1] - self.start_end[:, 0]
        self.cumulative_lengths = np.cumsum(self.episode_lengths)

        print("cumulative_lengths.shape:", self.cumulative_lengths.shape)
        print("cumulative_lengths.ndim :", self.cumulative_lengths.ndim)
        print("num_episodes:", len(self.cumulative_lengths))

        self.total_frames = self.cumulative_lengths[-1]

        print("total_frames in dataset:", self.total_frames)
        
        # 2. 加载元数据 (获取 dim_list 和 total_length)
        meta_json_path = os.path.join(root_dir, meta_filename)
        logging.info(f"[AgiBotDataset] Loading metadata from: {meta_json_path}")
        with open(meta_json_path, "r") as f:
            self.dims_meta = json.load(f)

        # 3. 加载统计数据 [关键修改]
        if self.normalization:
            # 初始化默认值 (以防加载失败)
            self.state_mean = torch.zeros(32)
            self.state_std = torch.ones(32)
            self.action_mean = torch.zeros(32)
            self.action_std = torch.ones(32)
            
            stats_path = os.getenv("NORM_STATS_FILE")
            
            # --- [Modification 1: Print the EXACT path being used] ---
            logging.info(f"🔍 [AgiBotDataset] ------------------------------------------------")
            logging.info(f"🔍 [AgiBotDataset] Loading stats from: {stats_path}")
            # ---------------------------------------------------------

            if not stats_path:
                error_msg = "❌ [AgiBotDataset] Environment variable 'NORM_STATS_FILE' is NOT set!"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            logging.info(f"🔍 [AgiBotDataset] Loading stats from: {stats_path}")
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Stats file not found: {stats_path}")

            try:
                with open(stats_path, 'r') as f:
                    full_stats = json.load(f)
                
                if "norm_stats" in full_stats:
                    stats = full_stats["norm_stats"]
                else:
                    stats = full_stats 

                # --- A. Load STATE Stats ---
                if "state" in stats:
                    if "q01" in stats["state"] and "q99" in stats["state"]:
                        self.state_q01 = torch.tensor(stats["state"]["q01"], dtype=torch.float32)
                        self.state_q99 = torch.tensor(stats["state"]["q99"], dtype=torch.float32)
                        
                        # --- [Modification 2: Print State Values] ---
                        logging.info(f"✅ [AgiBotDataset] Loaded STATE q01/q99")
                        logging.info(f"   --> State q01 (first 5 dims): {self.state_q01[:5].tolist()}")
                        logging.info(f"   --> State q99 (first 5 dims): {self.state_q99[:5].tolist()}")
                        # --------------------------------------------
                    else:
                         logging.warning("⚠️ [AgiBotDataset] 'q01/q99' missing in state stats.")

                # --- B. Load ACTION Stats ---
                if "actions" in stats:
                     if "q01" in stats["actions"] and "q99" in stats["actions"]:
                        self.action_q01 = torch.tensor(stats["actions"]["q01"], dtype=torch.float32)
                        self.action_q99 = torch.tensor(stats["actions"]["q99"], dtype=torch.float32)
                        
                        # --- [Modification 3: Print Action Values] ---
                        logging.info(f"✅ [AgiBotDataset] Loaded ACTION q01/q99")
                        logging.info(f"   --> Action q01 (first 5 dims): {self.action_q01[:5].tolist()}")
                        logging.info(f"   --> Action q99 (first 5 dims): {self.action_q99[:5].tolist()}")
                        # ---------------------------------------------
                     else:
                        logging.warning("⚠️ [AgiBotDataset] 'q01/q99' missing in action stats.")
                
                logging.info(f"🔍 [AgiBotDataset] ------------------------------------------------")
                    
            except Exception as e:
                logging.error(f"❌ [AgiBotDataset] JSON Decode Error in {stats_path}: {e}")
                raise e

            # while True:
            #     pass

    def __len__(self):
        return self.total_frames

    def _get_info_by_idx(self, global_idx):
        ep_idx = np.searchsorted(self.cumulative_lengths, global_idx, side='right')

        print("cumulative_lengths:", self.cumulative_lengths)


        if ep_idx == 0:
            frame_idx_in_ep = global_idx
        else:
            frame_idx_in_ep = global_idx - self.cumulative_lengths[ep_idx - 1]
        return ep_idx, frame_idx_in_ep

    def _load_video_frame(self, video_folder, view_prefix, abs_frame_idx):
            chunk_idx = abs_frame_idx // self.CHUNK_SIZE
            local_idx = abs_frame_idx % self.CHUNK_SIZE
            video_name = f"{view_prefix}_{chunk_idx}.mp4"
            video_path = os.path.join(video_folder, video_name)
            
            # [Debug] 检查路径
            if not os.path.exists(video_path):
                # print(f"❌ [DEBUG] File NOT Found: {video_path}")
                return torch.zeros((3, 224, 224), dtype=torch.float32)
                
            try:
                # width=224, height=224 让 decord 自动缩放
                # vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
                vr = VideoReader(video_path, ctx=cpu(0))
                idx = min(local_idx, len(vr) - 1)
                
                tensor_img = vr[idx] 
                
                # ❌ [删除这行] return tensor_img.permute(2, 0, 1).float() / 255.0
                
                # ✅ [改为这行] 保持 (H, W, C) 格式，直接归一化即可
                return tensor_img.float() / 255.0

            except Exception as e:
                # ... (错误处理) ...
                # ❌ [删除] return torch.zeros((3, 224, 224), dtype=torch.float32)
                # ✅ [修改] 保持一致的 (H, W, C) 格式
                return torch.zeros((224, 224, 3), dtype=torch.float32)

    def __getitem__(self, idx):


        # 1. 基础信息
        ep_idx, idx_in_seg = self._get_info_by_idx(idx)

        # print("idx:", idx, "ep_idx:", ep_idx, "idx_in_seg:", idx_in_seg)
        
        rel_path = self.video_paths[ep_idx] # "327/648642.mp4"
        rel_path_no_ext = os.path.splitext(rel_path)[0] # "327/648642"
        
        start_frame, end_frame = self.start_end[ep_idx]
        current_abs_frame = start_frame + idx_in_seg


        # print("start_frame:", start_frame, "end_frame:", end_frame, "current_abs_frame:", current_abs_frame)
        
        # 2. 获取该 Episode 的 Metadata
        if rel_path_no_ext in self.dims_meta:
            meta_info = self.dims_meta[rel_path_no_ext]
            dim_list = meta_info["dim_list"]
            total_file_length = meta_info["length"] # 这一集的总帧数 (action.npy 的行数)
        else:
            raise ValueError(f"Meta info not found for {rel_path_no_ext}")

        # ==========================================
        # 3. 读取视频
        # ==========================================
        video_folder = os.path.join(self.root_dir, "videos_h264", rel_path_no_ext, "videos")
        img_head = self._load_video_frame(video_folder, "head_color", current_abs_frame)
        img_left = self._load_video_frame(video_folder, "hand_left_color", current_abs_frame)
        img_right = self._load_video_frame(video_folder, "hand_right_color", current_abs_frame)

        # ==========================================
        # 4. 读取 Action & State (使用 Helper 类)
        # ==========================================
        action_path = os.path.join(self.root_dir, "actions_gaussian", rel_path_no_ext, "action.npy")

        # print("Action Path:", action_path)
        
        # 计算读取范围
        # 我们要读 [current, current + horizon]
        # 但必须限制在 total_file_length 以内，否则 load_range 里的 reshape 会报错
        read_start = current_abs_frame
        read_end = min(current_abs_frame + self.action_horizon, total_file_length)
        
        # [修改 2] 使用 AgibotActionState 读取
        # 这是一个轻量级的 seek + read，非常快
        try:
            action_obj = AgibotActionState.load_range_from_path(
                path=action_path,
                dim_list=dim_list,
                start=read_start,
                end=read_end
            )
        except Exception as e:
            # 容错：如果读取失败，返回全0
            logging.error(f"Error reading {action_path}: {e}")
            # 创建一个空的 action_obj 结构
            action_obj = AgibotActionState()
            # 填充全0数据 (L, Dim)
            fake_len = read_end - read_start
            action_obj.state_joint_position = np.zeros((fake_len, 14), dtype=np.float32)
            action_obj.action_effector_position = np.zeros((fake_len, 2), dtype=np.float32)

        # ==========================================
        # 5. 组装数据 (State & Action)
        # ==========================================
        
        # A. 当前 State (Abs)
        # State = State_joint_position (14) + Action_effector_position (2)
        # 取 index 0 (即 current_abs_frame)
        curr_joint = action_obj.state_joint_position[0]
        curr_gripper = action_obj.action_effector_position[0]
        
        state_t_16 = np.concatenate([curr_joint, curr_gripper]) # (16,)

        # B. 未来 Actions (Delta/Abs)
        # Future Joint (14) & Future Gripper (2)
        future_joints = action_obj.state_joint_position
        future_gripper = action_obj.action_effector_position
        
        # Delta Joint = Future - Current
        joint_delta = future_joints - curr_joint
        
        # Action = [Joint Delta (14), Gripper Abs (2)]
        actions_16 = np.concatenate([joint_delta, future_gripper], axis=1) # (L, 16)
        valid_len = actions_16.shape[0]

        # ==========================================
        # 6. Padding & Output
        # ==========================================
        
        # State Padding
        state_final = torch.zeros(32, dtype=torch.float32)
        state_final[:16] = torch.from_numpy(state_t_16)
        
        # Action Padding
        actions_final = torch.zeros((self.action_horizon, 32), dtype=torch.float32)
        actions_segment = torch.from_numpy(actions_16)
        
        actions_final[:valid_len, :16] = actions_segment
        
        # Repeat Padding (补齐不足 Horizon 的部分)
        if valid_len < self.action_horizon:
            actions_final[valid_len:, :16] = actions_segment[-1]
            
        action_is_pad = torch.zeros(self.action_horizon, dtype=torch.bool)
        action_is_pad[valid_len:] = True

        # ==========================================
        # 7. 提取 State Head & Waist (仅当前帧)
        # ==========================================
        state_head = torch.from_numpy(action_obj.state_head_position[0])   # (2,)
        state_waist = torch.from_numpy(action_obj.state_waist_position[0]) # (2,)

        # ==========================================================
        # 7. Normalization using q01 / q99 (MODIFIED)
        # Formula: 2 * (x - q01) / (q99 - q01) - 1
        # ==========================================================

        if self.normalization:
            # 只对关节 14 维做 q01/q99 归一化
            VALID_DIMS_JOINT = self.JOINT_DIM  # = 14

            # ---- STATE: 只归一化 state_final[:14] ----
            state_q01 = self.state_q01[:VALID_DIMS_JOINT]
            state_q99 = self.state_q99[:VALID_DIMS_JOINT]

            state_denom = state_q99 - state_q01
            state_denom = torch.where(
                torch.abs(state_denom) < 1e-3,
                torch.full_like(state_denom, 1e-3),
                state_denom,
            )

            state_joint = state_final[:VALID_DIMS_JOINT]
            state_joint = 2 * (state_joint - state_q01) / state_denom - 1
            state_joint = torch.clamp(state_joint, -10.0, 10.0)
            state_final[:VALID_DIMS_JOINT] = state_joint
            # state_final[14:16] 是 gripper，保持原值（或你后面想单独处理）
            # state_final[16:] 是 padding，不归一化

            # ---- ACTIONS: 只归一化 actions_final[:, :14] ----
            action_q01 = self.action_q01[:VALID_DIMS_JOINT]
            action_q99 = self.action_q99[:VALID_DIMS_JOINT]

            action_denom = action_q99 - action_q01
            action_denom = torch.where(
                torch.abs(action_denom) < 1e-3,
                torch.full_like(action_denom, 1e-3),
                action_denom,
            )

            act_joint = actions_final[:, :VALID_DIMS_JOINT]
            act_joint = 2 * (act_joint - action_q01) / action_denom - 1
            act_joint = torch.clamp(act_joint, -10.0, 10.0)
            actions_final[:, :VALID_DIMS_JOINT] = act_joint
            # actions_final[:, 14:16] 是 gripper，保持原值
            # actions_final[:, 16:] 是 padding，不归一化


        # # 构造最终返回的字典
        # result = {
        #     "head": img_head,
        #     "left_gripper": img_left,
        #     "right_gripper": img_right,
        #     "states": state_final,
        #     "actions": actions_final,
        #     "prompt": self.instructions[ep_idx],
        #     # Meta
        #     "episode_index": torch.tensor(ep_idx),
        #     "frame_index": torch.tensor(current_abs_frame),
        #     "timestamp": torch.tensor(current_abs_frame / 30.0),
        #     "next.done": torch.tensor(current_abs_frame == end_frame - 1),
        #     "action_is_pad": action_is_pad
        # }

        # print("\n" + "="*60)
        # print(f"🔍 [Dataset Debug] Index: {idx} | RelPath: {rel_path}")
        
        # # ▼▼▼▼▼▼▼▼▼▼ [新增] 打印 Action/State 读取路径 ▼▼▼▼▼▼▼▼▼▼
        # print(f"   📂 Action/State File Path: {action_path}") 
        # # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # # --- 1. 打印 Prompt (最关键) ---
        # print(f"   📝 Prompt: \"{result['prompt']}\"")

        # # --- 2. 打印 Image 采样 (检查是否全黑) ---
        # print("   🖼️  Image Check (Center Pixel [112,112] RGB):")
        # for k in ["head", "left_gripper", "right_gripper"]:
        #     if k in result:
        #         img = result[k]
        #         # 取中间一个像素的值，保留4位小数
        #         center_px = img[:, 112, 112].tolist() 
        #         formatted_px = [round(x, 4) for x in center_px]
        #         print(f"      |-- {k:<15}: {formatted_px} (Min: {img.min():.4f}, Max: {img.max():.4f})")

        # # --- 3. 打印 State/Action 具体数值 (完整版) ---
        # print("   🔢 FULL Vector Values:")
        
        # # (A) 打印完整的 State (32维)
        # states_list = result['states'].tolist()
        # # 格式化一下，保留4位小数，方便阅读
        # states_str = ", ".join([f"{x: .4f}" for x in states_list])
        # print(f"      |-- states (32):")
        # print(f"          [{states_str}]")
        
        # # (B) 打印完整的 Actions (30 x 32)
        # # 我们逐帧打印，这样你看得清楚每一帧的变化
        # actions_np = result['actions'].numpy()
        # print(f"      |-- actions ({actions_np.shape}):")
        # for t in range(actions_np.shape[0]):
        #     # 获取当前时间步的 32 维向量
        #     act_row = actions_np[t]
        #     # 格式化字符串
        #     row_str = ", ".join([f"{x: .4f}" for x in act_row])
            
        #     # 检查这一行是不是全 0 (辅助判断)
        #     is_zero = np.allclose(act_row, 0, atol=1e-5)
        #     zero_tag = "⚠️ ALL ZERO" if is_zero else ""
            
        #     print(f"          [Step {t:02d}]: {row_str} {zero_tag}")
        
        # print("="*60 + "\n")

        # while True:
        #     pass

        return {
            "head": img_head,
            "left_gripper": img_left,
            "right_gripper": img_right,
            "head_mask": torch.tensor(True),
            "left_gripper_mask": torch.tensor(True),
            "right_gripper_mask": torch.tensor(True),
            "states": state_final,
            "actions": actions_final,

            # Auxiliary Data (Current Frame Only)
            "state_head": state_head,     # (2,)
            "state_waist": state_waist,   # (2,)

            "prompt": self.instructions[ep_idx],
            # Meta
            "episode_index": torch.tensor(ep_idx),
            "frame_index": torch.tensor(current_abs_frame),
            "timestamp": torch.tensor(current_abs_frame / 30.0),
            "next.done": torch.tensor(current_abs_frame == end_frame - 1),
            "action_is_pad": action_is_pad
        }-