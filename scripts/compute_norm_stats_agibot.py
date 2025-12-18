import os
import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

# ================= 配置 =================
# 必须与 action.py 的 get_property_list 严格一致
INDEX_GRIPPER = 4
INDEX_JOINT = 8
JOINT_DIM = 14
TOTAL_OUTPUT_DIM = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Fast Parallel Dataset Mean/Std Compute")
    parser.add_argument("--root_dir", type=str, 
                        default="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/actions_gaussian")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")
    parser.add_argument("--output_json", type=str, default="dataset_stats_mp.json")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel processes")
    return parser.parse_args()

def load_data_custom(npy_path, dim_list):
    """
    Worker 进程读取单个文件的逻辑
    """
    try:
        expected_dim = sum(dim_list)
        with open(npy_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.float32)
        
        if raw_data.size % expected_dim != 0:
            return None
            
        T = raw_data.size // expected_dim
        matrix = raw_data.reshape(T, expected_dim)
        
        # 快速切片提取 Joint 数据
        current_idx = 0
        for i, dim in enumerate(dim_list):
            if i == INDEX_JOINT:
                return matrix[:, current_idx : current_idx + dim]
            current_idx += dim
        return None
    except Exception:
        return None

def process_episode(args):
    """
    Worker 函数：处理单个 Episode，返回统计量的中间结果 (Sum, SqSum, Count)
    参数 args 是一个 tuple: (root_dir, ep_key, dim_list)
    """
    root_dir, ep_key, dim_list = args
    npy_path = os.path.join(root_dir, ep_key, "action.npy")
    
    if not os.path.exists(npy_path):
        return None

    # 加载数据
    joints = load_data_custom(npy_path, dim_list)
    if joints is None or joints.shape[1] != JOINT_DIM:
        return None

    # 转 float64 保证累加精度
    joints = joints.astype(np.float64)
    T = joints.shape[0]
    
    if T < 2: return None

    # --- 1. State Stats (Joints Abs) ---
    s_sum = np.sum(joints, axis=0)
    s_sq_sum = np.sum(joints ** 2, axis=0)
    s_count = T

    # --- 2. Action Stats (Joints Delta) ---
    joint_delta = joints[1:] - joints[:-1]
    a_sum = np.sum(joint_delta, axis=0)
    a_sq_sum = np.sum(joint_delta ** 2, axis=0)
    a_count = T - 1

    return (s_sum, s_sq_sum, s_count, a_sum, a_sq_sum, a_count)

def main():
    args = parse_args()
    
    # 1. 加载 Meta Data
    meta_path = os.path.join(args.root_dir, args.meta_file)
    print(f"📖 Loading meta data from {meta_path}...")
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    episodes = list(meta_data.keys())
    total_episodes = len(episodes)
    
    # 2. 准备任务参数列表
    # 将不需要的大字典解耦，只传必要参数给 worker
    tasks = []
    for ep_key in episodes:
        tasks.append((args.root_dir, ep_key, meta_data[ep_key]["dim_list"]))
    
    # 3. 确定进程数
    # 如果没指定，默认使用 CPU 核心数 - 2 (留点余量)
    num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 2)
    print(f"🚀 Starting multiprocessing with {num_workers} workers for {total_episodes} episodes...")

    # 4. 全局累加器 (float64)
    total_state_sum = np.zeros(JOINT_DIM, dtype=np.float64)
    total_state_sq_sum = np.zeros(JOINT_DIM, dtype=np.float64)
    total_state_count = 0

    total_action_sum = np.zeros(JOINT_DIM, dtype=np.float64)
    total_action_sq_sum = np.zeros(JOINT_DIM, dtype=np.float64)
    total_action_count = 0

    valid_files = 0

    # 5. 并行执行
    # chunksize 稍微设大一点 (例如 100)，减少进程间通信开销
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 获取结果，配合 tqdm 显示进度
        results = list(tqdm.tqdm(pool.imap_unordered(process_episode, tasks, chunksize=50), total=total_episodes))

    print("📊 Aggregating results...")
    
    # 6. 汇总结果
    for res in results:
        if res is None:
            continue
            
        (s_sum, s_sq_sum, s_cnt, a_sum, a_sq_sum, a_cnt) = res
        
        total_state_sum += s_sum
        total_state_sq_sum += s_sq_sum
        total_state_count += s_cnt
        
        total_action_sum += a_sum
        total_action_sq_sum += a_sq_sum
        total_action_count += a_cnt
        
        valid_files += 1

    print(f"✅ Processed {valid_files} valid files.")

    if total_state_count == 0:
        print("❌ No valid data found.")
        return

    # 7. 计算 Mean/Std
    state_mean = total_state_sum / total_state_count
    state_std = np.sqrt((total_state_sq_sum / total_state_count) - (state_mean ** 2) + 1e-8)

    action_mean = total_action_sum / total_action_count
    action_std = np.sqrt((total_action_sq_sum / total_action_count) - (action_mean ** 2) + 1e-8)

    # 8. 格式化输出 (Padding)
    def format_output(mean_arr, std_arr):
        final_mean = np.zeros(TOTAL_OUTPUT_DIM, dtype=np.float32)
        final_std = np.ones(TOTAL_OUTPUT_DIM, dtype=np.float32)
        
        # 填入计算结果，转回 float32 存 JSON
        final_mean[:JOINT_DIM] = mean_arr.astype(np.float32)
        final_std[:JOINT_DIM] = std_arr.astype(np.float32)
        
        return final_mean.tolist(), final_std.tolist()

    st_mean_list, st_std_list = format_output(state_mean, state_std)
    act_mean_list, act_std_list = format_output(action_mean, action_std)

    stats_dict = {
        "norm_stats": {
            "state": {
                "mean": st_mean_list,
                "std":  st_std_list
            },
            "actions": {
                "mean": act_mean_list,
                "std":  act_std_list
            }
        }
    }

    save_path = os.path.join(args.root_dir, args.output_json)
    with open(save_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    print("\n" + "="*50)
    print(f"✅ Fast Stats saved to: {save_path}")
    print(f"   Workers used: {num_workers}")
    print(f"   Joint Mean (First 5): {state_mean[:5]}")
    print("="*50)

if __name__ == "__main__":
    main()