import os
import json
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
import tqdm

# ================= 配置 =================
# 必须与 action.py 的 get_property_list 严格一致
INDEX_GRIPPER = 4
INDEX_JOINT = 8
JOINT_DIM = 14          # 关节维度
TOTAL_OUTPUT_DIM = 32   # 最终对齐到 32 维（前 14 维有效）

# 每个 episode 采样多少条用于分位数估计（防止内存爆掉）
STATE_SAMPLES_PER_EP = 2000
ACTION_SAMPLES_PER_EP = 2000


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Parallel Dataset Mean/Std/Quantile Compute")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/actions_gaussian",
    )
    parser.add_argument("--meta_file", type=str, default="meta_data.json")
    parser.add_argument("--output_json", type=str, default="dataset_stats_mp.json")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel processes")
    return parser.parse_args()


def load_data_custom(npy_path, dim_list):
    """
    Worker 进程读取单个文件的逻辑：
    - 从 action.npy 中按 dim_list 切出 Joint 段 (INDEX_JOINT)
    - 返回 shape: (T, JOINT_DIM)
    """
    try:
        expected_dim = sum(dim_list)
        with open(npy_path, "rb") as f:
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

    返回：
      (s_sum, s_sq_sum, s_count,
       a_sum, a_sq_sum, a_count,
       sample_state, sample_action)
    其中 sample_state / sample_action 用于后面估计 q01/q99。
    """
    root_dir, ep_key, dim_list = args
    npy_path = os.path.join(root_dir, ep_key, "action.npy")

    if not os.path.exists(npy_path):
        return None

    # 加载数据
    joints = load_data_custom(npy_path, dim_list)
    if joints is None or joints.shape[1] != JOINT_DIM:
        return None

    joints = joints.astype(np.float64)
    T = joints.shape[0]
    if T < 2:
        return None

    # --- 1. State Stats (Joints Abs) ---
    s_sum = np.sum(joints, axis=0)
    s_sq_sum = np.sum(joints ** 2, axis=0)
    s_count = T

    # --- 2. Action Stats (Joints Delta) ---
    joint_delta = joints[1:] - joints[:-1]
    a_sum = np.sum(joint_delta, axis=0)
    a_sq_sum = np.sum(joint_delta ** 2, axis=0)
    a_count = T - 1

    # --- 3. 为分位数估计采样 ---
    # state 采样
    if T > STATE_SAMPLES_PER_EP:
        idx_state = np.random.choice(T, STATE_SAMPLES_PER_EP, replace=False)
        sample_state = joints[idx_state]
    else:
        sample_state = joints.copy()

    # action 采样
    Td = joint_delta.shape[0]
    if Td > ACTION_SAMPLES_PER_EP:
        idx_action = np.random.choice(Td, ACTION_SAMPLES_PER_EP, replace=False)
        sample_action = joint_delta[idx_action]
    else:
        sample_action = joint_delta.copy()

    return (s_sum, s_sq_sum, s_count,
            a_sum, a_sq_sum, a_count,
            sample_state, sample_action)


def main():
    args = parse_args()

    # 1. 加载 Meta Data
    meta_path = os.path.join(args.root_dir, args.meta_file)
    print(f"📖 Loading meta data from {meta_path}...")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    episodes = list(meta_data.keys())
    total_episodes = len(episodes)

    # 2. 准备任务参数列表
    tasks = []
    for ep_key in episodes:
        tasks.append((args.root_dir, ep_key, meta_data[ep_key]["dim_list"]))

    # 3. 确定进程数
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

    # 用于分位数估计的全局采样缓存
    all_state_samples = []
    all_action_samples = []

    # 5. 并行执行
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(process_episode, tasks, chunksize=50),
                total=total_episodes,
            )
        )

    print("📊 Aggregating results...")

    # 6. 汇总结果
    for res in results:
        if res is None:
            continue

        (
            s_sum,
            s_sq_sum,
            s_cnt,
            a_sum,
            a_sq_sum,
            a_cnt,
            sample_state,
            sample_action,
        ) = res

        total_state_sum += s_sum
        total_state_sq_sum += s_sq_sum
        total_state_count += s_cnt

        total_action_sum += a_sum
        total_action_sq_sum += a_sq_sum
        total_action_count += a_cnt

        all_state_samples.append(sample_state)
        all_action_samples.append(sample_action)

        valid_files += 1

    print(f"✅ Processed {valid_files} valid files.")

    if total_state_count == 0 or total_action_count == 0:
        print("❌ No valid data found.")
        return

    # 7. 计算 Mean/Std
    state_mean = total_state_sum / total_state_count
    state_std = np.sqrt((total_state_sq_sum / total_state_count) - (state_mean ** 2) + 1e-8)

    action_mean = total_action_sum / total_action_count
    action_std = np.sqrt((total_action_sq_sum / total_action_count) - (action_mean ** 2) + 1e-8)

    # 8. 计算 q01 / q99（基于采样）
    print("📌 Computing percentiles (q01 / q99) from samples...")

    all_state_samples_np = np.vstack(all_state_samples)  # (N_state, JOINT_DIM)
    all_action_samples_np = np.vstack(all_action_samples)  # (N_action, JOINT_DIM)

    state_q01 = np.percentile(all_state_samples_np, 1, axis=0)
    state_q99 = np.percentile(all_state_samples_np, 99, axis=0)

    action_q01 = np.percentile(all_action_samples_np, 1, axis=0)
    action_q99 = np.percentile(all_action_samples_np, 99, axis=0)

    # 9. 格式化输出 (Padding 到 32 维)
    def format_output(mean_arr, std_arr, q01_arr=None, q99_arr=None):
        final_mean = np.zeros(TOTAL_OUTPUT_DIM, dtype=np.float32)
        final_std = np.ones(TOTAL_OUTPUT_DIM, dtype=np.float32)

        final_mean[:JOINT_DIM] = mean_arr.astype(np.float32)
        final_std[:JOINT_DIM] = std_arr.astype(np.float32)

        out = {
            "mean": final_mean.tolist(),
            "std": final_std.tolist(),
        }

        if q01_arr is not None and q99_arr is not None:
            final_q01 = np.zeros(TOTAL_OUTPUT_DIM, dtype=np.float32)
            final_q99 = np.zeros(TOTAL_OUTPUT_DIM, dtype=np.float32)
            final_q01[:JOINT_DIM] = q01_arr.astype(np.float32)
            final_q99[:JOINT_DIM] = q99_arr.astype(np.float32)
            out["q01"] = final_q01.tolist()
            out["q99"] = final_q99.tolist()

        return out

    state_stats = format_output(state_mean, state_std, state_q01, state_q99)
    action_stats = format_output(action_mean, action_std, action_q01, action_q99)

    stats_dict = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats,
        }
    }

    save_path = os.path.join(args.root_dir, args.output_json)
    with open(save_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print("\n" + "=" * 50)
    print(f"✅ Fast Stats saved to: {save_path}")
    print(f"   Workers used: {num_workers}")
    print(f"   State Joint Mean (First 5): {state_mean[:5]}")
    print(f"   State Joint q01  (First 5): {state_q01[:5]}")
    print(f"   State Joint q99  (First 5): {state_q99[:5]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
