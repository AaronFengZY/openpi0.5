import os
import json
import numpy as np
# 确保 action.py 在当前目录下，或者在 python 的搜索路径中
from action import AgibotActionState 

# 1. 路径配置 (保持不变)
file_path = '/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/actions/327/648642/action.npy'
meta_path = '/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/actions_gaussian/meta_data.json'

# 2. 获取 dim_list
# 既然你上次运行已经确认了维度是这个，我们可以直接硬编码写死，或者继续从文件读
# 为了稳妥，这里还是从文件读，逻辑和你上次成功的一样
try:
    with open(meta_path, 'r') as f:
        meta_all = json.load(f)
        key = "327/648642"
        if key in meta_all:
            dim_list = meta_all[key]["dim_list"]
            # print(f"✅ Metadata Loaded: {dim_list}")
        else:
            print(f"❌ Key {key} not found in metadata.")
            exit()
except Exception as e:
    print(f"读取 Meta 失败: {e}")
    exit()

# 3. 读取指定范围 [185, 190)
# 这将读取: 185, 186, 187, 188, 189 这5帧
start_idx = 185
end_idx = 190

print(f"正在读取帧范围: {start_idx} 到 {end_idx-1} ...")

try:
    action_obj = AgibotActionState.load_range_from_path(
        path=file_path,
        dim_list=dim_list,
        start=start_idx,
        end=end_idx 
    )

    print("\n=== State Joint Position (Frames 185-189) ===")
    # 打印的时候加上索引方便对照
    for i, row in enumerate(action_obj.state_joint_position):
        print(f"[Frame {start_idx + i}]:")
        print(row)
        print("-" * 40)

except Exception as e:
    print(f"\n❌ 读取失败: {e}")