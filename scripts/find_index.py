import numpy as np
import os

# 你提供的特定文件路径
FILE_PATH = "/home/v-zhifeng/HPE/v-zhifeng/agibot_beta_split_500/episodic_dataset_fixed_static.npy"
# 你要查找的目标 ID
TARGET_ID = "327/648642"

def main():
    print(f"📂 Loading dataset from: {FILE_PATH}")
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found!")
        return

    # 加载 .npy 文件
    #通常这类数据集是保存为一个字典对象的
    try:
        data = np.load(FILE_PATH, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return

    # 获取视频路径列表
    if "video_path" not in data:
        print("❌ Error: Key 'video_path' not found in dataset keys:", data.keys())
        return

    video_paths = data["video_path"]
    print(f"🔍 Searching for '{TARGET_ID}' in {len(video_paths)} episodes...")

    found_count = 0

    # 遍历查找
    for i, path in enumerate(video_paths):
        # 只要 path 字符串里包含了 "327/648642" 就算找到
        if TARGET_ID in path:
            print("\n" + "="*50)
            print(f"✅ FOUND MATCH at Index: {i}")
            print("-" * 30)
            print(f"   📂 Full Path:   {path}")
            
            # 顺便打印一下对应的 Instruction 和 Start/End，方便你确认
            if "instructions" in data:
                print(f"   📝 Instruction: \"{data['instructions'][i]}\"")
            
            if "start_end" in data:
                print(f"   ⏱️  Frame Range: {data['start_end'][i]}")
                
            print("="*50)
            found_count += 1

    if found_count == 0:
        print(f"\n❌ Not found. The ID '{TARGET_ID}' is not in the list.")
    else:
        print(f"\n✨ Total matches found: {found_count}")

if __name__ == "__main__":
    main()