import numpy as np
import os

# 你的文件路径
file_path = '/home/v-zhifeng/HPE/openpi/episodic_dataset_train.npy'

def inspect_data(data, indent=0):
    """递归打印数据结构的辅助函数"""
    prefix = "  " * indent
    
    # 1. 处理 Numpy 数组
    if isinstance(data, np.ndarray):
        # 如果是 0 维数组 (通常包裹着 dict 或 list)
        if data.ndim == 0:
            print(f"{prefix}📦 Wrapped Object (0-d array)")
            # 取出内容继续递归
            inspect_data(data.item(), indent)
        else:
            print(f"{prefix}📐 Array | Shape: {data.shape} | Dtype: {data.dtype}")
            # 如果比较小，可以打印预览
            if data.size < 10:
                print(f"{prefix}   Value: {data}")
    
    # 2. 处理字典 (常见的 dataset 格式)
    elif isinstance(data, dict):
        print(f"{prefix}🔑 Dict with {len(data)} keys:")
        for key, value in data.items():
            print(f"{prefix}   - Key: '{key}'")
            inspect_data(value, indent + 2)
            
    # 3. 处理列表/元组
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}📜 {type(data).__name__} with length: {len(data)}")
        if len(data) > 0:
            print(f"{prefix}   Checking first element:")
            inspect_data(data[0], indent + 2)
            
    # 4. 其他类型
    else:
        print(f"{prefix}📄 Value: {data} ({type(data)})")

def main():
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件: {file_path}")
        return

    print(f"🔄 正在加载: {file_path} ...")
    
    try:
        # allow_pickle=True 是必须的，因为复杂数据集通常包含 pickled 对象
        content = np.load(file_path, allow_pickle=True)
        print("✅ 加载成功！文件内容结构如下：\n")
        print("="*40)
        inspect_data(content)
        print("="*40)
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    main()