def parse_txt_file(file_path):
    """
    解析特定格式的txt文件，返回字典列表
    
    参数:
        file_path (str): txt文件路径
        
    返回:
        list: 包含字典的列表，每个字典代表一行数据
    """
    data_list = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取第一行获取键名
        keys = file.readline().strip().split()
        
        # 读取剩余行作为数据
        for line in file:
            values = line.strip().split()
            # 确保键和值数量匹配
            if len(values) == len(keys):
                data_dict = dict(zip(keys, values))
                # 将值转换为浮点数对应的key为 可开放容量(kWh)
                data_dict["可开放容量(kWh)"] = float(data_dict["可开放容量(kWh)"])
                data_list.append(data_dict)
            else:
                print(f"警告: 数据行 '{line}' 的列数与键数不匹配，已跳过")
    
    return data_list

import pandas as pd
import matplotlib.pyplot as plt

#保存到csv文件
def save_to_csv(data_list, csv_file_path):
    """
    将字典列表保存为CSV文件
    
    参数:
        data_list (list): 包含字典的列表
        csv_file_path (str): 输出CSV文件路径
    """
    df = pd.DataFrame(data_list)
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存到 {csv_file_path}")
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
 

def plot_range_distribution(data):
    # 确定最大值和区间划分
    max_val = max(data)
    bins = np.arange(0, max_val + 500, 500)  # 创建0,500,1000,...的区间边界
    
    # 计算每个区间的数量
    counts, _ = np.histogram(data, bins=bins)
    
    # 计算占比
    total = len(data)
    percentages = (counts / total) * 100
    
    # 创建区间标签
    labels = [f"{int(bins[i])}~{int(bins[i+1])}" for i in range(len(bins)-1)]
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, percentages, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    # 设置图表标题和标签
    plt.title('数值区间分布占比', fontsize=15)
    plt.xlabel('数值区间', fontsize=12)
    plt.ylabel('占比 (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以便阅读
    
    # 自动调整布局
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    file_path = "hdev-depot-charging-2021/src/111.txt"  # 替换为你的文件路径
    result = parse_txt_file(file_path)
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    
    # 保存为CSV文件
    csv_file_path = "hdev-depot-charging-2021/src/111.csv"  # 替换为你想保存的CSV文件路径
    save_to_csv(result, csv_file_path)
    
    power_list = []
    for item in result:
        power_list.append(0.5*item["可开放容量(kWh)"])
    
    #柱状图绘图：从300KW到5100KW，间隔300KW
    plot_range_distribution(power_list)
    
