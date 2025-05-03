import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arrow, FancyArrowPatch
from pylab import mpl
def plot_hub_spoke_model():
    plt.figure(figsize=(8, 8))
    
    # 绘制集散中心
    hub = plt.Circle((0, 0), 0.5, color='red', alpha=0.8)
    plt.gca().add_patch(hub)
    plt.text(0, 0, '集散中心', ha='center', va='center', color='white')
    
    # 随机生成接收点
    n_receivers = 8
    receiver_positions = []
    for i in range(n_receivers):
        angle = np.random.rand()*0.5 + i * (2 * np.pi / n_receivers)
        radius = 5 * (0.7 + np.random.rand() * 0.6)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        receiver_positions.append((x, y))

    
    # 绘制接收点和连接线
    for i, (x, y) in enumerate(receiver_positions):
        # 接收点
        receiver = plt.Circle((x, y), 0.4, color='cyan', alpha=0.6)
        plt.gca().add_patch(receiver)
        plt.text(x, y, f'接收点{i+1}', ha='center', va='center', color='black')
        
        # 连接线
        plt.plot([0, x], [0, y], 'k--', alpha=0.4)
        

    
    # 随机选择几个接收点高亮显示
    selected = np.random.choice(range(n_receivers), size=3, replace=False)
    for idx in selected:
        x, y = receiver_positions[idx]
        arrow = Arrow(0, 0, x*0.9, y*0.9, width=0.2, color='blue', alpha=0.6)
        plt.gca().add_patch(arrow)
    
    plt.title("“点对点”类低空经济工作形式", pad=20)
    plt.xlim(-radius-1, radius+1)
    plt.ylim(-radius-1, radius+1)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_base_station_model():
    plt.figure(figsize=(10, 8))
    
    # 绘制基站
    base = plt.Circle((0, 0), 0.6, color='purple', alpha=0.8)
    plt.gca().add_patch(base)
    plt.text(0, 0, '基站', ha='center', va='center', color='white')
    
    # 生成不确定的发送点和接收点
    n_pairs = 5
    angles = np.linspace(0, 2*np.pi, n_pairs, endpoint=False)
    radius = 5
    
    for i, angle in enumerate(angles):
        # 发送点位置
        sender_r = radius * (0.7 + np.random.rand()*0.6)
        sender_x, sender_y = sender_r*np.cos(angle), sender_r*np.sin(angle)
        
        # 接收点位置 (与发送点有一定偏移)
        receiver_r = radius * (0.7 + np.random.rand()*0.6)
        receiver_angle = angle + (np.random.rand()-0.5)*2
        receiver_x, receiver_y = receiver_r*np.cos(receiver_angle), receiver_r*np.sin(receiver_angle)
        
        # 绘制发送点
        sender = plt.Circle((sender_x, sender_y), 0.25, color='orange', alpha=0.7)
        plt.gca().add_patch(sender)
        plt.text(sender_x, sender_y, f'S{i+1}', ha='center', va='center', fontsize=8)
        
        # 绘制接收点
        receiver = plt.Circle((receiver_x, receiver_y), 0.25, color='cyan', alpha=0.7)
        plt.gca().add_patch(receiver)
        plt.text(receiver_x, receiver_y, f'R{i+1}', ha='center', va='center', fontsize=8)
        
        # 绘制完整的请求路径: 基站 -> 发送点 -> 接收点 -> 基站
        # 基站到发送点
        plt.plot([0, sender_x], [0, sender_y], 'b--', alpha=0.3)
        arrow1 = FancyArrowPatch((0, 0), (sender_x*0.95, sender_y*0.95), 
                                arrowstyle='->', color='b', mutation_scale=15, alpha=0.5)
        plt.gca().add_patch(arrow1)
        
        # 发送点到接收点
        plt.plot([sender_x, receiver_x], [sender_y, receiver_y], 'g--', alpha=0.3)
        arrow2 = FancyArrowPatch((sender_x, sender_y), (receiver_x*0.95, receiver_y*0.95), 
                                arrowstyle='->', color='g', mutation_scale=15, alpha=0.5)
        plt.gca().add_patch(arrow2)
        
        # 接收点到基站
        plt.plot([receiver_x, 0], [receiver_y, 0], 'r--', alpha=0.3)
        arrow3 = FancyArrowPatch((receiver_x, receiver_y), (0, 0), 
                                arrowstyle='->', color='r', mutation_scale=15, alpha=0.5)
        plt.gca().add_patch(arrow3)
    
    plt.title("“基站-发送-接收”类低空经济工作形式", pad=20)
    plt.xlim(-radius-1, radius+1)
    plt.ylim(-radius-1, radius+1)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
plot_base_station_model()

plot_hub_spoke_model()