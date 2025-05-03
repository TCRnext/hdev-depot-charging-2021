from ast import Num
from matplotlib import use
import drone_module
import scene_module
from Wuhan_realmap import WuhanMapProcessor
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
processor = WuhanMapProcessor("hdev-depot-charging-2021/img_src/wuhan_map.png")
use_real_map = False
submap, points_in_range = processor.get_submap(114.27538, 30.582784, 8)
rx_weight = []
tx_weight = []
mix_weight = []
rx_list = []
tx_list = []
mix_list = []
Num_rx = 0
Num_tx = 0
Num_mix = 0
for point in points_in_range:
    if point['type'] == 0:
        rx_weight.append(point['weight'])
        rx_list.append([point['x'],point['y']])
        Num_rx += 1
        print("接收点坐标：",point['x'],point['y'],"标签：",point['label'])
    elif point['type'] == 1:
        tx_weight.append(point['weight'])
        tx_list.append([point['x'],point['y']])
        Num_tx += 1
        print("发送点坐标：",point['x'],point['y'],"标签：",point['label'])
    elif point['type'] == 2:
        mix_weight.append(point['weight'])
        mix_list.append([point['x'],point['y']])
        Num_mix += 1
        print("混合点坐标：",point['x'],point['y'],"标签：",point['label'])


if use_real_map:
    scene = scene_module.Taxi_generator(per_step_base_generate=1.0,
                                        num_clusters_mixed=Num_mix,
                                        num_clusters_tx=Num_tx,
                                        num_clusters_rx=Num_rx,
                                        tx_cluster_weight=tx_weight,
                                        rx_cluster_weight=rx_weight,
                                        mixed_cluster_weight=mix_weight,
                                        is_real_geometry=True,
                                        real_geometry_mix_list=mix_list,
                                        real_geometry_tx_list=tx_list,
                                        real_geometry_rx_list=rx_list)
else:
    scene = scene_module.Taxi_generator(per_step_base_generate=1.0)
    
output = []
heatmapdata = []
for i in range(6*60,24*60):
    piece = scene.generate(i)
    for j in piece:
        output.append(j)
        heatmapdata.append([j['tx'][0],j['tx'][1]])
        heatmapdata.append([j['rx'][0],j['rx'][1]])

result_img = processor.create_heatmap(
    submap, 
    points_in_range, 
    heatmapdata, 
    8
)
if use_real_map:
    result_img.save("hdev-depot-charging-2021/img/result_map.png")

#分析output
#格式例如[{'tx': [-0.8988601974603858, -0.987957248589785], 'rx': [1.8157597328294663, -1.849872504747552], 'time': 1439, 'distance': 7.45353555966014}, {'tx': [-0.05436485269461411, -0.10924260408540087, -0.12661694079825914, -1.3876057902272034], 'rx': [-0.0898803359189948, 0.03969300460469238, -0.43667173185315367, -0.3366504343004024], 'time': 1439, 'distance': 0.4107277910371912}, {'tx': [-1.1617822460152736, 1.626022927981384], 'rx': [0.03326393963264174, 0.18541767588446056, -0.43667173185315367, -0.3366504343004024], 'time': 1439, 'distance': 4.4644133261945695}, {'tx': [1.91170027856918, -2.1905845484793676], 'rx': [-2.0903032461943556, -0.2698340250816891], 'time': 1439, 'distance': 10.399579538708249}]
#1. 一共有多少个订单
#2. 按照时间来绘制订单的分布
#3. 绘制订单的距离分布

#1. 一共有多少个订单
print("订单共有",len(output),"个")
print("平均每分钟的订单数",len(output)/(24*60-6*60))     
print("平均订单距离",sum([i['distance'] for i in output])/len(output))


#2. 按照时间来绘制订单的分布
#plt横轴修改为小时，而不是分钟
import matplotlib.pyplot as plt
time_minutes = []
for i in output:
    time_minutes.append(i['time'])
time_hours = [t/60 for t in time_minutes]
plt.hist(time_hours, range=(6,24), bins=18*6)
plt.xlabel('hour')
plt.ylabel('number of orders')
plt.title('order distribution in time')
plt.show()




#3. 绘制订单的距离分布
distance = []
for i in output:
    distance.append(i['distance'])
plt.hist(distance, range=(0,40), bins=100)
plt.xlabel('distance')
plt.ylabel('number of orders')
plt.title('order distribution in distance')
plt.show()

#计算峰值时间和峰值订单量：按照订单最多的30分钟计算
minute_counts = {minute: 0 for minute in range(6*60, 24*60)}
for order in output:
    minute_counts[order['time']] += 1

max_orders = 0
peak_time = 0
minutes = sorted(minute_counts.keys())
order_list = [minute_counts[m] for m in minutes]
window_sum = sum(order_list[:30])
if window_sum > max_orders:
    max_orders = window_sum
    peak_time = minutes[0]
for i in range(1, len(order_list) - 30 + 1):
    window_sum = window_sum - order_list[i-1] + order_list[i+30-1]
    if window_sum > max_orders:
        max_orders = window_sum
        peak_time = minutes[i]
        
print("峰值时间（30分钟起始）:", peak_time/60, "小时, 订单量:", max_orders)


