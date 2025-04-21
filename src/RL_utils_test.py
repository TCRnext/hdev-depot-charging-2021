import re
import gym
from matplotlib.pylab import f
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import test
import drone_module
import scene_module
from typing import List, Tuple, Dict, Any, Union
import copy
import time
from gym import spaces
from gym.envs.registration import register

drone_num = 100
drone = drone_module.Normal_Drone_Model(620,25,120,130,min_litoff_land_time=1.5,max_battery_energy=64.26)
scene_num = 3
simulation_period = (6*60,24*60,1)
scenelist:List[Union[scene_module.Taxi_generator,scene_module.Goods_delivery_generator]] = []   
for i in range(scene_num):
    scene_piece = scene_module.Taxi_generator(min_per_step=simulation_period[2])
    scenelist.append(scene_piece)
    
register(
    id='DroneChargingEnv-v0',
    entry_point='RL_utils:Drone_charging_env',
)
env_name = 'DroneChargingEnv-v0'
seed = 0
env = gym.make(
    env_name,
    drone = drone,
    drone_num = drone_num, 
    scene_list = scenelist, 
    time_info = simulation_period,
    num_simulation = 1,
    seed = seed,
    )

random.seed(seed)
np.random.seed(seed)


env.reset()
state = {
            "standby_drone_battery_info":[0,0,0,0,0,0,0,0,0,0.99],
            "charging_drone_info":np.zeros(1),
            "task_drone_info":np.zeros(1),
            "current_time":[0],
            "current_order_num":[0],
            "existing_order_num":[0]
        }
while True:
    state = env.reset()
    if env.index == 0:
        break
done = False

action = 40

power_list = []
power_list_thisrun = []
avg_power_list = []
peak_power_list = []

index = 1
end_all_sample = False
while not end_all_sample:
    reward_list_this_run = []
    while not done:
        state, reward, done,end_all_sample,_ = env.step(action)
        action = random.randint(30,40)
        time_now = state["current_time"][0]
        if time_now%60 == 0:
            #print("time_now:",time_now/60.0,"h")
            pass
        reward_list_this_run.append(reward)
    powerlist_thisrun ,avg_power_this_run,max_power_this_run  = env.get_power_info()
    avg_power_list.append(avg_power_this_run)
    peak_power_list.append(max_power_this_run)
    power_list.append(powerlist_thisrun)
    done = False
    SLA = env.cal_SLA()
    state = env.reset()


    print("SLA:",SLA)
    print("index:",env.index)
    index += 1


max_index = peak_power_list.index(max(peak_power_list))

# 画出充电功率的折线图
fig = plt.figure()
ax = fig.add_subplot(111)
# 设置x轴和y轴的范围
ax.set_xlim(simulation_period[0]/60.0,simulation_period[1]/60.0)
ax.set_ylim(0, (max (peak_power_list)*120)//100)
for i in range(len(peak_power_list)):
    time_list = []
    p_list = []
    p_list = power_list[i]
    time_list = [j/60.0 for j in range(simulation_period[0],simulation_period[1],simulation_period[2])]
    if i == max_index:
        ax.plot(time_list ,p_list ,color = '#00BFFF', linewidth = 1)
    else:
        ax.plot(time_list ,p_list ,color = '#87CEFA', linewidth = 0.3,alpha=0.4)
# 设置x轴和y轴的标签
ax.set_xlabel('time(h)')
ax.set_ylabel('charging power(KW)')
# 设置标题
ax.set_title('charging power of taxi drone with ' + 'DQN' + ' strategy')
# 设置备注,保留到整数
ax.text(simulation_period[1]/60.0*0.8, max (peak_power_list)*0.3, 'Peak charging power: ' + str(int(peak_power_list[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')
ax.text(simulation_period[1]/60.0*0.8, max (peak_power_list)*0.15, 'Avg charging power: ' + str(int(avg_power_list[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')

# 显示图形
plt.show()
