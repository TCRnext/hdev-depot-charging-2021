from calendar import c
from doctest import debug
import random
from matplotlib.pylab import rand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from pyparsing import col, line
import drone_module
import scene_module
import copy

def scene_Taxi_analysis(
    non_default_scene_module = None,
    non_default_drone_module = None,
    num_drone = 100,
    simulation_period = (6*60,24*60,1),
    num_simulation = 1,
    is_same_scene = True,
    charge_strat = 'immediate', #充电策略 
    minimum_num_charging_drone = 25,
    power_limit = 950,              #KW
    standby_drone_battery = 0.9,          #KW
    least_standby_drone_percent = 0.40,    #KW
    per_step_base_generate = 1
    
):
    assert charge_strat in ['immediate', 
                        'Limited_power', 
                        'Least_standby',
                        'delayed'
                        ], "charge_strat not recognized!"
    drone_list = []
    if non_default_scene_module is not None:
        scene= non_default_scene_module(min_per_step=simulation_period[2])
    else:
        scene = scene_module.Taxi_generator(min_per_step=simulation_period[2],per_step_base_generate = per_step_base_generate)
    if non_default_drone_module is not None:
        drone = non_default_drone_module
    else:
        drone = drone_module.Normal_Drone_Model(620,25,120,130,min_litoff_land_time=1.5,max_battery_energy=64.26)
    order_list = []
    chargingpower_list = []
    delta_battery_energy_list = []
    if is_same_scene:
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece ,delta_battery_energy= Taxi_simulate_core(
                num_drone,
                simulation_period,
                scene,
                drone,
                drone_full_power_at_begin = True,
                enable_partly_charging = True,
                charge_strat = charge_strat,
                minimum_num_charging_drone = minimum_num_charging_drone,
                power_limit = power_limit,              #KW
                standby_drone_battery = standby_drone_battery,          #KW
                least_standby_drone_percent = least_standby_drone_percent,    #KW
            )
            order_list.append(order_list_piece)
            chargingpower_list.append(chargingpower_list_piece)
            delta_battery_energy_list.append(delta_battery_energy)
            print("第",simulation,"次模拟")
    else:
        if non_default_scene_module is not None:
            scene= non_default_scene_module(min_per_step=simulation_period[2])
        else:
            scene = scene_module.Taxi_generator(min_per_step=simulation_period[2])
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece ,delta_battery_energy= Taxi_simulate_core(
                num_drone,
                simulation_period,
                scene,
                drone,
                drone_full_power_at_begin = True,
                enable_partly_charging = False,
                charge_strat = charge_strat,
                minimum_num_charging_drone = 25,
                power_limit = 100,              #KW
                standby_drone_battery = 0.9,          #KW
                least_standby_drone_percent = least_standby_drone_percent,    #KW
            )
            order_list.append(order_list_piece)
            chargingpower_list.append(chargingpower_list_piece)
            delta_battery_energy_list.append(delta_battery_energy)
            print("第",simulation,"次模拟")
    return order_list, chargingpower_list ,delta_battery_energy_list
        
def Taxi_simulate_core(
                                num_drone,
                                simulation_period,
                                scene:scene_module.Taxi_generator ,
                                drone_example:drone_module.Normal_Drone_Model ,
                                drone_full_power_at_begin = True,
                                enable_partly_charging = True,
                                charge_strat = 'Least_standby',
                                minimum_num_charging_drone = 25,
                                power_limit = 100,              #KW
                                standby_drone_battery = 0.9,          #KW
                                least_standby_drone_percent = 0.2,    #KW                         
                                ):
    assert charge_strat in ['immediate', 
                        'Limited_power', 
                        'Least_standby',
                        'delayed'
                        ], "charge_strat not recognized!"
    
    order_list = []
    order_fifo = []
    drone_list = []
    chargingpower_list = []


    
    if drone_full_power_at_begin:
        drone_example.now_battery = 1.0
    
    for i in range(num_drone):
        drone_list.append(copy.deepcopy(drone_example))

    
    for time in range(simulation_period[0],simulation_period[1],simulation_period[2]):
        
        piece = scene.generate(time)
        for j in piece:
            order_fifo.append(j)
        drone:drone_module.Normal_Drone_Model
        for drone in drone_list:
            drone.update_status(simulation_period[2])
        drone_non_tasking_list = []
        for drone in drone_list:
            if drone.drone_status == drone.status_standby or drone.drone_status == drone.status_charge:
                drone_non_tasking_list.append(drone)
        drone_non_tasking_list.sort(key=lambda x: x.now_battery+x.drone_status, reverse=False)
        
        order_is_finished = []
        
        for order in order_fifo:
            for drone in drone_non_tasking_list:
                if drone.drone_status == drone.status_charge and enable_partly_charging == False:
                    continue
                if drone.status_to_flight(order['distance']):
                    order_is_finished.append(order)
                    order['tx_time'] = time
                    order['rx_time'] = time + drone.flight_time_left
                    order_list.append(copy.deepcopy(order))
                    drone_non_tasking_list.remove(drone)
                    break
        #删除已经完成的订单
        for order in order_is_finished:
            order_fifo.remove(order)
        if len(order_fifo) > 0:
            print("delayed at",time//60,":",time%60)
   
        if charge_strat == 'immediate':
            for drone in drone_list:
                if drone.drone_status == drone.status_standby and drone.now_battery < 0.9:
                    drone.status_to_charge()
                    
        elif charge_strat == 'Limited_power':
            minimum_num_charging_drone = power_limit / (drone_example.max_battery_energy/(drone_example.max_charge_time/60)) 
            num_charging_drone = 0
            for drone in drone_list:
                if drone.drone_status == drone.status_charge:
                    num_charging_drone += 1
            if num_charging_drone < minimum_num_charging_drone:
                for drone in drone_list:
                    if drone.drone_status == drone.status_standby and drone.now_battery < 0.9:
                        drone.status_to_charge()
                        num_charging_drone += 1
                    if num_charging_drone >= minimum_num_charging_drone:
                        break
        
        elif charge_strat == 'Least_standby':
            num_standby_drone = 0
            num_charging_drone = 0
            for drone in drone_list:
                if drone.drone_status == drone.status_standby and drone.now_battery > standby_drone_battery:
                    num_standby_drone += 1
                if drone.drone_status == drone.status_charge:
                    num_charging_drone += 1
            if num_standby_drone < num_drone * least_standby_drone_percent:
                num_need_charge =  num_drone * least_standby_drone_percent - num_charging_drone
            else:
                num_need_charge =  num_drone * least_standby_drone_percent*3/4 - num_charging_drone
            for drone in drone_list:
                if num_need_charge <= 0:
                    break
                if drone.drone_status == drone.status_standby and drone.now_battery < 0.9:
                    drone.status_to_charge()
                    num_need_charge -= 1

        elif charge_strat == 'delayed':
            num_charging_drone = 0
            for drone in drone_list:
                if drone.drone_status == drone.status_charge:
                    num_charging_drone += 1
                elif drone.drone_status == drone.status_standby and drone.now_battery < 0.25:
                    drone.status_to_charge()
                    num_charging_drone += 1
                if  num_charging_drone >= minimum_num_charging_drone *1.5:
                    break
            if num_charging_drone < minimum_num_charging_drone:
                for drone in drone_list:
                    if drone.drone_status == drone.status_standby and drone.now_battery < 0.9:
                        drone.status_to_charge()
                        num_charging_drone += 1
                    if num_charging_drone >= minimum_num_charging_drone:
                        break
                                    
        if(time%60 == 0):
            print("time:",time//60)

        current_charging_power = 0
        current_is_charging_drone = 0
        for drone in drone_list:
            if drone.is_parking_outside == False:
                if drone.drone_status == drone.status_charge:
                    current_is_charging_drone += 1
                    current_charging_power += drone.current_charge_power
                
        chargingpower_list.append((time//60,time%60,current_charging_power,current_is_charging_drone))
    
    battery_total = 0   
    for drone in drone_list:
        battery_total += (1-drone.now_battery)
    delta_battery_energy = battery_total * drone.max_battery_energy 
                        
    return order_list, chargingpower_list, delta_battery_energy


if __name__ == '__main__':
    charge_strat = 'Least_standby'
    least_standby_drone_percent = 0.50
    minimum_num_charging_drone = 30
    power_limit = 950
    #charge_strat = 'immediate' or 'Limited_power' or 'Least_standby' or 'delayed' 
    random.seed(0)
    np.random.seed(0)
    simulation_period = (6*60,24*60,1)
    order_list ,chargingpower_list ,delta_battery_energy_list  = scene_Taxi_analysis(
        num_drone = 100,
        simulation_period = (6*60,24*60,1),
        num_simulation = 20,
        is_same_scene = True,
        charge_strat = charge_strat, #充电策略 
        minimum_num_charging_drone = minimum_num_charging_drone,
        power_limit = power_limit,              #KW
        standby_drone_battery = 0.9,          #KW
        least_standby_drone_percent = least_standby_drone_percent,    #KW
        per_step_base_generate = 1
        )
    peak_chargingpower_list = []
    avg_chargingpower_list = []
    peak_avg_ratio_list = []
    
    for i in range(len(order_list)):
        order_list_piece = order_list[i]
        chargingpower_list_piece = chargingpower_list[i]
        delta_battery_energy = delta_battery_energy_list[i]
        print("第",i+1,"次模拟")
        print("订单共有",len(order_list_piece),"个")
        total_time = 0
        total_delay = 0
        for order in order_list_piece:
            total_time += order['rx_time'] - order['time']
            total_delay +=  order['tx_time'] - order['time']
        sla = 1- total_delay / total_time
        print("SLA:",sla)
        print("平均每个订单的时间",total_time/len(order_list_piece))
        peak_chargingpower = 0
        avg_chargingpower = 0
        for chargingpower in chargingpower_list_piece:
            peak_chargingpower = max(peak_chargingpower,chargingpower[2])
            avg_chargingpower += chargingpower[2]
        avg_chargingpower /= len(chargingpower_list_piece)
        avg_chargingpower = (avg_chargingpower *(simulation_period[1] - simulation_period[0]) / 60.0 + delta_battery_energy)/24.0
        peak_chargingpower_list.append(peak_chargingpower)
        avg_chargingpower_list.append(avg_chargingpower)
        peak_avg_ratio_list.append(peak_chargingpower/avg_chargingpower)
        print("平均充电功率",avg_chargingpower)
        print("峰值充电功率",peak_chargingpower)
        print("充电功率占比",peak_chargingpower/avg_chargingpower)
            
    # 从peak_chargingpower_list中获取最大值,中位值，最小值的index
    max_index = peak_avg_ratio_list.index(max(peak_avg_ratio_list))
    min_index = peak_avg_ratio_list.index(min(peak_avg_ratio_list))
    mid_index = peak_avg_ratio_list.index(sorted(peak_avg_ratio_list)[len(peak_avg_ratio_list)//2])
    
    # 逐一计算每10分钟的平均订单量    
    num_period = (simulation_period[1] - simulation_period[0]) // 10 +1
    order_num_list = np.zeros(num_period)
    for i in range(len(order_list)):
        order_list_piece = order_list[i]
        for order in order_list_piece:
            period_index = (order['tx_time'] - simulation_period[0]) // 10
            if period_index < num_period:
                order_num_list[period_index] += 1 / len (order_list)   
    time_list_10min = []
    for i in range(num_period):
        time_list_10min.append((simulation_period[0] + i*10)/60.0)
    
    # 画出充电功率的折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置x轴和y轴的范围
    ax.set_xlim(simulation_period[0]/60.0,simulation_period[1]/60.0)
    ax.set_ylim(0, (max (peak_chargingpower_list)*120)//100)
    ax_2 = ax.twinx()
    ax_2.set_ylim(0, (max (order_num_list)*120)//100)
    for i in range(len(peak_chargingpower_list)):
        time_list = []
        p_list = []
        for chargingpower_1min in chargingpower_list[i]:
            time_list.append((chargingpower_1min[0]*60.0 + chargingpower_1min[1])/60.0)
            p_list.append(chargingpower_1min[2])
        if i == max_index:
            ax.plot(time_list ,p_list ,color = '#00BFFF', linewidth = 1)
        else:
            ax.plot(time_list ,p_list ,color = '#87CEFA', linewidth = 0.3,alpha=0.4)
        ax_2.plot(time_list_10min,order_num_list, color = '#FF0000', linewidth = 0.5)
    # 设置x轴和y轴的标签
    ax.set_xlabel('time(h)')
    ax.set_ylabel('charging power(KW)')
    ax_2.set_ylabel('avg order num 10min')
    # 设置标题
    ax.set_title('charging power of taxi drone with ' + charge_strat + ' strategy')
    # 设置备注,保留到整数
    ax.text(simulation_period[1]/60.0*0.8, max (peak_chargingpower_list)*0.3, 'Peak charging power: ' + str(int(peak_chargingpower_list[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')
    ax.text(simulation_period[1]/60.0*0.8, max (peak_chargingpower_list)*0.15, 'Avg charging power: ' + str(int(avg_chargingpower_list[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')
    # 倍率，保留三位小数
    ax.text(simulation_period[1]/60.0*0.8, max (peak_chargingpower_list)*0.45, 'Peak/Avg ratio: ' + str(round(peak_avg_ratio_list[max_index],3)), ha='center', va='center', fontsize=10, color='#000000')
    # 显示图形
    plt.show()
    # 保存图形,生成时间戳
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    #plt.savefig('charging_power_' + timestamp + '.png')

    #保存数据到csv文件,生成时间戳
    df = pd.DataFrame(chargingpower_list)
    #df.to_csv('chargingpower_list_' + timestamp + '.csv', index=True)
    df = pd.DataFrame(order_list)
    #df.to_csv('order_list_' + timestamp + '.csv', index=True)
    
