from calendar import c
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
    simulation_period = (0*60,24*60,1),
    num_simulation = 1,
    is_same_scene = True,
    charge_strat = 'immediate', #充电策略 
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
    drone_list = []
    if non_default_scene_module is not None:
        scene= non_default_scene_module(min_per_step=simulation_period[2])
    else:
        scene = scene_module.Taxi_generator(min_per_step=simulation_period[2])
    if non_default_drone_module is not None:
        drone = non_default_drone_module
    else:
        drone = drone_module.Normal_Drone_Model(620,25,120,130,min_litoff_land_time=1.5,max_battery_energy=64.26)
    order_list = []
    chargingpower_list = []
    if is_same_scene:
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece = Taxi_simulate_core(
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
            print("第",simulation,"次模拟")
    else:
        if non_default_scene_module is not None:
            scene= non_default_scene_module(min_per_step=simulation_period[2])
        else:
            scene = scene_module.Taxi_generator(min_per_step=simulation_period[2])
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece = Taxi_simulate_core(
                num_drone,
                simulation_period,
                scene,
                drone,
                drone_full_power_at_begin = True,
                enable_partly_charging = True,
                charge_strat = charge_strat,
                minimum_num_charging_drone = 25,
                power_limit = 100,              #KW
                standby_drone_battery = 0.9,          #KW
                least_standby_drone_percent = 0.2,    #KW
            )
            order_list.append(order_list_piece)
            chargingpower_list.append(chargingpower_list_piece)
            print("第",simulation,"次模拟")
    return order_list, chargingpower_list  
        
def Taxi_simulate_core(
                                num_drone,
                                simulation_period,
                                scene:scene_module.Taxi_generator ,
                                drone_example:drone_module.Normal_Drone_Model ,
                                drone_full_power_at_begin = True,
                                enable_partly_charging = True,
                                charge_strat = 'immediate',
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
        
        order_is_finished = []
        for order in order_fifo:
            for drone in drone_list:
                if drone.drone_status == drone.status_charge and enable_partly_charging == False:
                    continue
                if drone.status_to_flight(order['distance']):
                    order_is_finished.append(order)
                    order['tx_time'] = time
                    order['rx_time'] = time + drone.flight_time_left
                    order_list.append(copy.deepcopy(order))
                    break
        #删除已经完成的订单
        for order in order_is_finished:
            order_fifo.remove(order)
        if len(order_fifo) > 0:
            print("delayed at",time//60,":",time%60)
        

        drone:drone_module.Normal_Drone_Model
        for drone in drone_list:
            drone.update_status(simulation_period[2])

     
        if charge_strat == 'immediate':
            for drone in drone_list:
                if drone.drone_status == drone.status_standby and drone.now_battery < 0.99:
                    drone.status_to_charge()
                    
        elif charge_strat == 'Limited_power':
            for drone in drone_list:
                if current_charging_power < power_limit and drone.drone_status == drone.status_standby and drone.now_battery < 0.99:
                    drone.status_to_charge()
                    current_charging_power += drone.current_charge_power
                if current_charging_power >= power_limit:
                    break
        
        elif charge_strat == 'Least_standby':
            num_standby_drone = 0
            num_charging_drone = 0
            for drone in drone_list:
                if drone.drone_status == drone.status_standby and drone.now_battery < standby_drone_battery:
                    num_standby_drone += 1
                if drone.drone_status == drone.status_charge:
                    num_charging_drone += 1
            if num_standby_drone < num_drone * least_standby_drone_percent:
                num_need_charge = max(minimum_num_charging_drone , int(num_drone * least_standby_drone_percent) - num_standby_drone) - num_charging_drone
            for drone in drone_list:
                if num_need_charge <= 0:
                    break
                if drone.drone_status == drone.status_standby and drone.now_battery < 0.99:
                    drone.status_to_charge()
                    num_need_charge -= 1

        elif charge_strat == 'delayed':
            num_charging_drone = 0
            for drone in drone_list:
                if drone.drone_status == drone.status_charge:
                    num_charging_drone += 1
            if num_charging_drone < minimum_num_charging_drone:
                for drone in drone_list:
                    if drone.drone_status == drone.status_standby and drone.now_battery < 0.2:
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
        
                
    #print("订单共有",len(order_list),"个") 
    #print(chargingpower_list)                        
    return order_list, chargingpower_list 


if __name__ == '__main__':
    simulation_period = (0*60,24*60,1)
    order_list ,chargingpower_list = scene_Taxi_analysis(simulation_period = simulation_period)
    peak_chargingpower_list = []
    avg_chargingpower_list = []
    
    for i in range(len(order_list)):
        order_list_piece = order_list[i]
        chargingpower_list_piece = chargingpower_list[i]
        print("第",i,"次模拟")
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
        peak_chargingpower_list.append(peak_chargingpower)
        avg_chargingpower_list.append(avg_chargingpower)
        print("平均充电功率",avg_chargingpower)
        print("峰值充电功率",peak_chargingpower)
            
    # 从peak_chargingpower_list中获取最大值,中位值，最小值的index
    max_index = peak_chargingpower_list.index(max(peak_chargingpower_list))
    min_index = peak_chargingpower_list.index(min(peak_chargingpower_list))
    mid_index = peak_chargingpower_list.index(sorted(peak_chargingpower_list)[len(peak_chargingpower_list)//2])
    
    # 画出充电功率的折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 设置x轴和y轴的范围
    ax.set_xlim(simulation_period[0]/60.0,simulation_period[1]/60.0)
    ax.set_ylim(0, (max (peak_chargingpower_list)*120)//100)
    for i in range(len(peak_chargingpower_list)):
        time_list = []
        p_list = []
        for chargingpower_1min in chargingpower_list[i]:
            time_list.append((chargingpower_1min[0]*60.0 + chargingpower_1min[1])/60.0)
            p_list.append(chargingpower_1min[2])
        if i == max_index:
            ax.plot(time_list ,p_list ,color = '#00BFFF', linewidth = 2)
        else:
            ax.plot(time_list ,p_list ,color = '#87CEFA', linewidth = 0.2,alpha=0.4)
    # 设置x轴和y轴的标签
    ax.set_xlabel('time(h)')
    ax.set_ylabel('charging power(KW)')
    # 设置标题
    ax.set_title('charging power')
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
    
