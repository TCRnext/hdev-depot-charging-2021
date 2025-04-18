import gym
from matplotlib.pylab import f
import numpy as np
import random
import math

import test
import drone_module
import scene_module
from typing import List, Tuple, Dict, Any, Union
import copy
import time
from gym import spaces

class Drone_charging_env(gym.Env):
    def __init__(
        self,
        drone:drone_module.Normal_Drone_Model,
        drone_num:int,
        scene_list:List[Union[scene_module.Taxi_generator,scene_module.Goods_delivery_generator]] ,
        time_info = (0*60,24*60,1),
        num_simulation = 10,
        use_external_data = False,
        Full_battery_at_start = True,
        enable_partly_charging = True,
        enable_charging_outside = False,
        seed = None,
        ):
        
        self.seed(seed)
        self.drone:drone_module.Normal_Drone_Model = drone        
        self.drone_num = drone_num
        self.scene_list = scene_list
        self.time_info = time_info
        self.num_simulation = num_simulation
        self.use_external_data = use_external_data
        self.Full_battery_at_start = Full_battery_at_start
        self.enable_partly_charging = enable_partly_charging
        self.enable_charging_outside = enable_charging_outside
        
        
        self.delay_reward_base = 0.2
        self.delay_reward_k = -2.0
        
        self.power_reward_A = 1.0
        self.power_reward_base_1 = 0.2
        self.power_reward_base_2 = 0.5
        self.power_reward_k_1 = -0.25
        self.power_reward_k_2 = -0.75
        
        
        
        """
            动作空间：
            最大无人机充电数量(占比)(离散化 10%-100%) 1-10
            转为充电状态无人机的最低电量阈值(离散化 20%-80% 按10%)   2-8 
        """
        
        self.action_space = spaces.Dict({
            "max_drone_percent_to_charge":spaces.Discrete(10,start=1),
            "drone_battery_threshold":spaces.Discrete(7,start=2)
        })
        

        """
            状态空间:
            空闲无人机电量占比:10%分位 
            充电中无人机占比：
            任务中无人机占比：
            
            当前时间:
            当前订单量:
            上一时间的累计订单量:
        """
        self.observation_space = spaces.Dict({
            "standby_drone_battery_info":spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            "charging_drone_info":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "task_drone_info":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "current_time":spaces.Box(low=time_info[0], high=time_info[1], shape=(1,), dtype=np.float32),
            "current_order_num":spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "existing_order_num":spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        })
        
        self.state = {
            "standby_drone_battery_info":[0,0,0,0,0,0,0,0,0,0.99],
            "charging_drone_info":np.zeros(1),
            "task_drone_info":np.zeros(1),
            "current_time":[self.time_info[0]],
            "current_order_num":[0],
            "existing_order_num":[0]
        }
        
        if self.use_external_data is False:
            self.generate_order_list()
            ## 随机划分训练集和测试集
            test_size = int(len(self.order_list) * 0.2)
            train_size = len(self.order_list) - test_size
            train_indices = random.sample(range(len(self.order_list)), train_size)
            test_indices = list(set(range(len(self.order_list))) - set(train_indices))
            self.train_order_list = [copy.deepcopy(self.order_list[i]) for i in train_indices]
            self.train_max_power_list = [copy.deepcopy(self.max_power_list[i]) for i in train_indices]
            self.train_avg_power_list = [copy.deepcopy(self.avg_power_list[i]) for i in train_indices]
            self.test_order_list = [copy.deepcopy(self.order_list[i]) for i in test_indices]
            self.test_max_power_list = [copy.deepcopy(self.max_power_list[i]) for i in test_indices]
            self.test_avg_power_list = [copy.deepcopy(self.avg_power_list[i]) for i in test_indices]

            
        self.max_index = 0
        self.order_index = 0
        self.index = None
        self.max_power_this_run = 0
        self.avg_power_this_run = 0
        self.order_list_this_run = []
        self.power_list_this_run = []
        self.drone_list_this_run:List[drone_module.Normal_Drone_Model] = []
        self.order_fifo_this_run = []
        self.handle_order_this_run = []
        self.SLA_this_run = []
        self.reset()
        
        self.mode = 'default'
        return
    
    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            #时间戳作为随机种子
            seed = int((math.floor(time.time() * 1000) % 1000000))
            random.seed(seed)
            np.random.seed(seed)
        return [seed]
    
    def import_order_data(self,order_list:List[List[Dict[str,Any]]],max_power_list:List[float],avg_power_list:List[float]):
        if self.use_external_data:
            assert len(order_list) == len(max_power_list), "order_list and max_power_list must have the same length"
            assert len(order_list) == len(avg_power_list), "order_list and avg_power_list must have the same length"
            self.order_list = order_list
            self.max_power_list = max_power_list
            self.avg_power_list = avg_power_list
            test_size = int(len(self.order_list) * 0.2)
            train_size = len(self.order_list) - test_size
            train_indices = random.sample(range(len(self.order_list)), train_size)
            test_indices = list(set(range(len(self.order_list))) - set(train_indices))
            self.train_order_list = [copy.deepcopy(self.order_list[i]) for i in train_indices]
            self.train_max_power_list = [copy.deepcopy(self.max_power_list[i]) for i in train_indices]
            self.train_avg_power_list = [copy.deepcopy(self.avg_power_list[i]) for i in train_indices]
            self.test_order_list = [copy.deepcopy(self.order_list[i]) for i in test_indices]
            self.test_max_power_list = [copy.deepcopy(self.max_power_list[i]) for i in test_indices]
            self.test_avg_power_list = [copy.deepcopy(self.avg_power_list[i]) for i in test_indices]
        else:
            print("Fail to import order data, use_external_data is False")
        return 
    
    def export_order_data(self):
        if self.use_external_data:
            print("Fail to export order data, use_external_data is True")
            return None
        return self.order_list ,self.max_power_list, self.avg_power_list
    
    def generate_order_list(self):
        self.order_list = []
        self.max_power_list = []
        self.avg_power_list = []
        for one_scene in self.scene_list:
            for _ in range(self.num_simulation):
                order_list_one_time,max_power,avg_power = self.simulate_once(one_scene)
                self.order_list.append(order_list_one_time)
                self.max_power_list.append(max_power)
                self.avg_power_list.append(avg_power)
                print("finish generating order list for scene: ",len(self.order_list))
        return
    
    def simulate_once(self,one_scene:Union[scene_module.Taxi_generator,scene_module.Goods_delivery_generator]):
        drone_private:drone_module.Normal_Drone_Model = copy.deepcopy(self.drone)
        if self.Full_battery_at_start:
            drone_private.now_battery = 1.0
        
        drone_list:List[drone_module.Normal_Drone_Model] = []
        for _ in range(self.drone_num):
            drone_list.append(copy.deepcopy(drone_private))
        
        order_fifo = []
        power_list = []
        order_list = []
        minimum_num_charging_drone_percent = 0.25
        minimum_num_charging_drone = self.drone_num * minimum_num_charging_drone_percent

        for time in range(self.time_info[0],self.time_info[1],self.time_info[2]):
            piece = one_scene.generate(time)
            for order in piece:
                order_fifo.append(order)
            order_is_finished = []
            
            for order in order_fifo:
                for drone in drone_list:
                    if drone.drone_status == drone.status_charge and self.enable_partly_charging == False:
                        continue
                    if drone.status_to_flight(order['distance'],is_single_path = self.enable_charging_outside,is_P2P_path = self.enable_charging_outside) :
                        order_is_finished.append(order)
                        order['tx_time'] = time
                        order['rx_time'] = time + drone.flight_time_left
                        order_list.append(copy.deepcopy(order))
                        break
            
            for order in order_is_finished:
                order_fifo.remove(order)

            current_charging_power = 0
            current_is_charging_drone = 0
            drone:drone_module.Normal_Drone_Model
            for drone in drone_list:
                drone.update_status(self.time_info[2])

            num_charging_drone = 0
            if True:
                for drone in drone_list:
                    if drone.drone_status == drone.status_charge:
                        num_charging_drone += 1
                if num_charging_drone < minimum_num_charging_drone:
                    for drone in drone_list:
                        if drone.drone_status == drone.status_standby and drone.now_battery < 0.99:
                            drone.status_to_charge()
                            num_charging_drone += 1
                        if num_charging_drone >= minimum_num_charging_drone:
                            break
            else:
                standby_drone_battery = 0.9
                least_standby_drone_percent = 0.25
                num_drone = len(drone_list)

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


            for drone in drone_list:
                if drone.is_parking_outside == False:
                    if drone.drone_status == drone.status_charge:
                        current_is_charging_drone += 1
                        current_charging_power += drone.current_charge_power
            power_list.append(current_charging_power)
            

        avg_power = np.average(power_list)
        max_power = np.max(power_list)
        return order_list,max_power,avg_power

    def setmode(self,mode:str):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        elif mode == 'default':
            self.mode = 'default'
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        self.index = None
        self.reset()
        return

    def step(self, action):
        max_drone_num_to_charge = action["max_drone_percent_to_charge"]/10.0*self.drone_num
        drone_battery_threshold = action["drone_battery_threshold"]/10.0
        
        self.load_order_to_fifo(self.state["current_time"])
        delay_reward = self.handle_order(self.state["current_time"])
        power_reward = self.handle_drone(max_drone_num_to_charge,drone_battery_threshold)
        reward = delay_reward + power_reward
        
        self.state["current_time"][0] += self.time_info[2]
        if self.state["current_time"][0] > self.time_info[1]:
            done = True
            if self.index == self.max_index:
                truncated = True
            else:   
                truncated = False
    
        return self.state,reward,done,truncated,[]

    def load_order_to_fifo(self,current_time):
        current_order_num = 0
        while current_time <= self.order_list_this_run[self.order_index]['time']:
            current_order_num += 1
            self.order_fifo_this_run.append(self.order_list_this_run[self.order_index])
            self.order_index += 1
            if self.order_index >= len(self.order_list_this_run):
                break
        self.state["current_order_num"] = current_order_num
        self.state["existing_order_num"] = len(self.order_fifo_this_run)
        return

    def handle_order(self,current_time):
        order_is_finished = []
        reward_order = 0
        for order in self.order_fifo_this_run:
            for drone in self.drone_list_this_run:
                if drone.drone_status == drone.status_charge and self.enable_partly_charging == False:
                    continue
                if drone.status_to_flight(order['distance'],is_single_path = self.enable_charging_outside,is_P2P_path = self.enable_charging_outside) == False:
                    order_is_finished.append(order)
                    reward_order += (self.delay_reward_base + self.delay_reward_k *(current_time - order['time']))
                    order['tx_time'] = current_time
                    order['rx_time'] = current_time + drone.flight_time_left
                    self.handle_order_this_run.append(copy.deepcopy(order))
                    break
        
        for order in order_is_finished:
            self.order_fifo_this_run.remove(order)        
        return reward_order

    def handle_drone(self,max_drone_num_to_charge,drone_battery_threshold):
        list_standby_drone = []
        current_charging_power = 0
        current_is_charging_drone = 0
        for drone in self.drone_list_this_run:
            drone.update_status(self.time_info[2])
            if drone.is_parking_outside == False:
                if drone.drone_status == drone.status_charge:
                    current_is_charging_drone += 1
                    current_charging_power += drone.current_charge_power
                elif drone.drone_status == drone.status_standby:
                    list_standby_drone.append(drone)
        current_is_standby_drone = len(list_standby_drone)
        current_is_tasking_drone = self.drone_num - current_is_standby_drone - current_is_charging_drone
        self.power_list_this_run.append((time//60,time%60,current_charging_power,current_is_charging_drone))        
        
        #根据电量对list_standby_drone进行排序
        list_standby_drone.sort(key=lambda x: x.now_battery, reverse=True)
        standby_drone_battery_list = np.zeros(10)
        temp = 1
        temp2 = 1
        
        for drone in list_standby_drone:
            if drone > temp * 0.1:
                temp +=1
            standby_drone_battery_list[temp] += 1.0/self.drone_num
            
        for drone in list_standby_drone:
            if temp2 <= max_drone_num_to_charge and drone.now_battery < drone_battery_threshold:
                    drone.status_to_charge()
                    temp2 += 1
            else:
                break
        self.state["standby_drone_battery_info"] = standby_drone_battery_list
        self.state["charging_drone_info"] = np.array([current_is_charging_drone/self.drone_num])
        self.state["task_drone_info"] = np.array([current_is_tasking_drone/self.drone_num])
        
        power_reward = 0
        if current_charging_power < self.avg_power_this_run-0.1:
            power_reward = self.power_reward_base_1 + self.power_reward_k_1 * (self.avg_power_this_run - current_charging_power)
        elif current_charging_power > self.max_power_this_run+0.1:
            power_reward = self.power_reward_base_2 + self.power_reward_k_2 * (current_charging_power - self.max_power_this_run)
        else:
            power_reward = self.power_reward_A
        
        return power_reward

    def reset(self):
        if self.index is None:
            self.index = 0
            if self.mode == 'train':
                ##生成索引列表，按列表打乱顺序
                self.max_index = len(self.train_order_list) - 1
                index_list = list(range(len(self.train_order_list)))
                random.shuffle(index_list)
                self.shuffle_list = [self.train_order_list[i] for i in index_list]
                self.shuffle_max_power_list = [self.train_max_power_list[i] for i in index_list]
                self.shuffle_avg_power_list = [self.train_avg_power_list[i] for i in index_list]
            elif self.mode == 'test':
                self.max_index = len(self.test_order_list) - 1
                self.shuffle_list = self.test_order_list
                self.shuffle_max_power_list = self.test_max_power_list   
                self.shuffle_avg_power_list = self.test_avg_power_list
            else:
                self.max_index = len(self.order_list) - 1
                index_list = list(range(len(self.order_list)))
                random.shuffle(index_list)
                self.shuffle_list = [self.order_list[i] for i in index_list]
                self.shuffle_max_power_list = [self.max_power_list[i] for i in index_list]
                self.shuffle_avg_power_list = [self.avg_power_list[i] for i in index_list]                

        else:
            self.index += 1
            if self.index > self.max_index:
                self.index = None
                return self.reset()
        self.order_list_this_run = self.shuffle_list[self.index]
        self.max_power_this_run = self.shuffle_max_power_list[self.index]
        self.avg_power_this_run = self.shuffle_avg_power_list[self.index]
        self.drone_list_this_run = []
        self.order_fifo_this_run = []
        self.order_index = 0
        
        
        drone_private:drone_module.Normal_Drone_Model = self.drone.deepcopy()
        if self.Full_battery_at_start:
            drone_private.now_battery = 1.0
            
        for _ in range(self.drone_num):
            self.drone_list_this_run.append(copy.deepcopy(drone_private))
        
        self.state = {
            "standby_drone_battery_info":[0,0,0,0,0,0,0,0,0,0.99],
            "charging_drone_info":np.zeros(1),
            "task_drone_info":np.zeros(1),
            "current_time":self.time_info[0],
            "current_order_num":0,
            "existing_order_num":0
        }
        
        return self.state 

    def cal_SLA(self):
        SLA = 0.0
        total_time = 0
        for order in self.handle_order_this_run:
            total_time += order['rx_time'] - order['time']
            delay_time += order['tx_time'] - order['time']
        SLA = (total_time - delay_time) / total_time
        return SLA
     
    def render(self):
        # No need to render 
        pass