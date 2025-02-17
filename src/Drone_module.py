from ast import Pow
from operator import is_
import os
import math
import random
import datetime
import re
import numpy as np
import pandas as pd


#无人机能量消耗——补充模型
class Normal_Drone_Model:
    status_standby = 0
    status_flight  = 1
    status_charge  = 2
    is_parking_outside = False
    parking_to_center_distance = 0
    def __init__(
                    self, 
                    weight,
                    max_flight_time,    #最大飞行时间
                    max_charge_time,    #充电时间
                    max_speed,          #最大速度
                    
                    max_battery = None, #最大电池容量
                    max_distance = None, #最大飞行距离
                    
                    battery_weight = None, #电池重量
                    battery_energy_density = None, #电池能量密度 
                    
                    now_battery = 1.0, #当前电池容量百分比
                    Ct = 5.0,           #螺旋桨效率5KG/KW
                    TWR_max = 2.0,      #最大推力比
                    min_litoff_land_time = 1, #最小起降时间(分钟)      
                    min_battery = 0,  #最小安全电量     
                    init_status = 0, #初始状态 0-待命 1-起飞 2-充电        
                    parking_outside_charging_powerlimit = None #外部停机坪充电功率限制                     
                ):
        self.weight = weight
        self.max_flight_time = max_flight_time
        self.max_charge_time = max_charge_time
        self.max_speed = max_speed
        self.Ct = Ct
        self.TWR_max = TWR_max
        self.min_litoff_land_time = min_litoff_land_time
        
        if max_battery is None:
            self.max_battery = self.weight * Ct*(TWR_max*(self.max_flight_time - self.min_litoff_land_time*2) + self.min_litoff_land_time*2)/60
            if battery_weight is not None and battery_energy_density is not None:
                self.max_battery = battery_weight * battery_energy_density
        else:
            self.max_battery = max_battery
            
        if max_distance is None:
            self.max_distance = self.max_speed * (self.max_flight_time - self.min_litoff_land_time*2) / 60
        else:
            self.max_distance = max_distance
            
        self.now_battery = now_battery 
        self.battery_weight = battery_weight
        self.battery_energy_density = battery_energy_density
        self.min_battery = min_battery
        self.drone_status = init_status
        self.flight_time_left = 0
        self.charge_time_left = 0
        self.Power_limit = None
        if parking_outside_charging_powerlimit is not None:
            self.parking_outside_charging_powerlimit = parking_outside_charging_powerlimit
        else:
            self.parking_outside_charging_powerlimit = None

    
    def status_to_flight(self, distance,is_single_path = False,is_P2P_path = False):
        if is_P2P_path:
            if is_single_path:
                battery = self.now_battery*(1-(2*self.min_litoff_land_time+distance/self.max_flight_time)/self.max_flight_time)
                if battery < self.min_battery:
                    return False
                else:
                    self.now_battery = battery
                    self.flight_time_left = 2*self.min_litoff_land_time+distance/self.max_speed
                    self.drone_status = self.status_flight
                    if self.is_parking_outside == False:
                        self.is_parking_outside = True
                        self.parking_to_center_distance = distance
                    else :
                        self.parking_to_center_distance = 0
                    return True            
            else:
                battery = self.now_battery*(1-(4*self.min_litoff_land_time+2*distance/self.max_flight_time)/self.max_flight_time)
                if battery < self.min_battery:
                    return False
                else:
                    self.now_battery = battery
                    self.flight_time_left = 4*self.min_litoff_land_time+2*distance/self.max_speed
                    self.drone_status = self.status_flight
                    return True             

        else:
            battery = self.now_battery*(1-(6*self.min_litoff_land_time+distance/self.max_flight_time)/self.max_flight_time)
            if battery < self.min_battery:
                return False
            else:
                self.now_battery = battery
                self.flight_time_left = 6*self.min_litoff_land_time+distance/self.max_speed
                self.drone_status = self.status_flight
                return True
    

    
    def status_to_charge(self, Power_limit = None):
        self.drone_status = self.status_charge
        if Power_limit is None:
            self.Power_limit = None
        else:
            self.Power_limit = Power_limit
        return True

    
    def update_status(
                    self,
                    period = 1 #时间间隔,单位为分钟
                    ):
        if self.drone_status == self.status_flight:
            self.flight_time_left -= period
            if self.flight_time_left <= 0:
                self.drone_status = self.status_standby
                self.flight_time_left = 0
                if self.is_parking_outside:
                    self.status_to_charge(power_limit=self.parking_outside_charging_powerlimit)
                    return self.charge_policy(period=period)
                return self.drone_status ,None
            else:
                return self.drone_status ,None
        elif self.drone_status == self.status_charge:
            return self.charge_policy(period=period)
        else:
            return self.drone_status ,None


    def charge_policy(self,period):
        if self.Power_limit is None:
            self.now_battery += period/self.max_charge_time
            if self.now_battery >= 1:
                self.now_battery = 1
                self.drone_status = self.status_standby
                if self.is_parking_outside:
                    self.status_to_flight(self.parking_to_center_distance,is_single_path=True,is_P2P_path=True)
            return self.drone_status ,self.max_battery/self.max_charge_time
        
        else:
            self.now_battery += self.Power_limit*period/self.max_battery/60
            if self.now_battery >= 1:
                self.now_battery = 1
                self.drone_status = self.status_standby
            return self.drone_status ,self.Power_limit

        
    def output_info(self):
        print('起飞重量:',self.weight)
        print('最大飞行时间:',self.max_flight_time,'分钟')
        print('最大充电时间:',self.max_charge_time,'分钟')
        print('最大速度:',self.max_speed,'km/h')
        print('最大电池容量:',self.max_battery,'KW*h')
        print('最大飞行距离:',self.max_distance,'km')
        print('电池重量:',self.battery_weight,'kg')
        print('电池能量密度:',self.battery_energy_density,'KW/kg')
        print('当前电池容量:',self.now_battery*100,'%')
        status = '待命' if self.drone_status == 0 else '起飞' if self.drone_status == 1 else '充电'
        print('当前状态:',status)
        if self.drone_status == self.status_flight:
            print('剩余飞行时间:',self.flight_time_left,'分钟')
        elif self.drone_status == self.status_charge:
            print('当前充电速率:',self.Power_limit,'KW')
            
        
        
        
        