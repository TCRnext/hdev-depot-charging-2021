import dis
import select
from turtle import distance
import Drone_module
from ast import Num, Pow, type_param
import os
import math
import random
import datetime
import re
import numpy as np
import pandas as pd

class General_delivery_generator:
    def __init__(
                self ,
                range_radius = 3 ,                                  ## 配送场景中的配送半径，eg:5km
                cluster_radius = 0.2 ,                              ## 配送场景中的集群半径，eg:0.2km
                num_clusters_tx = 5 ,                               ## 配送场景中的集群发送点数量，eg:餐馆，食品配送中心等
                num_clusters_rx = 7 ,                               ## 配送场景中的集群接收点数量，eg:写字楼，住宅区等
                tx_cluster_weight = [2,2,2,2,2] ,                   ## 配送场景中的集群发送点权重
                rx_cluster_weight = [2,2,2,2,2,2,2] ,               ## 配送场景中的集群接收点权重
                non_cluster_weight = 1 ,                            ## 配送场景中的非集群权重
                is_busy_time = True ,                               ## 是否设置繁忙时间段
                busy_time = [8,12,18,20] ,                          ## 配送场景中的繁忙时间段(小时)
                busy_time_length = 2 ,                              ## 配送场景中的繁忙时间段长度(小时)
                busy_time_weight = [0.5,1,1,0.5] ,                  ## 配送场景中的繁忙时间段权重
                busy_time_cluster_extra_weight = [0,0.2,0.2,0] ,    ## 配送场景中的繁忙时间段集群额外权重
                is_center_cluster_tx = True ,                       ## 是否将一个集群发送点设置为中心点
                per_min_base_generate = 5                           
                ):
        
        assert (len(rx_cluster_weight) == num_clusters_rx) or (len(rx_cluster_weight) == 1) , "rx_cluster_weight长度应该等于num_clusters_rx或者等于1"
        assert len(tx_cluster_weight) == num_clusters_tx or (len(tx_cluster_weight) == 1) , "tx_cluster_weight长度应该等于num_clusters_tx或者等于1"
        assert len(busy_time_weight) == len(busy_time) , "busy_time_weight长度应该等于busy_time"
        assert len(busy_time_cluster_extra_weight) == len(busy_time) , "busy_time_cluster_extra_weight长度应该等于busy_time"
                
        self.range_radius = range_radius
        self.cluster_radius = cluster_radius
        self.num_clusters_tx = num_clusters_tx
        self.num_clusters_rx = num_clusters_rx
        self.is_center_cluster_tx = is_center_cluster_tx
        self.cluster_weight = rx_cluster_weight
        self.non_cluster_weight = non_cluster_weight
        self.per_min_base_generate = per_min_base_generate
        ## 生成配送场景 在以半径为range_radius的圆内生成num_clusters_tx个集群发送点和num_clusters_rx个集群接收点
        if is_center_cluster_tx:
            self.cluster_tx = [[0,0]]
            for i in range(num_clusters_tx-1):
                tx =[range_radius,range_radius]
                while tx[0]*tx[0] + tx[1]*tx[1]  > range_radius*range_radius:
                    tx = [random.uniform(-range_radius,range_radius),random.uniform(-range_radius,range_radius)]
                self.cluster_tx.append(tx)
        else:
            self.cluster_tx = []
            for i in range(num_clusters_tx):
                tx =[range_radius,range_radius]
                while tx[0]*tx[0] + tx[1]*tx[1] > range_radius*range_radius:
                    tx = [random.uniform(-range_radius,range_radius),random.uniform(-range_radius,range_radius)]
                self.cluster_tx.append(tx)
                
       
        self.cluster_rx = []
        for i in range(num_clusters_rx):
            rx =[range_radius,range_radius]
            while rx[0]*rx[0] + rx[1]*rx[1] > range_radius*range_radius:
                rx = [random.uniform(-range_radius,range_radius),random.uniform(-range_radius,range_radius)]
            self.cluster_rx.append(rx)
    
        self.busy_time = []
        for i in len(busy_time):
            self.busy_time.append(busy_time[i] * 60)
             
        if is_busy_time:
            self.is_busy_time = True
            self.busy_time_length = busy_time_length*60
            self.busy_time_weight = busy_time_weight
            self.busy_time_cluster_extra_weight = busy_time_cluster_extra_weight
        else:
            self.is_busy_time = False
             
    def generate(self,time):
        if self.is_busy_time:
            if self.is_now_busy(time):
                return self.generate_busy_time(time)
            else:
                return self.generate_non_busy_time(time)
        else:
            return self.generate_non_busy_time(time)
   
    def generate_busy_time(self,time):
        index = 0
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                index = i
                break
        avg_cluster_tx_weight = np.mean(self.cluster_weight) + self.busy_time_cluster_extra_weight[index] + self.busy_time_weight[index]
        avg_cluster_rx_weight = np.mean(self.cluster_weight) + self.busy_time_cluster_extra_weight[index] + self.busy_time_weight[index]
        
        avg_non_cluster_weight = self.non_cluster_weight + self.busy_time_weight[index]
        generate_num = random.gauss(1,0.33)*self.per_min_base_generate*avg_non_cluster_weight
        generate_num = int(generate_num)
        output = []
        for i in range(generate_num):
            if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                tx = self.generate_cluster_tx()
            else :
                tx = self.genrate_non_cluster()
            if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                rx = self.generate_cluster_rx()
            else :
                rx = self.genrate_non_cluster()
            distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)
            output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance,"index of this minute":index}
            output.append(output_piece)
        return output
    
    def generate_non_busy_time(self,time):
        avg_cluster_tx_weight = np.mean(self.cluster_weight)
        avg_cluster_rx_weight = np.mean(self.cluster_weight)
        avg_non_cluster_weight = self.non_cluster_weight
        generate_num = random.gauss(1,0.33)*self.per_min_base_generate
        if generate_num < 0:
            generate_num = 0
        generate_num = int(generate_num)
        output = []
        for i in range(generate_num):
            if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                tx = self.generate_cluster_tx()
            else :
                tx = self.genrate_non_cluster()
            if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                rx = self.generate_cluster_rx()
            else :
                rx = self.genrate_non_cluster()
            distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)
            output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance}
            output.append(output_piece)
        return output
           
    def generate_cluster_tx(self):
        ## 根据权重生成集群发送点
        select = random.uniform(0,1)
        for i in range(len(self.cluster_weight)):
            if select < sum(self.cluster_weight[:i+1])/sum(self.cluster_weight):
                tx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                while tx[0]*tx[0] + tx[1]*tx[1] > self.cluster_radius*self.cluster_radius:
                    tx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                tx += self.cluster_tx[i]
                return tx
        
    def generate_cluster_rx(self):
        select = random.uniform(0,1)
        for i in range(len(self.cluster_weight)):
            if select < sum(self.cluster_weight[:i+1])/sum(self.cluster_weight):
                rx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                while rx[0]*rx[0] + rx[1]*rx[1] > self.cluster_radius*self.cluster_radius:
                    rx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                rx += self.cluster_rx[i]
                return rx
        
    def genrate_non_cluster(self):
        point = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
        while point[0]*point[0] + point[1]*point[1] > self.range_radius*self.range_radius:
            point = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
        return point
        
    def is_now_busy(self,time):
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                return True
        return False

class P2P_delivery_generator:
    def __init__(   
                    self ,
                    is_center = True ,
                    is_has_charging_station_at_destination = False
                ):
        
        pass
    
class Food_delivery_generator(General_delivery_generator):
    def __init__(
                self ,
                range_radius = 3 ,                                 ## 食物配送场景中的配送半径，eg:5km
                cluster_radius = 0.2 ,                              ## 食物配送场景中的集群半径，eg:0.2km
                num_clusters_tx = 5 ,                               ## 食物配送场景中的集群发送点数量，eg:餐馆，食品配送中心等
                num_clusters_rx = 7 ,                               ## 食物配送场景中的集群接收点数量，eg:写字楼，住宅区等
                tx_cluster_weight = [2,2,2,2,2] ,                   ## 食物配送场景中的集群发送点权重
                rx_cluster_weight = [2,2,2,2,2,2,2] ,               ## 食物配送场景中的集群接收点权重
                non_cluster_weight = 1 ,                            ## 食物配送场景中的非集群权重
                is_busy_time = True ,                               ## 是否设置繁忙时间段
                busy_time = [8,12,18,20] ,                          ## 食物配送场景中的繁忙时间段(小时)
                busy_time_length = 2 ,                              ## 食物配送场景中的繁忙时间段长度(小时)
                busy_time_weight = [0.5,1,1,0.5] ,                  ## 食物配送场景中的繁忙时间段权重
                busy_time_cluster_extra_weight = [0,0.2,0.2,0] ,    ## 食物配送场景中的繁忙时间段集群额外权重
                is_center_cluster_tx = True ,                       ## 是否将一个集群发送点设置为中心点
                per_min_base_generate = 5,                           ## 每分钟生成的食品配送数量
                distance_extra_weight = 0.1                         ## 食品配送距离额外权重
                
                ):
        super().__init__(range_radius,cluster_radius,num_clusters_tx,num_clusters_rx,tx_cluster_weight,rx_cluster_weight,non_cluster_weight,is_busy_time,busy_time,busy_time_length,busy_time_weight,busy_time_cluster_extra_weight,is_center_cluster_tx,per_min_base_generate)
        self.distance_extra_weight = distance_extra_weight
        
    def generate(self,time):
        output = []
        if self.is_busy_time:
            if self.is_now_busy(time):
                output = self.generate_busy_time(time)
            else:
                output = self.generate_non_busy_time(time)
        else:
            output = self.generate_non_busy_time(time)
        for i in range(len(output)):
            output[i]["distance"] = output[i]["distance"]*(1+self.distance_extra_weight)
        return output