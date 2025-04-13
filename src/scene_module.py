import math
import random
import numpy as np


class General_delivery_generator:
    def __init__(
                self ,
                range_radius = 3 ,                                  ## 配送场景中的配送半径，eg:5km
                is_cluster_enable = True ,                          ## 是否生成集群
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
                per_min_base_generate = 5   ,
                min_per_step = 1 ,                                  ## 每分钟生成的配送数量
                ):
        
        
        #assert (len(rx_cluster_weight) == num_clusters_rx) or (len(rx_cluster_weight) == 1) , "rx_cluster_weight长度应该等于num_clusters_rx或者等于1"
        #assert len(tx_cluster_weight) == num_clusters_tx or (len(tx_cluster_weight) == 1) , "tx_cluster_weight长度应该等于num_clusters_tx或者等于1"
        #assert len(busy_time_weight) == len(busy_time) , "busy_time_weight长度应该等于busy_time"
        #assert len(busy_time_cluster_extra_weight) == len(busy_time) , "busy_time_cluster_extra_weight长度应该等于busy_time"
                
        self.is_cluster_enable = is_cluster_enable
        self.range_radius = range_radius
        self.cluster_radius = cluster_radius
        self.num_clusters_tx = num_clusters_tx
        self.num_clusters_rx = num_clusters_rx
        self.is_center_cluster_tx = is_center_cluster_tx
        self.rx_cluster_weight = rx_cluster_weight
        self.tx_cluster_weight = tx_cluster_weight
        self.non_cluster_weight = non_cluster_weight
        self.per_step_base_generate = per_min_base_generate * min_per_step
        ## 生成配送场景 在以半径为range_radius的圆内生成num_clusters_tx个集群发送点和num_clusters_rx个集群接收点
        
        if is_cluster_enable:
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
        for i in range(len(busy_time)):
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
        output = []
        realtime_busy_time_weight = self.busy_time_weight[index] * random.gauss(1,0.25)*(1-math.fabs(time-self.busy_time[index])/self.busy_time_length*2)
        realtime_busy_time_weight_cluster_extra = self.busy_time_cluster_extra_weight[index] * random.gauss(1,0.25)*(1-math.fabs(time-self.busy_time[index])/self.busy_time_length*2)
        if realtime_busy_time_weight < 0:
            realtime_busy_time_weight = 0
        
        if self.is_cluster_enable:
            
            avg_cluster_tx_weight = np.mean(self.tx_cluster_weight) + realtime_busy_time_weight_cluster_extra + realtime_busy_time_weight
            avg_cluster_rx_weight = np.mean(self.rx_cluster_weight) + realtime_busy_time_weight_cluster_extra + realtime_busy_time_weight
            
            avg_non_cluster_weight = self.non_cluster_weight + realtime_busy_time_weight
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate*avg_non_cluster_weight
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                    tx = self.generate_cluster_tx()
                else :
                    tx = self.genrate_non_cluster()
                if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                    rx = self.generate_cluster_rx()
                else :
                    rx = self.genrate_non_cluster()
                distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance,"index of this minute":index}
                output.append(output_piece)
        else:
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate*(self.non_cluster_weight + realtime_busy_time_weight)
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                tx = self.genrate_non_cluster()
                rx = self.genrate_non_cluster()
                distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance,"index of this minute":index}
                output.append(output_piece)
                
            
        return output
    
    def generate_non_busy_time(self,time):
        output = []
        if self.is_cluster_enable:
            avg_cluster_tx_weight = np.mean(self.tx_cluster_weight)
            avg_cluster_rx_weight = np.mean(self.rx_cluster_weight)
            avg_non_cluster_weight = self.non_cluster_weight
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate
            if generate_num < 0:
                generate_num = 0
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                    tx = self.generate_cluster_tx()
                else :
                    tx = self.genrate_non_cluster()
                if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                    rx = self.generate_cluster_rx()
                else :
                    rx = self.genrate_non_cluster()
                distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance}
                output.append(output_piece)
        else:
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                tx = self.genrate_non_cluster()
                rx = self.genrate_non_cluster()
                distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance}
                output.append(output_piece)
        return output
           
    def generate_cluster_tx(self):
        ## 根据权重生成集群发送点
        select = random.uniform(0,1)
        for i in range(len(self.tx_cluster_weight)):
            if select < sum(self.tx_cluster_weight[:i+1])/sum(self.tx_cluster_weight):
                tx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                while tx[0]*tx[0] + tx[1]*tx[1] > self.cluster_radius*self.cluster_radius:
                    tx = [random.uniform(-self.range_radius,self.range_radius),random.uniform(-self.range_radius,self.range_radius)]
                tx += self.cluster_tx[i]
                return tx
        
    def generate_cluster_rx(self):
        select = random.uniform(0,1)
        for i in range(len(self.rx_cluster_weight)):
            if select < sum(self.rx_cluster_weight[:i+1])/sum(self.rx_cluster_weight):
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
                    is_has_charging_station_at_destination = False ,
                    range_radius = 50 ,                                     ## 配送场景中的配送半径，eg:5km
                    num_rx = 20 ,                                           ## 配送场景中的接收点数量
                    is_busy_time = False ,                                  ## 是否设置繁忙时间段
                    per_RX_base_generate = 0.02 ,                           ## 每分钟生成的接收点数量
                    busy_time = [] ,                                        ## 配送场景中的繁忙时间段(小时)
                    busy_time_length = 0 ,                                  ## 配送场景中的繁忙时间段长度(小时)
                    busy_time_weight = [] ,                                ## 配送场景中的繁忙时间段权重
                    min_per_step = 1 ,                                     ## 每分钟生成的配送数量
                ):
        self.is_center = is_center
        self.is_has_charging_station_at_destination = is_has_charging_station_at_destination
        self.range_radius = range_radius
        self.is_busy_time = is_busy_time
        self.num_rx = num_rx
        self.per_RX_base_generate = per_RX_base_generate * min_per_step
        self.busy_time = []
        self.busy_time_weight = busy_time_weight
        self.is_busy_time = is_busy_time
        
        if is_busy_time:
            self.busy_time_length = busy_time_length*60
            for i in len(busy_time):
                self.busy_time.append(busy_time[i] * 60)

        self.rx_list = []
        for i in range(num_rx):
            rx =[range_radius,range_radius]
            while rx[0]*rx[0] + rx[1]*rx[1] > range_radius*range_radius:
                rx = [random.uniform(-range_radius,range_radius),random.uniform(-range_radius,range_radius)]
            self.rx_list.append(rx)
        
    def generate(self,time):
        if self.is_busy_time:
            if self.is_now_busy(time):
                return self.generate_busy_time(time)
            else:
                return self.generate_non_busy_time(time)
        else:
            return self.generate_non_busy_time(time)
    
    def generate_busy_time(self,time):
        output = []
        index = 0
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                index = i
                break
        generate_num = random.gauss(1,0.33)*self.per_RX_base_generate + self.busy_time_weight[index]
        if generate_num < 0:
            generate_num = 0
        generate_num = int(generate_num)
        for i in range(generate_num):
            rx = self.rx_list[random.randint(0,len(self.rx_list)-1)]
            distance = math.sqrt(rx[0]**2+rx[1]**2)
            output_piece = {"rx":rx,"time":time,"distance":distance,"index of this minute":index}
            output.append(output_piece)
        return output
    
    def generate_non_busy_time(self,time):
        output = []
        generate_num = random.gauss(1,0.33)*self.per_RX_base_generate
        generate_num = int(generate_num)
        for i in range(generate_num):
            rx = self.rx_list[random.randint(0,len(self.rx_list)-1)]
            distance = math.sqrt(rx[0]**2+rx[1]**2)
            output_piece = {"rx":rx,"time":time,"distance":distance}
            output.append(output_piece)
        return output
    
    def is_now_busy(self,time):
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                return True
        return False

class Taxi_generator(General_delivery_generator):
    def __init__(
                self ,
                range_radius = 8 ,                                 ## 出租车场景中的配送半径，eg:30km
                cluster_radius = 0.5 ,                              ## 出租车场景中的集群半径，eg:0.5km
                is_cluster_enable = True ,                          ## 是否生成集群
                is_cluster_mixed = True ,                           ## 是否生成混合集群
                num_clusters_tx = 5 ,                               ## 出租车场景中的集群发送点数量，eg:写字楼，住宅区等
                num_clusters_rx = 5 ,                               ## 出租车场景中的集群接收点数量，eg:写字楼，住宅区等
                num_clusters_mixed = 10 ,                           ## 出租车场景中的混合集群数量，eg:写字楼，住宅区等
                tx_cluster_weight = [2,2,2,2,2] ,                   ## 出租车场景中的集群发送点权重
                rx_cluster_weight = [2,2,2,2,2] ,                   ## 出租车场景中的集群接收点权重
                mixed_cluster_weight = [2,2,2,2,2,2,2,2,2,2] ,      ## 出租车场景中的混合集群权重
                non_cluster_weight = 1 ,                            ## 出租车场景中的非集群权重
                is_busy_time = True ,                               ## 是否设置繁忙时间段
                busy_time = [8,12,18,20] ,                          ## 出租车场景中的繁忙时间段(小时)
                busy_time_length = 2 ,                              ## 出租车场景中的繁忙时间段长度(小时)
                busy_time_weight = [1,0.5,1,0.5] ,                  ## 出租车场景中的繁忙时间段权重
                busy_time_cluster_extra_weight = [0.2,0,0.2,0] ,    ## 出租车场景中的繁忙时间段集群额外权重
                is_center_cluster_tx = False ,                      ## 是否将一个集群发送点设置为中心点
                per_step_base_generate = 5,                          ## 每分钟生成的出租车配送数量
                minimum_distance = 3 ,                              ## 出租车配送的最小距离
                distance_extra_weight = 0.1 ,                        ## 出租车配送距离额外权重
                min_per_step = 1 ,                                   ## 每分钟生成的配送数量
                ):
        super().__init__(range_radius,is_cluster_enable,cluster_radius,num_clusters_tx,num_clusters_rx,tx_cluster_weight,rx_cluster_weight,non_cluster_weight,is_busy_time,busy_time,busy_time_length,busy_time_weight,busy_time_cluster_extra_weight,is_center_cluster_tx,per_step_base_generate,min_per_step)
        self.is_cluster_mixed = is_cluster_mixed
        if is_cluster_mixed:
            self.num_clusters_rx = num_clusters_rx + num_clusters_mixed
            self.num_clusters_tx = num_clusters_tx + num_clusters_mixed
            self.tx_cluster_weight += mixed_cluster_weight
            self.rx_cluster_weight += mixed_cluster_weight
            for i in range(num_clusters_mixed):
                mix =[range_radius,range_radius]
                while mix[0]*mix[0] + mix[1]*mix[1] > range_radius*range_radius:
                    mix = [random.uniform(-range_radius,range_radius),random.uniform(-range_radius,range_radius)]
                self.cluster_tx.append(mix)
                self.cluster_rx.append(mix)
        self.minimum_distance = minimum_distance
        self.distance_extra_weight = distance_extra_weight
        
    def generate_busy_time(self,time):
        
        index = 0
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                index = i
                break
        output = []
        realtime_busy_time_weight = self.busy_time_weight[index] * random.gauss(1,0.25)*(1-math.fabs(time-self.busy_time[index])/self.busy_time_length*2)
        realtime_busy_time_weight_cluster_extra = self.busy_time_cluster_extra_weight[index] * random.gauss(1,0.25)*(1-math.fabs(time-self.busy_time[index])/self.busy_time_length*2)
        if realtime_busy_time_weight < 0:
            realtime_busy_time_weight = 0
        
        if self.is_cluster_enable:
            
            avg_cluster_tx_weight = np.mean(self.tx_cluster_weight) + realtime_busy_time_weight_cluster_extra + realtime_busy_time_weight
            avg_cluster_rx_weight = np.mean(self.rx_cluster_weight) + realtime_busy_time_weight_cluster_extra + realtime_busy_time_weight
            
            avg_non_cluster_weight = self.non_cluster_weight + realtime_busy_time_weight
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate*avg_non_cluster_weight
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                while True:
                    if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                        tx = self.generate_cluster_tx()
                    else :
                        tx = self.genrate_non_cluster()
                    if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                        rx = self.generate_cluster_rx()
                    else :
                        rx = self.genrate_non_cluster()
                    distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                    output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance,"index of this minute":index}
                    if distance > self.minimum_distance:
                        break
                output.append(output_piece)
        else:
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate*(self.non_cluster_weight + realtime_busy_time_weight)
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                while True:
                    tx = self.genrate_non_cluster()
                    rx = self.genrate_non_cluster()
                    distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                    output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance,"index of this minute":index}
                    if distance > self.minimum_distance:
                        break
                output.append(output_piece)
                
        return output
    
    def generate_non_busy_time(self,time):
        output = []
        if self.is_cluster_enable:
            avg_cluster_tx_weight = np.mean(self.tx_cluster_weight)
            avg_cluster_rx_weight = np.mean(self.rx_cluster_weight)
            avg_non_cluster_weight = self.non_cluster_weight
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate
            if generate_num < 0:
                generate_num = 0
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                while True:
                    if random.uniform(0,1) < avg_cluster_tx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                        tx = self.generate_cluster_tx()
                    else :
                        tx = self.genrate_non_cluster()
                    if random.uniform(0,1) < avg_cluster_rx_weight/(avg_cluster_tx_weight+avg_cluster_rx_weight+avg_non_cluster_weight):
                        rx = self.generate_cluster_rx()
                    else :
                        rx = self.genrate_non_cluster()
                    distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                    output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance}
                    if distance > self.minimum_distance:
                        break
                output.append(output_piece)
        else:
            generate_num = random.gauss(1,0.33)*self.per_step_base_generate
            generate_num = int(generate_num)
            
            for i in range(generate_num):
                while True:
                    tx = self.genrate_non_cluster()
                    rx = self.genrate_non_cluster()
                    distance = math.sqrt((tx[0]-rx[0])**2 + (tx[1]-rx[1])**2)+math.sqrt(tx[0]**2+tx[1]**2)+math.sqrt(rx[0]**2+rx[1]**2)
                    output_piece = {"tx":tx,"rx":rx,"time":time,"distance":distance}
                    if distance > self.minimum_distance:
                        break
                output.append(output_piece)
        return output


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
        
        
class Goods_delivery_generator(P2P_delivery_generator):
    def __init__(
        self ,
        range_radius = 15 ,                                    ## 配送场景中的配送半径，eg:20km
        num_rx = 50 ,                                          ## 配送场景中的接收点数量
        is_has_charging_station_at_destination = False ,       ## 是否在接收点设置充电站
        is_busy_time = False ,                                 ## 是否设置繁忙时间段
        per_RX_base_generate = 5 ,                             ## 每分钟生成的接收点数量
        cluster_extra_distance = 0.5 ,                         ## 配送场景中的集群额外距离
        cluster_num_avg = 5 ,                                  ## 配送场景中的集群数量平均值
        cluster_num_std = 2 ,                                  ## 配送场景中的集群数量标准差
        min_per_step = 1 ,                                     ## 每分钟生成的配送数量
    ):
        super().__init__(False,is_has_charging_station_at_destination,range_radius,num_rx,is_busy_time,per_RX_base_generate,[],0,[],min_per_step)
        self.cluster_extra_distance = cluster_extra_distance
        self.cluster_num_avg = cluster_num_avg
        self.cluster_num_std = cluster_num_std
    
    def generate_busy_time(self,time):
        output = []
        index = 0
        for i in range(len(self.busy_time)):
            if time >= self.busy_time[i]-self.busy_time_length/2 and time <= self.busy_time[i] + self.busy_time_length/2:
                index = i
                break
        generate_num = random.gauss(1,0.33)*self.per_RX_base_generate + self.busy_time_weight[index]
        if generate_num < 0:
            generate_num = 0
        generate_num = int(generate_num)
        for i in range(generate_num):
            rx = self.rx_list[random.randint(0,len(self.rx_list)-1)]
            distance = math.sqrt(rx[0]**2+rx[1]**2)
            cluster_num = int(random.gauss(self.cluster_num_avg,self.cluster_num_std))
            if cluster_num < 1:
                cluster_num = 1
            distance += cluster_num * self.cluster_extra_distance
            output_piece = {"rx":rx,"time":time,"distance":distance,"index of this minute":index}
            output.append(output_piece)
        return output
    
    def generate_non_busy_time(self,time):
        output = []
        generate_num = random.gauss(1,0.33)*self.per_RX_base_generate
        generate_num = int(generate_num)
        for i in range(generate_num):
            rx = self.rx_list[random.randint(0,len(self.rx_list)-1)]
            distance = math.sqrt(rx[0]**2+rx[1]**2)
            cluster_num = int(random.gauss(self.cluster_num_avg,self.cluster_num_std))
            if cluster_num < 1:
                cluster_num = 1
            distance += cluster_num * self.cluster_extra_distance
            output_piece = {"rx":rx,"time":time,"distance":distance}
            output.append(output_piece)   
        return output     
    
class Food_delivery_generator(General_delivery_generator):
    def __init__(
                self ,
                range_radius = 3 ,                                 ## 食物配送场景中的配送半径，eg:5km
                cluster_radius = 0.2 ,                              ## 食物配送场景中的集群半径，eg:0.2km
                is_cluster_enable = True ,                          ## 是否生成集群
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
                per_step_base_generate = 5,                           ## 每分钟生成的食品配送数量
                distance_extra_weight = 0.1                         ## 食品配送距离额外权重
                
                ):
        super().__init__(range_radius,is_cluster_enable,cluster_radius,num_clusters_tx,num_clusters_rx,tx_cluster_weight,rx_cluster_weight,non_cluster_weight,is_busy_time,busy_time,busy_time_length,busy_time_weight,busy_time_cluster_extra_weight,is_center_cluster_tx,per_step_base_generate)
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