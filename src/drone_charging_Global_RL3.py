import gym 
import RL_utils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import pandas as pd
import math
import copy
import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F
import collections
from tqdm import tqdm
import drone_module
import scene_module
from typing import List, Union , Dict, Any, Tuple

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))  # 激活函数使用LeakyReLU
        return self.fc2(x)
    

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state,is_train = True):  # epsilon-贪婪策略采取动作
        if  is_train: #如果不是训练阶段,则不使用epsilon-greedy策略
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor([state], dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

if __name__ == "__main__":
    drone_num = 100
    drone = drone_module.Normal_Drone_Model(620,25,120,130,min_litoff_land_time=1.5,max_battery_energy=64.26)
    scene_num = 10
    simulation_period = (6*60,24*60,1)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scenelist:List[Union[scene_module.Taxi_generator,scene_module.Goods_delivery_generator]] = []   
    for i in range(scene_num):
        scene_piece = scene_module.Taxi_generator(min_per_step=simulation_period[2])
        scenelist.append(scene_piece)
    

    lr = 2e-4

    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    from gym.envs.registration import register

    register(
        id='DroneChargingEnv-v0',
        entry_point='RL_utils:Drone_charging_env',
    )
    env_name = 'DroneChargingEnv-v0'
    env = gym.make(
        env_name,
        drone = drone,
        drone_num = drone_num, 
        scene_list = scenelist, 
        time_info = simulation_period,
        num_simulation = 5,
        seed = seed
        )
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    replay_buffer = ReplayBuffer(buffer_size)
    
    state_dim = env.observation_space['standby_drone_battery_info'].shape[0] + env.observation_space['charging_drone_info'].shape[0] + env.observation_space['task_drone_info'].shape[0] + env.observation_space['current_time'].shape[0] + env.observation_space['current_order_num'].shape[0] + env.observation_space['existing_order_num'].shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)
    infer_only = False #是否只进行推理
    if infer_only:
        #载入权重
        agent.q_net.load_state_dict(torch.load('model/dqn_model_20231031-1530.pth'))

    if not infer_only:
        env.setmode('train')
        return_list = []
        return_power_list = []
        return_order_list = []
        max_power_list = []
        SLA_list = []
        for i in range(10):
            if i > 3 and i % 2 == 0:
                lr = lr / 2
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode_return = 0
                    power_return = 0
                    order_return = 0
                    state = env.reset()
                    #将state转换为numpy数组
                    state = np.concatenate([state['standby_drone_battery_info'],
                                    state['charging_drone_info'],
                                    state['task_drone_info'],
                                    state['current_time'],
                                    state['current_order_num'],
                                    state['existing_order_num']])
                    done = False
                    while not done:
                        action = agent.take_action(state)
                                # 将动作转换为字典类型
                        next_state, reward, done, total_done, _ = env.step(action)
                        #将next_state转换为numpy数组
                        next_state = np.concatenate([next_state['standby_drone_battery_info'],
                                    next_state['charging_drone_info'],
                                    next_state['task_drone_info'],
                                    next_state['current_time'],
                                    next_state['current_order_num'],
                                    next_state['existing_order_num']])
                        
                        replay_buffer.add(state, action, reward[0], next_state, done)
                        state = next_state
                        episode_return += reward[0]
                        power_return += reward[2]
                        order_return += reward[1]

                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            agent.update(transition_dict)
                    
                    SLA = env.cal_SLA()
                    SLA_list.append(SLA)
                    _,_, max_power = env.get_power_info()
                    max_power_list.append(max_power)
                    return_list.append(episode_return)
                    return_power_list.append(power_return)
                    return_order_list.append(order_return)

                    if (i_episode + 1) % 5 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'power':
                            '%.1f' % np.mean(return_power_list[-5:]),
                            'order':
                            '%.1f' % np.mean(return_order_list[-5:]),
                            'SLA':
                            '%.3f' % np.mean(SLA_list[-5:]),
                            'max_power':
                            '%.1f' % np.mean(max_power_list[-5:])
                        })
                    pbar.update(1)
    
    #保存模型，记录时间到文件名
    if not os.path.exists('model'):
        os.makedirs('model')
    torch.save(agent.q_net.state_dict(), 'model/dqn_model_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))
    #推理
    env.setmode('test')
    while env.index !=0:
        state = env.reset()

    #推理数据统计
    power_list_infer = []
    order_list_infer = []
    SLA_list_infer = []
    max_power_list_infer = []
    avg_power_list_infer = []
    done = False

    while env.index <= env.max_index-1:
        while not done:
            action = agent.take_action(state,is_train=False)
            next_state, reward, done, total_done, _ = env.step(action)
            #将next_state转换为numpy数组
            next_state = np.concatenate([next_state['standby_drone_battery_info'],
                                next_state['charging_drone_info'],
                                next_state['task_drone_info'],
                                next_state['current_time'],
                                next_state['current_order_num'],
                                next_state['existing_order_num']])
            state = next_state
        SLA = env.cal_SLA()
        SLA_list_infer.append(SLA)
        power_list_this_run ,avg_power_this_run,max_power_this_run  = env.get_power_info()
        power_list_infer.append(power_list_this_run)
        avg_power_list_infer.append(avg_power_this_run)
        max_power_list_infer.append(max_power_this_run)
        order_list_this_run = env.order_list_this_run
        order_list_infer.append(order_list_this_run)
        done = False
        env.reset()
    
    #训练数据绘图
    
    episodes_list = list(range(len(return_list)))
    fig1 = plt.figure()
    plt.plot(episodes_list, return_list, label='Reward',color='blue')
    plt.plot(episodes_list, return_power_list, label='Power_reward',color='orange')
    plt.plot(episodes_list, return_order_list, label='Order_reward',color='green') 



    #保存推理数据
    if not os.path.exists('infer_data'):
        data ={}
        data['power_list'] = power_list_infer
        data['order_list'] = order_list_infer
        data['SLA_list'] = SLA_list_infer
        data['max_power_list'] = max_power_list_infer
        data['avg_power_list'] = avg_power_list_infer


        

        