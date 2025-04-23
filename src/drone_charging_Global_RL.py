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


class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.dropout1 = torch.nn.Dropout(0.2)  # Dropout层,防止过拟合
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        A = self.fc_A(F.leaky_relu(x))
        V = self.fc_V(F.leaky_relu(x))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
    

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = VAnet(state_dim, hidden_dim,self.action_dim).to(device)
        self.target_q_net = VAnet(state_dim, hidden_dim,self.action_dim).to(device)
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
    
    def train(self):
        self.q_net.train()
        self.target_q_net.train()
    
    def eval(self):
        self.q_net.eval()

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
    

    lr = 2e-3

    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    num_simulation = 5
    infer_only = True #是否只进行推理
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    from gym.envs.registration import register
    if infer_only:
        num_simulation = 20
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
        num_simulation = num_simulation,
        seed = seed
        )
    env.delay_reward_k = -0.01
    env.power_reward_k_2 = -6.0
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    replay_buffer = ReplayBuffer(buffer_size)
    
    state_dim = env.observation_space['standby_drone_battery_info'].shape[0] + env.observation_space['charging_drone_info'].shape[0] + env.observation_space['task_drone_info'].shape[0] + env.observation_space['current_time'].shape[0] + env.observation_space['current_order_num'].shape[0] + env.observation_space['existing_order_num'].shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    if infer_only:
        #载入权重
        agent.q_net.load_state_dict(torch.load('model/dqn_model_20250423-113618.pth'))
    #agent.q_net.load_state_dict(torch.load('model/dqn_model_20250423-101941.pth'))
    if not infer_only:
        env.setmode('train')
        agent.train()
        return_list = []
        return_power_list = []
        return_order_list = []
        max_power_list = []
        SLA_list = []
        is_early_stop = False
        for i in range(10):
            if i > 3 and i % 2 == 0:
                lr = lr / 2
            if is_early_stop:
                break
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
                        if np.mean(return_list[-5:]) > -20 and i> 5:
                            print('Early stop, stop training')
                            is_early_stop = True
                            break
                    pbar.update(1)

    
    #保存模型，记录时间到文件名
    if not os.path.exists('model'):
        os.makedirs('model')
    if not infer_only:
        torch.save(agent.q_net.state_dict(), 'model/dqn_model_{}.pth'.format(time.strftime("%Y%m%d-%H%M%S")))
    #推理
    agent.eval()
    env.setmode('test')
    if infer_only:
        env.setmode('default')
    state = env.reset()
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
        state = env.reset()
        state = np.concatenate([state['standby_drone_battery_info'],
                            state['charging_drone_info'],
                            state['task_drone_info'],
                            state['current_time'],
                            state['current_order_num'],
                            state['existing_order_num']])
        done = False
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
        print('index:',env.index)
        SLA = env.cal_SLA()
        SLA_list_infer.append(SLA)
        power_list_this_run ,avg_power_this_run,max_power_this_run  = env.get_power_info()
        power_list_infer.append(power_list_this_run)
        avg_power_list_infer.append(avg_power_this_run)
        max_power_list_infer.append(max_power_this_run)
        order_list_this_run = env.order_list_this_run
        order_list_infer.append(order_list_this_run)
        done = False
        
    
    #训练数据绘图
    if not infer_only:
        episodes_list = list(range(len(return_list)))
        mv_return = RL_utils.moving_average(return_list, 9)
        mv_return_power = RL_utils.moving_average(return_power_list, 9)
        mv_return_order = RL_utils.moving_average(return_order_list, 9)
        fig1 = plt.figure()
        plt.plot(episodes_list, mv_return, label='Reward',color='blue')
        plt.plot(episodes_list, mv_return_power, label='Power_reward',color='orange')
        plt.plot(episodes_list, mv_return_order, label='Order_reward',color='green') 
        #图例
        plt.legend(loc='upper left')
        plt.show()
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        if not os.path.exists('img'):
            os.makedirs('img')
        fig1.savefig('img/Reward.png')
        plt.close(fig1)

    #保存推理数据到json 文件名带时间戳
    if not os.path.exists('infer_data'):
        os.makedirs('infer_data')
    data ={}
    data['power_list'] = power_list_infer
    data['order_list'] = order_list_infer
    data['SLA_list'] = SLA_list_infer
    data['max_power_list'] = max_power_list_infer
    data['avg_power_list'] = avg_power_list_infer
    import json
    with open('infer_data/infer_data_{}.json'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
        json.dump(data, f, indent=4)
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv('infer_data/infer_data_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")), index=False)


    avg_max_ratio_list = []
    for i in range(len(power_list_infer)):
        avg_max_ratio_list.append(max_power_list_infer[i]/avg_power_list_infer[i])
    
    max_index = np.argmax(avg_max_ratio_list)
    #绘图
    num_period = (simulation_period[1] - simulation_period[0]) // 10 +1
    order_num_list = np.zeros(num_period)
    for i in range(len(order_list_infer)):
        order_list_piece = order_list_infer[i]
        for order in order_list_piece:
            period_index = (order['tx_time'] - simulation_period[0]) // 10
            if period_index < num_period:
                order_num_list[period_index] += 1    
    time_list_10min = []
    for i in range(num_period):
        time_list_10min.append((simulation_period[0] + i*10)/60.0)

    # 画出充电功率的折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置x轴和y轴的范围
    ax.set_xlim(simulation_period[0]/60.0,simulation_period[1]/60.0)
    ax.set_ylim(0, (max (max_power_list_infer)*120)//100)
    ax_2 = ax.twinx()
    ax_2.set_ylim(0, (max (order_num_list)*120)//100)
    for i in range(len(max_power_list_infer)):
        time_list = []
        p_list = []
        start_time = simulation_period[0] 
        index = 0
        for chargingpower_1min in power_list_infer[i]:
            index += 1
            time_list.append(start_time/60.0)
            start_time += simulation_period[2]
            p_list.append(chargingpower_1min)
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
    ax.set_title('charging power of taxi drone with ' + 'DQN' + ' strategy')
    # 设置备注,保留到整数
    ax.text(simulation_period[1]/60.0*0.8, max (max_power_list_infer)*0.3, 'Peak charging power: ' + str(int(max_power_list_infer[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')
    ax.text(simulation_period[1]/60.0*0.8, max (max_power_list_infer)*0.15, 'Avg charging power: ' + str(int(avg_power_list_infer[max_index])) + 'KW', ha='center', va='center', fontsize=10, color='#000000')
    # 倍率，保留三位小数
    ax.text(simulation_period[1]/60.0*0.8, max (max_power_list_infer)*0.45, 'Peak/Avg ratio: ' + str(round(avg_max_ratio_list[max_index],3)), ha='center', va='center', fontsize=10, color='#000000')
    # 显示图形
    plt.show()
        
    print('SLA:', np.mean(SLA_list_infer))    
    

        
        