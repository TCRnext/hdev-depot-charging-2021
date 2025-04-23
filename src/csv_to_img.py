

if __name__ == "__main__":
    import pandas as pd



    file_path = "infer_data\infer_data_20250423-122846.csv"""
    df = pd.read_csv(file_path)

    # 将DataFrame转换回原始字典格式
    data = {
        'power_list': df['power_list'].tolist(),
        'order_list': df['order_list'].tolist(),
        'SLA_list': df['SLA_list'].tolist(),
        'max_power_list': df['max_power_list'].tolist(),
        'avg_power_list': df['avg_power_list'].tolist()
    }
    import numpy as np
    # 现在可以使用这些列表了
    power_list_infer = data['power_list']
    order_list_infer = data['order_list']
    SLA_list_infer = data['SLA_list']
    max_power_list_infer = data['max_power_list']
    avg_power_list_infer = data['avg_power_list']
    simulation_period =(6*60,24*60,1)

    for piece in power_list_infer:
        piece = piece.tolist()
    for piece in order_list_infer:
        pass
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