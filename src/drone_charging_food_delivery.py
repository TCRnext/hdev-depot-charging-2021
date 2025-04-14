from matplotlib.pylab import f
import drone_module
import scene_module
import copy

def scene_food_delivery_analysis(
    non_default_scene_module = None,
    non_default_drone_module = None,
    num_drone = 500,
    simulation_period = (6*60,24*60,1),
    num_simulation = 1,
    is_same_scene = True,
    
):
    drone_list = []
    if non_default_scene_module is not None:
        scene= non_default_scene_module(min_per_step=simulation_period[2])
    else:
        scene = scene_module.Food_delivery_generator(per_min_base_generate = 40,min_per_step=simulation_period[2] )
    if non_default_drone_module is not None:
        drone = non_default_drone_module
    else:
        drone = drone_module.Normal_Drone_Model(1.5,19,65,40,max_battery_energy=0.067)
    order_list = []
    chargingpower_list = []
    if is_same_scene:
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece = food_delivery_simulate_core(
                num_drone,
                simulation_period,
                scene,
                drone,
                drone_full_power_at_begin = True,
                enable_partly_charging = True,
                charge_strat = 'immediate',
                minimum_num_charging_drone = 25,
                power_limit = 100,              #KW
                standby_drone_battery = 0.9,          #KW
                least_standby_drone_percent = 0.2,    #KW
            )
            order_list.append(order_list_piece)
            chargingpower_list.append(chargingpower_list_piece)
    else:
        if non_default_scene_module is not None:
            scene= non_default_scene_module(min_per_step=simulation_period[2])
        else:
            scene = scene_module.Food_delivery_generator(per_min_base_generate = 40 ,min_per_step=simulation_period[2])
        for simulation in range(num_simulation):
            order_list_piece, chargingpower_list_piece = food_delivery_simulate_core(
                num_drone,
                simulation_period,
                scene,
                drone,
                drone_full_power_at_begin = True,
                enable_partly_charging = True,
                charge_strat = 'immediate',
                minimum_num_charging_drone = 25,
                power_limit = 100,              #KW
                standby_drone_battery = 0.9,          #KW
                least_standby_drone_percent = 0.2,    #KW
            )
            order_list.append(order_list_piece)
            chargingpower_list.append(chargingpower_list_piece)
    return order_list, chargingpower_list
        
  
def food_delivery_simulate_core(
                                num_drone,
                                simulation_period,
                                scene:scene_module.Food_delivery_generator ,
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
        
        current_charging_power = 0
        current_is_charging_drone = 0
        drone:drone_module.Normal_Drone_Model
        for drone in drone_list:
            drone.update_status(simulation_period[2])
            if drone.is_parking_outside == False:
                if drone.drone_status == drone.status_charge:
                    current_is_charging_drone += 1
                    current_charging_power += drone.current_charge_power
                
        chargingpower_list.append((time//60,time%60,current_charging_power,current_is_charging_drone))
        

     
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
            for drone in drone_list:
                if drone.drone_status == drone.status_standby:
                    num_charging_drone += 1
            if num_charging_drone < minimum_num_charging_drone:
                for drone in drone_list:
                    if drone.drone_status == drone.status_standby and drone.now_battery < 0.99:
                        drone.status_to_charge()
                        num_charging_drone += 1
                    if num_charging_drone >= minimum_num_charging_drone:
                        break
        if(time%60 == 0):
            print("time:",time//60)

                
    print("订单共有",len(order_list),"个") 
    print(chargingpower_list)                        
    return order_list, chargingpower_list 

if __name__ == '__main__':
    scene_food_delivery_analysis()