import copy
import cProfile
import gc
import math
import pstats
import random
import sys

import networkx as nx
import numpy as np
import torch.nn.functional as F

# sys.path.append('/Users/tobiaskolstobergeland/Documents/IndØk/10.Semester/ProsjektOppgave/Repo/SpeedOptiRepo/MIRPSO_M.py')
from MIRPSO_M import (build_problem, build_simplified_RL_model,
                      visualize_network_for_vessel)


class MIRPSOEnv():
    def __init__(self, PORTS, VESSELS, VESSEL_ARCS, NODES, TIME_PERIOD_RANGE, SOURCE_NODE, SINK_NODE, NODE_DICT):
        # Ports have initial inventory and rate of consumption/production
        self.PORTS = PORTS
        self.VESSELS = VESSELS
        self.VESSEL_ARCS = VESSEL_ARCS
        self.NODES = NODES
        self.TIME_PERIOD_RANGE = TIME_PERIOD_RANGE
        self.SOURCE_NODE = SOURCE_NODE
        self.SINK_NODE = SINK_NODE
        self.NODE_DICT = NODE_DICT
        self.PORT_DICT = {p.number: p for p in PORTS}
        self.PORT_DICT[SINK_NODE.port.number] = SINK_NODE.port
        # self.PORT_DICT[SOURCE_NODE.port.number] = SOURCE_NODE.port
        self.VESSEL_DICT = {v.number: v for v in VESSELS}
        self.TRAVEL_TIME_DICT = {}
            
        for origin_port in self.PORT_DICT.values():
            if origin_port.number in [self.SOURCE_NODE.port.number, self.SINK_NODE.port.number]:
                continue
            
            travel_times = [0] * len(PORTS)  # or a high value to indicate no direct path
            # Iterate over all possible destination ports
            for destination_port in self.PORT_DICT.values():
                if destination_port.number in [self.SOURCE_NODE.port.number, self.SINK_NODE.port.number]:
                    continue
                # Skip arcs to the same port, those from source, and to sink
                if destination_port != origin_port:
                    for arc in self.VESSEL_ARCS[self.VESSELS[0]]:
                        if arc.origin_node.port == origin_port and arc.destination_node.port == destination_port:
                            # Update the travel time for the destination port, adjusting for 0-based index
                            travel_times[destination_port.number - 1] = arc.travel_time
                            break  # Break here ensures only the first found arc is used
            self.TRAVEL_TIME_DICT[origin_port.number] = travel_times
       
        self.INITIAL_PORT_INVENTORY= {}
        # Create a dictionary to store the initial inventory of each port. Use it later to reset the environment
        for port in self.PORTS:
            self.INITIAL_PORT_INVENTORY[port] = port.inventory
            
        # Do similarly for the vessels
        self.INITIAL_VESSEL_INVENTORY = {}
        for vessel in self.VESSELS:
            self.INITIAL_VESSEL_INVENTORY[vessel] = vessel.inventory
        
        self.vessel_paths = {}
        # Set the position of the vessels to the source node, and initialize an empty path for each vessel
        for vessel in self.VESSELS:
            vessel.position = self.SOURCE_NODE.port
            vessel.in_transit_towards = None
            self.vessel_paths[vessel] = vessel.action_path
            
        self.state = {
            'ports': self.PORTS,
            'vessels': self.VESSELS,
            'port_dict': self.PORT_DICT,
            'vessel_dict': self.VESSEL_DICT,
            'time' : 0,
            'done' : False,
            'infeasible' : False,
        }
        
    def reset(self):
        # Reset the environment to an initial state
        # Initialize your state here
        # Ports, vessels, vessel_arcs, nodes and timeperiodrange is already initialized in the __init__ method. Only port- and vessel inventory and vessel position needs to be reset.
        for port in self.PORTS:
            port.inventory = self.INITIAL_PORT_INVENTORY[port]
        for vessel in self.VESSELS:
            vessel.inventory = self.INITIAL_VESSEL_INVENTORY[vessel]
            vessel.position = self.SOURCE_NODE.port
            vessel.in_transit_towards = None
            vessel.path = []
            vessel.action_path = []
            vessel.isFinished = False
            
        self.state = {
            'ports': self.PORTS,
            'vessels': self.VESSELS,
            'port_dict': self.PORT_DICT,
            'vessel_dict': self.VESSEL_DICT,
            'time' : 0,
            'done' : False,
            'infeasible' : False,
        }
            
        return self.state
   
    # def encode_state(self, state, vessel):
    #     port_inventories = np.array([(port.inventory)/port.capacity for port in state['ports']])
    #     vessel_inventories = np.array([(vessel.inventory)/(vessel.max_inventory) for vessel in state['vessels']])
    #     current_vessel_number = [vessel.number]
        
    #     # Initialize arrays to store the position and in-transit status
    #     vessel_positions = np.zeros(len(state['vessels']))
    #     vessel_in_transit = np.zeros(len(state['vessels']))  # 0 if at port, remaining time to next arrival if in transit

    #     for i, vessel in enumerate(state['vessels']):
    #         if vessel.position:  # Vessel is at a port
    #             vessel_positions[i] = vessel.position.number
    #         elif vessel.in_transit_towards:  # Vessel is in transit
    #             vessel_positions[i] = vessel.in_transit_towards[0].number
    #             # Time the vessel reaches the port
    #             vessel_in_transit[i] = vessel.in_transit_towards[1]
        
    #     # time_period = np.array([self.state['time'] / max(self.TIME_PERIOD_RANGE)])
    #     time = [state['time']]
    #     # vessel_number = [vessel.number]
    #     #print(time_period)
    #     encoded_state = np.concatenate([port_inventories, vessel_inventories, vessel_positions, vessel_in_transit, time, current_vessel_number])
    #     return encoded_state
    
    def custom_deep_copy_of_state(self, state):
        new_state = {
            'port_dict': {},
            'vessel_dict': {},
            'time': state['time']  # Directly copy the scalar value
        }
        
         # Copy necessary port attributes
        for port_number, port in state['port_dict'].items():
            new_state['port_dict'][port_number] = {
                'capacity': port.capacity,
                'inventory': port.inventory,
                'number': port.number,
                'isLoadingPort': port.isLoadingPort,
                'rate': port.rate
            }
        
        # Copy necessary vessel attributes
        for vessel_number, vessel in state['vessel_dict'].items():
            new_state['vessel_dict'][vessel_number] = {
                'max_inventory': vessel.max_inventory,
                'inventory': vessel.inventory,
                'number': vessel.number,
                #If position is None put None, else put the position
                'position': vessel.position.number if vessel.position is not None else None,
                'in_transit_towards': None if vessel.in_transit_towards is None else {
                    'destination_port_number': vessel.in_transit_towards[0].number,
                    'destination_time': vessel.in_transit_towards[1]
                }
            }
        return new_state
        
        
    
    def encode_state(self, state, vessel_simp):
        #Simplified Vessel also
        # We instead take in a simplified state with only the necessary attributes
        
        # Skip the sink port
        port_dict = {k: port for k, port in state['port_dict'].items() if port['capacity'] is not None}
        vessel_dict = state['vessel_dict']
        
        port_critical_times = []
        # Use the rates to calculate the amount of time periods it takes before the port is empty
        for port in port_dict.values():
            rate = port['rate']
            current_inventory = port['inventory']
            if port['isLoadingPort'] != 1:
                time_to_empty = max(0, math.floor(current_inventory / rate))
            else:
                time_to_empty = max(0, math.floor((port['capacity'] - current_inventory )/ rate))
            port_critical_times.append(time_to_empty)
            
        # Convert to numpy array
        port_critical_times = np.array(port_critical_times)
                
            
        
        # If the port is a loading port, invert the inventory value. This way, all ports status are "good" if the value are high, and "bad" if the value is low.
        port_inventories = np.array([(port['capacity'] - port['inventory']) / port['capacity'] if port.get('is_loading_port', 1) else port['inventory'] / port['capacity'] for port in port_dict.values()])

        # port_inventories = np.array([port['inventory'] / port['capacity'] for port in port_dict.values()])
        
        vessel_inventories = np.array([vessel['inventory'] / vessel['max_inventory'] for vessel in vessel_dict.values()])
        current_vessel_number = np.array([vessel_simp['number']])
        if vessel_simp['position'] is len(port_dict) + 1:
            travel_times = np.array([0 for p in port_dict.values()])
        else:
            travel_times = np.array(self.TRAVEL_TIME_DICT[vessel_simp['position']])
            
        
        time = state['time']
        # Create a try catch block to handle cases where a vessel in the vessel_dict is on wrong format
        try :
            vessel_positions = np.array([v['position'] if v['position'] else v['in_transit_towards']['destination_port_number'] for v in vessel_dict.values()])
            vessel_in_transit = np.array([v['in_transit_towards']['destination_time'] if v['in_transit_towards'] else 0 for v in vessel_dict.values()])
        except:
            vessel_positions = np.array([0 for v in vessel_dict.values()])
            vessel_in_transit = np.array([0 for v in vessel_dict.values()])
        # Create the vessel_positions and vessel_in_transit arrays. If vessel position is None put in 0.
        
        # Create a np.array with values eqaul to (time - vessel_in_transit) values
        vessel_in_transit = np.array([v['in_transit_towards']['destination_time'] - time if v['in_transit_towards'] else 0 for v in vessel_dict.values()])
        
        
        
        # vessel_positions = np.array([v['position'] if v['position'] else v['in_transit_towards']['destination_port_number'] for v in vessel_dict.values()])
        # vessel_in_transit = np.array([v['in_transit_towards']['destination_time'] if v['in_transit_towards'] else 0 for v in vessel_dict.values()])
        
        # time = np.array([state['time']])
        
        # Combine all features into a single numpy array
        # encoded_state = np.concatenate([port_critical_times, vessel_inventories, vessel_positions, vessel_in_transit, travel_times, time, current_vessel_number])
        encoded_state = np.concatenate([port_critical_times, vessel_inventories, vessel_positions, vessel_in_transit, travel_times, current_vessel_number])
        return encoded_state
    
    
    def encode_actual_state(self, state, vessel):
        port_inventories = np.array([port.inventory / port.capacity for port in state['ports']])
        vessel_inventories = np.array([v.inventory / v.max_inventory for v in state['vessels']])
        current_vessel_number = np.array([vessel.number])

        # Precompute vessel positions and in-transit statuses using numpy operations where possible
        vessel_positions = np.array([v.position.number if v.position else v.in_transit_towards[0].number for v in state['vessels']])
        vessel_in_transit = np.array([v.in_transit_towards[1] if v.in_transit_towards else 0 for v in state['vessels']])
        travel_times = np.array(self.TRAVEL_TIME_DICT[vessel.position])
        
        
        # Assuming state['time'] is a scalar value representing the current time
        time = np.array([state['time']])

        # Combine all features into a single numpy array
        encoded_state = np.concatenate([port_inventories, vessel_inventories, vessel_positions, vessel_in_transit, travel_times, time, current_vessel_number])
        return encoded_state
        
    
     
    
    def find_legal_actions_for_vessel(self, state, vessel):
        # Initialize the operation type and vessel ID

        # Determine the operation type based on the vessel's position and the state's time
        if state['time'] == 0:
            operation_type = 0
        elif vessel.position.isLoadingPort == 1:
            operation_type = 1  # Loading at a production port
        else:
            operation_type = 2  # Offloading at a consumption port

        # Get legal quantities and arcs for the vessel
        legal_quantity = self.get_legal_quantities(operation_type=operation_type, vessel=vessel)
        legal_arcs = self.get_legal_arcs(state=state, vessel=vessel, quantity=legal_quantity)

        # Generate legal actions, considering special conditions
        legal_actions = []
        for arc in legal_arcs:
            legal_actions.append((vessel.number, operation_type, legal_quantity, arc))
        return legal_actions
    
    def sim_find_legal_actions_for_vessel(self, state, vessel):
        # Initialize the operation type and vessel ID
        position_port_number = vessel['position']
        position_port = state['port_dict'][position_port_number]
        # Determine the operation type based on the vessel's position and the state's time
        if state['time'] == 0:
            operation_type = 0
        elif position_port['isLoadingPort'] == 1:
            operation_type = 1
        else:
            operation_type = 2
            
        legal_quantity = self.sim_get_legal_quantities(operation_type=operation_type, vessel=vessel, position_port=position_port)
        legal_arcs = self.sim_get_legal_arcs(state=state, vessel=vessel, quantity=legal_quantity)
        
        legal_actions = []
        for arc in legal_arcs:
            legal_actions.append((vessel['number'], operation_type, legal_quantity, arc))
        return legal_actions
        
        
    def sim_get_legal_arcs(self, state, vessel, quantity):
            
            # Find the node the vessel is currently at
            current_node_key = (vessel['position'], state['time'])
            current_node = self.NODE_DICT[current_node_key]
            
            # Check if vessel is at a loading port or a consumption port
            if current_node.port.isLoadingPort == 1:
                # Vessel is at a loading port
                inventory_after_operation = vessel['inventory'] + quantity
            else:
                # Vessel is at a consumption port
                inventory_after_operation = vessel['inventory'] - quantity
                
            non_sim_vessel = self.VESSEL_DICT[vessel['number']]
                
            # Pre-filter arcs that originate from the current node
            potential_arcs = [arc for arc in self.VESSEL_ARCS[non_sim_vessel] if arc.origin_node == current_node]
            
            # Filter based on vessel state and destination port characteristics
            if inventory_after_operation == 0:
                # Vessel is empty, can only travel to loading ports or sink
                legal_arcs = [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node.port == self.SINK_NODE.port]
            elif inventory_after_operation == vessel['max_inventory']:
                # Vessel is full, can only travel to consumption ports or sink
                legal_arcs = [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1 or arc.destination_node.port == self.SINK_NODE.port]
            else:
                # Vessel can travel anywhere except back to the same port
                legal_arcs = [arc for arc in potential_arcs if arc.destination_node.port != arc.origin_node.port]
                
            # Remove the sink arc if there are other legal arcs
            if len(legal_arcs) > 1:
                legal_arcs = [arc for arc in legal_arcs if arc.destination_node.port != self.SINK_NODE.port]
    
            return legal_arcs

    
    def get_legal_arcs(self, state, vessel, quantity):
        
        # Find the node the vessel is currently at
        current_node_key = (vessel.position.number, state['time'])
        current_node = self.NODE_DICT[current_node_key]
        
        # Check if vessel is at a loading port or a consumption port
        if current_node.port.isLoadingPort == 1:
            # Vessel is at a loading port
            inventory_after_operation = vessel.inventory + quantity
        else:
            # Vessel is at a consumption port
            inventory_after_operation = vessel.inventory - quantity
            
        
        # Pre-filter arcs that originate from the current node
        potential_arcs = [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
        
        # Filter based on vessel state and destination port characteristics
        if inventory_after_operation == 0:
            # Vessel is empty, can only travel to loading ports or sink
            legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node == self.SINK_NODE}
        elif inventory_after_operation == vessel.max_inventory:
            # Vessel is full, can only travel to consumption ports or sink
            legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1 or arc.destination_node == self.SINK_NODE}
        else:
            # Vessel can travel anywhere except back to the same port
            legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port != arc.origin_node.port}
            
        # Remove the sink arc if there are other legal arcs
        if len(legal_arcs) > 1:
            legal_arcs = {arc for arc in legal_arcs if arc.destination_node.port != self.SINK_NODE.port}

        return legal_arcs
    
    
    def sim_get_legal_quantities(self, operation_type, vessel, position_port):

            if operation_type == 0:
                return 0
            
            else:
                
                # Calculate available capacity or inventory based on port type.
                if position_port['isLoadingPort'] == 1:
                    # For loading ports, calculate the maximum quantity that can be loaded onto the vessel.
                    port_limiting_quantity = position_port['inventory']
                    vessel_limiting_quantity = vessel['max_inventory'] - vessel['inventory']
                    return min(port_limiting_quantity, vessel_limiting_quantity, position_port['capacity'])
                else:
                    # For discharging ports, calculate the maximum quantity that can be unloaded from the vessel.
                    port_limiting_quantity = position_port['capacity'] - position_port['inventory']
                    vessel_limiting_quantity = vessel['inventory']
                    return min(port_limiting_quantity, vessel_limiting_quantity, position_port['capacity'])
                
                
                
    def get_legal_quantities(self, operation_type, vessel):
        
        if operation_type == 0:
            return 0
        
        else:
            port = vessel.position
            # Calculate available capacity or inventory based on port type.
            if port.isLoadingPort == 1:
                # For loading ports, calculate the maximum quantity that can be loaded onto the vessel.
                port_limiting_quantity = port.inventory
                vessel_limiting_quantity = vessel.max_inventory - vessel.inventory
                return min(port_limiting_quantity, vessel_limiting_quantity, port.max_amount)
            else:
                # For discharging ports, calculate the maximum quantity that can be unloaded from the vessel.
                port_limiting_quantity = port.capacity - port.inventory
                vessel_limiting_quantity = vessel.inventory
                return min(port_limiting_quantity, vessel_limiting_quantity, port.max_amount)
                
    def update_vessel_status(self, state):
        time = state['time']
        for vessel in state['vessels']:
            destination_port = vessel.in_transit_towards[0]
            destination_time = vessel.in_transit_towards[1]
            if time == destination_time:
                # Vessel has reached its destination
                # Update the vessel's position
                self.update_vessel_position(vessel=vessel, new_position=destination_port)
                
                
    def sim_update_vessel_status(self, state):
        time = state['time']
        for vessel in state['vessel_dict'].values():
            destination_port = vessel['in_transit_towards']['destination_port_number']
            destination_time = vessel['in_transit_towards']['destination_time']
            if time == destination_time:
                # Vessel has reached its destination
                # Update the vessel's position
                self.sim_update_vessel_position(vessel=vessel, new_position=destination_port)
                
    def update_vessel_in_transition_and_inv_for_state(self, state, vessel, destination_port, destination_time, origin_port, quantity, operation_type):
        
        # vessel_copy = state['vessel_dict'][vessel.number]
        # origin_port = state['port_dict'][origin_port.number]
        # destination_port = state['port_dict'][destination_port.number]
        
        # vessel_copy.in_transit_towards = (destination_port, destination_time)
        # vessel_copy.position = None
        
        # if operation_type == 1:
        #     origin_port.inventory -= quantity
        #     vessel_copy.inventory += quantity
        # else:
        #     origin_port.inventory += quantity
        #     vessel_copy.inventory -= quantity
            
        v = state['vessel_dict'][vessel['number']]
        # if v['position'] is not None:
        in_transit_towards = {}
        in_transit_towards['destination_port_number'] = destination_port.number
        in_transit_towards['destination_time'] = destination_time
        v['in_transit_towards'] = in_transit_towards
        # v['in_transit_towards']['destination_port'] = destination_port.number
        # v['in_transit_towards']['destination_time'] = destination_time
        # v['in_transit_towards'] = (destination_port, destination_time)    
        v['position'] = None
        
        origin_port = state['port_dict'][origin_port.number]
        destination_port = state['port_dict'][destination_port.number]
        if operation_type == 1:
            origin_port['inventory'] -= quantity
            v['inventory'] += quantity
        else:
            origin_port['inventory'] += quantity
            v['inventory'] -= quantity
            
        return state
        
                
        
    def is_infeasible(self, state):
        # Implement a check for infeasible states
        # For example, if the inventory of a port or vessel is negative, the state is infeasible
        for port in state['ports']:
            # Use 0 as a lower limit for inventory
            if port.inventory < 0 or port.inventory > port.capacity:
                return True
        for vessel in state['vessels']:
            if vessel.inventory < 0 or vessel.inventory > vessel.max_inventory:
                return True
        return False
    
    def sim_is_infeasible(self, state):
        # Implement a check for infeasible states
        # For example, if the inventory of a port or vessel is negative, the state is infeasible
        port_dict = {k: v for k, v in state['port_dict'].items() if v['capacity'] is not None}
        
        for port in port_dict.values():
            # Use 0 as a lower limit for inventory
            if port['inventory'] < 0 or port['inventory'] > port['capacity']:
                return True
        for vessel in state['vessel_dict'].values():
            if vessel['inventory'] < 0 or vessel['inventory'] > vessel['max_inventory']:
                return True
        return False

    def is_terminal(self, state):
        if state['time'] == len(self.TIME_PERIOD_RANGE):
            return True
        
    def check_state(self, state, experience_path, replay, agent):
        '''Evaluates the state and returns status and reward.'''
        total_reward_for_path  = 0
        
        if self.is_terminal(state):
            experience_path = self.update_rewards_in_experience_path(experience_path, agent)
            for exp in experience_path:
                replay.push(exp, self)
                rew = exp[3]
                total_reward_for_path += rew
            state['done'] = True
            state['infeasible'] = self.is_infeasible(state=state)
            return state, total_reward_for_path
        return state, None
    
    def log_episode(self, episode, total_reward_for_path, experience_path):
        infeasibility_counter = 0
        infeasibility_dict = {}
        
        for exp in experience_path:
            result_state = exp[4]
            time = result_state['time']
            if result_state['infeasible']:
                if time not in infeasibility_dict.keys():
                    infeasibility_dict[time] = True
        
        infeasibility_counter = len(infeasibility_dict.keys())
        
            
        print(f"Episode {episode}: Total Reward = {total_reward_for_path}, Infeasibility Counter = {infeasibility_counter}")
        if infeasibility_counter > 0:
            print('Infeasible time periods:', infeasibility_dict.keys())
        if infeasibility_counter == 0:
            print('Feasible solution:)')
            
        first_infeasible_time = experience_path[0][7]
        return first_infeasible_time, infeasibility_counter
        
        
    def update_rewards_in_experience_path(self, experience_path, agent):
        
        feasible_path = True
        first_infeasible_time = None
        for exp in experience_path:
            _, _, _, _, next_state, _, _, fi_time,= exp
            port_dict = {k: port for k, port in next_state['port_dict'].items() if port['capacity'] is not None}
            num_infeasible_ports = len([port for port in port_dict.values() if port['inventory'] < 0 or port['inventory'] > port['capacity']])
            if num_infeasible_ports > 0:
                # There may be a simpler infeasibility check
                feasible_path = False
                if first_infeasible_time is None:
                    first_infeasible_time = next_state['time']
                break  # Exit the loop early if any infeasible state is found
            
        if feasible_path:
            print('Feasible path found')

        # Extra reward for fully feasible path
        extra_reward_for_feasible_path = 10000
        penalty_for_infeasibility = -10000
        
        
        for exp in experience_path:
            current_state, _, vessel, _, next_state, earliest_vessel, _, fi_time = exp
            
            
            # Num feasible ports in next state
            port_dict = {k: port for k, port in next_state['port_dict'].items() if port['capacity'] is not None}
            # num_feasible_ports = len([port for port in port_dict.values() if 0 <= port['inventory'] <= port['capacity']])
            num_infeasible_ports = len([port for port in port_dict.values() if port['inventory'] < 0 or port['inventory'] > port['capacity']])
            
            if num_infeasible_ports > 0:
                # num_ports = len(port_dict)
                # immediate_reward = -(num_infeasible_ports / num_ports)
                immediate_reward = -num_infeasible_ports*100
            else:
                immediate_reward = 100
            
            # num_infeasible_ports = len([port for port in next_state['port'] if port.inventory < 0 or port.inventory > port.capacity])
            # Calculate when the state became infeasible. Save for later
            # immediate_reward = num_feasible_ports - 100  * num_infeasible_ports
            # Encode the next state for the given vessel
            vessel_simp = next_state['vessel_dict'][earliest_vessel['number']]
            encoded_next_state = self.encode_state(next_state, vessel_simp)
            # Use the target model to predict the future Q-values for the next state
            future_rewards = agent.target_model(torch.FloatTensor(encoded_next_state)).detach().numpy()
            # Select the maximum future Q-value as an indicator of the next state's potential
            max_future_reward = np.max(future_rewards)
            # Clip the future reward to be within the desired range, e.g., [-1, 1]
            clipped_future_reward = np.clip(max_future_reward, -10000, 10000)
            # Update the reward using the clipped future reward
            updated_reward = immediate_reward + clipped_future_reward
            
            if feasible_path:
                time = next_state['time']
                time_until_terminal = len(self.TIME_PERIOD_RANGE) - time
                # Add a reward for finding a feasible path and discount it based on the time until terminal using a discount factor
                discount_factor = agent.gamma ** time_until_terminal
                feasibility_reward = extra_reward_for_feasible_path*discount_factor
                updated_reward += feasibility_reward
                
            else:
                current_time = next_state['time']
                if first_infeasible_time is not None:
                    if current_time == first_infeasible_time:
                        # Apply the penalty to the updated_reward
                        updated_reward += penalty_for_infeasibility
                    
                        
                    # # Calculate the time difference from the first infeasible time
                    # time_diff = first_infeasible_time - current_time
                    
                    # if time_diff >= 0:
                    #     # Calculate a discount factor that heavily penalizes states closer to the first infeasible time
                    #     discount_factor = agent.sigma ** time_diff
                    #     infeasibility_penalty = penalty_for_infeasibility * discount_factor
                        
                    #     # Apply the penalty to the updated_reward
                    #     updated_reward += infeasibility_penalty
                    
            # updated_reward = immediate_reward + max_future_reward
            exp[3] = updated_reward
            exp[6] = feasible_path
            exp[7] = first_infeasible_time
        return experience_path


    def execute_action(self, vessel, action):
        # Action is on the form (vessel_id, operation_type, quantity, arc)
        # Execute the action and update the state
        _, operation_type, quantity, arc = action
        
        port = arc.origin_node.port
        # Update the vessel's inventory
        if operation_type == 1:
            #Loading
            vessel.inventory += quantity
            port.inventory -= quantity
        elif operation_type == 2:
            #Unloading
            vessel.inventory -= quantity
            port.inventory += quantity
        
        # ROUTING
        # Update the vessel's position and in_transit_towards attributes
        vessel.position = None
        vessel.in_transit_towards = (arc.destination_node.port, arc.destination_node.time)
        vessel.action_path.append(action)
        if arc.destination_node.port == self.SINK_NODE.port:
            vessel.isFinished = True
        
    def update_vessel_position(self, vessel, new_position):
        vessel.position = new_position
        vessel.in_transit_towards = None
        
    def sim_update_vessel_position(self, vessel, new_position):
        vessel['position'] = new_position
        vessel['in_transit_towards'] = None
        
        
        
    def sim_increment_time_and_produce(self, state):
        state['time'] += 1
        # 2. Ports consume/produce goods. Production happens before loading. Consumption happens after unloading.
        for port in state['port_dict'].values():
            if port['isLoadingPort'] == 1:
                port['inventory'] += port['rate']                
        return state
    
    def increment_time_and_produce(self, state):
        state['time'] += 1
        # 2. Ports consume/produce goods. Production happens before loading. Consumption happens after unloading.
        for port in state['ports']:
            if port.isLoadingPort == 1:
                port.inventory += port.rate
        return state
                
    def consumption(self, state):
        for port in state['ports']:
            if port.isLoadingPort == -1:
                port.inventory -= port.rate
        return state
    
    def sim_consumption(self, state):
        for port in state['port_dict'].values():
            if port['isLoadingPort'] == -1:
                port['inventory'] -= port['rate']
        return state
                
    def step(self, state, actions, experience_path, decision_basis_states):
        # We want to perform the actual vessel operations, but not consume or produce goods yet
        # Argument state is the actual state of the environment
        vessels_performing_actions = actions.keys()
        
        # Take a copy of the state here before doing the actions
        # old_state = copy.deepcopy(state)
        
        # Perform operations and update vessel positions for the actual state
        for vessel in vessels_performing_actions:
            self.execute_action(vessel=vessel, action=actions[vessel])
            
        # Take a copy of the state which contains the routing and the operations of the vessels performing actions. Will consume and produce goods in this state.
        simulation_state = self.custom_deep_copy_of_state(state)
        
        # Find the next time a vessel reaches a port
        earliest_time = 1000000
        earliest_vessel = None
        for vessel in simulation_state['vessel_dict'].values():
            # All vessels are now in transit
            # Find the earliest time a vessel reaches a port
            time_to_port = vessel['in_transit_towards']['destination_time']
            # time_to_port = vessel.in_transit_towards[1]
            if time_to_port < earliest_time:
                earliest_time = time_to_port
                earliest_vessel = vessel
        
        # Consume first in the current time period
        simulation_state = self.sim_consumption(simulation_state)
        
        while simulation_state['time'] < earliest_time:
            # Produce first
            simulation_state = self.sim_increment_time_and_produce(simulation_state)
            if simulation_state['time'] == earliest_time:
                # Only update vessel positions, do not consume
                self.sim_update_vessel_status(simulation_state)
            else:
                simulation_state = self.sim_consumption(simulation_state)
        
        # Save this experience if the destination port is not the sink
        for vessel, action in actions.items():
            destination_port_number = action[3].destination_node.port.number
            origin_port_number = action[3].origin_node.port.number
            # Save if the action is not to the sink or from the source
            if destination_port_number != len(simulation_state['port_dict']) and origin_port_number != 0:
                # old_state = state
                # action = actions[vessel]
                decision_basis_state = decision_basis_states[vessel.number]
                # Check infeasibility for simulation_state
                simulation_state['infeasible'] = self.sim_is_infeasible(simulation_state)
                feasible_path = None
                exp = [decision_basis_state, action, vessel, None, simulation_state, earliest_vessel, feasible_path, None]
                # exp_dict[vessel] = exp
                experience_path.append(exp)
        return state

    def find_available_vessels(self, state):
        available_vessels = []
        for v in state['vessels']:
            if v.position is not None:
                available_vessels.append(v)
        return available_vessels
    
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)  # First fully connected layer
        self.relu = nn.ReLU()                 # ReLU activation
        self.fc2 = nn.Linear(24, 24)          # Second fully connected layer
        self.fc3 = nn.Linear(24, action_size) # Output layer

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    

class ReplayMemory:
    def __init__(self, capacity, feasibility_priority=0.5):
        self.capacity = capacity  # Total capacity of the memory
        self.feasibility_priority = feasibility_priority  # Priority for feasible experiences
        self.memory = deque()  # Memory for infeasible experiences
        self.feasible_memory = deque()  # Memory for feasible experiences

    def push(self, event, env):
        state, action, vessel, reward, next_state, earliest_simp_vessel, is_feasible, first_infeasible_time = event
        
        vessel_simp = state['vessel_dict'][vessel.number]
        encoded_state = env.encode_state(state, vessel_simp)
        
        # Important action data is quantity and destination port
        vnum, operation_type, quantity, arc = action
        destination_port = arc.destination_node.port.number
        
        # Convert encoded_state to a tuple for hashing
        encoded_state_tuple = tuple(encoded_state.flatten())
        
        # Create a unique identifier for the experience
        exp_id = hash((encoded_state_tuple, quantity, destination_port))
        
        # Logic to handle feasible and infeasible experiences
        if is_feasible:
            self.feasible_memory = deque([(id, exp) for id, exp in self.feasible_memory if id != exp_id])
            self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
            self.feasible_memory.append((exp_id, event))
        else:
            if not any(id == exp_id for id, _ in self.feasible_memory):
                self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
                self.memory.append((exp_id, event))

    def sample(self, batch_size):
        num_feasible = int(batch_size * self.feasibility_priority)
        num_regular = batch_size - num_feasible
        
        samples = []
        if len(self.feasible_memory) < num_feasible:
            num_feasible = len(self.feasible_memory)
            num_regular = batch_size - num_feasible
        
        if num_feasible > 0:
            samples.extend(random.sample(self.feasible_memory, num_feasible))
        if num_regular > 0 and len(self.memory) > 0:
            samples.extend(random.sample(self.memory, num_regular))
        
        return samples

    def __len__(self):
        return len(self.memory) + len(self.feasible_memory)

    def clean_up(self):
        # Calculate the target size for each memory based on 75% of their respective capacities
        target_infeasible_size = int((self.capacity - int(self.capacity * self.feasibility_priority)) * 0.75)
        target_feasible_size = int((self.capacity * self.feasibility_priority) * 0.75)
        
        # Reduce the size of each memory to the target size
        while len(self.memory) > target_infeasible_size:
            self.memory.popleft()
        # while len(self.feasible_memory) > target_feasible_size:
        #     self.feasible_memory.popleft()

    

class DQNAgent:
    def __init__(self, ports, vessels, TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY, NON_RANDOM_ACTION_EPISODE_FREQUENCY, BATCH_SIZE, replay):
        # ports plus source and sink, vessel inventory, (vessel position, vessel in transit), time period, vessel_number
        # state_size = len(ports) + 3 * len(vessels) + 2 + len(ports)
        state_size = len(ports) + 3 * len(vessels) + 1 + len(ports)
        # Ports plus sink port
        action_size = len(ports)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = replay
        self.gamma = 0.99    # discount rate feasible paths
        self.sigma = 0.5     # discount rate infeasible paths
        self.epsilon = 0.75   # exploration rate
        self.epsilon_min = 0.25
        self.epsilon_decay = 0.99
        self.main_model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        # Load the state dictionaries
        self.optimizer = optim.Adam(self.main_model.parameters())
        self.TRAINING_FREQUENCY = TRAINING_FREQUENCY
        self.TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY
        self.NON_RANDOM_ACTION_EPISODE_FREQUENCY = NON_RANDOM_ACTION_EPISODE_FREQUENCY
        self.BATCH_SIZE = BATCH_SIZE
            
    def select_action(self, state, legal_actions,  env, vessel_simp, exploit):  
        
        if not exploit:
            if np.random.rand() < self.epsilon:
                action = random.choice(legal_actions)
                arc = action[3]
                # new_state = copy.deepcopy(state)
                new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                
                # Choose one of the legal actions at random
                return random.choice(legal_actions), new_state
            
        # If the only legal action is to travel to sink, do it
        if len(legal_actions) == 1:
            action = legal_actions[0]
            arc = action[3]
            # new_state = copy.deepcopy(state)
            new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
            return action, new_state
        
        # Encode state and add vessel number
        # vessel_simp= state['vessel_dict'][vessel.number]
        encoded_state = env.encode_state(state, vessel_simp)
        # print(encoded_state)
        # Convert the state to a tensor
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        # Get the q-values from the main model. One value for each destination port
        q_values = self.main_model(state_tensor).detach().numpy()
        
        # Make q_values into a list
        q_values = q_values[0]

        
        # Sort the q-values, but keep track of the original indices
        q_values = [(q_value, index +1) for index, q_value in enumerate(q_values)]
        q_values.sort(reverse=True)
        # Choose the action with the highest q-value that is legal
        for q_value, destination_port_number in q_values:
            for action in legal_actions:
                arc = action[3]
                if arc.destination_node.port.number == destination_port_number:
                    # We now have the action with the highest q-value that is legal
                    # Create a deep copy of the state and update the vessel's position and in_transit_towards attributes
                    # new_state = copy.deepcopy(state)
                    # new_state = env.update_vessel_in_transition_for_state(new_state, vessel, arc.destination_node.port, arc.destination_node.time)
                    new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                    
                    return action, new_state
                
                
    def train_main_network(self, env):
        if len(self.memory) < self.BATCH_SIZE:
            return  # Not enough samples to train
        
        total_loss = 0
        minibatch = self.memory.sample(self.BATCH_SIZE)
        for _, exp in minibatch:
            state, action, vessel, reward, next_state, _, _, _= exp
            vessel_simp = state['vessel_dict'][vessel.number]
            encoded_state = env.encode_state(state, vessel_simp)
            encoded_state = torch.FloatTensor(encoded_state).unsqueeze(0)
            
            _, _, _, arc = action
            destination_port = arc.destination_node.port
            action_idx = destination_port.number - 1
            
            # Reward is already adjusted in the experience path, so use it directly
            adjusted_reward = torch.FloatTensor([reward]).to(encoded_state.device)
            
            # Predicted Q-values for the current state
            current_q_values = self.main_model(encoded_state)
            
            # Extract the Q-value for the action taken. This time keeping it connected to the graph.
            correct_q = current_q_values.gather(1, torch.tensor([[action_idx]], dtype=torch.long).to(encoded_state.device)).squeeze()

            # Use the adjusted_reward directly as the target
            target_q = adjusted_reward
            target_q = target_q.squeeze()
            
            # Compute loss
            loss = F.mse_loss(correct_q, target_q)
            # Print the actual loss value
            total_loss += loss.item()
                        
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update epsilon
        # print(f"LossAvg: {total_loss/batch_size}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.main_model.state_dict())   
        
    def save_replay_buffer(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)
        print(f"Replay buffer saved to {file_name}.")
        
    def load_replay_buffer(self, file_name):
        with open(file_name, 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f"Replay buffer loaded from {file_name}.")
        return replay_buffer
    
def evaluate_agent_until_solution_is_found(env, agent, replay):
    solution_found = False
    while not solution_found:
    
        experience_path = []
        state = env.reset()
        done = False
        decision_basis_states = {vessel.number: state for vessel in state['vessels']}
        for vessel in state['vessels']:
            decision_basis_states[vessel.number] = env.custom_deep_copy_of_state(decision_basis_states[vessel.number])
        actions = {}
        for vessel in state['vessels']:
            legal_actions_for_vessel =  env.find_legal_actions_for_vessel(state=state, vessel=vessel)
            actions[vessel] = legal_actions_for_vessel[0]
        state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
        
        while not done:
            # Increase time and make production ports produce.
            state = env.increment_time_and_produce(state=state)
            # Check if state is infeasible or terminal        
            state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent)
            # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
            env.update_vessel_status(state=state)
            # Find the vessels that are available to perform an action
            available_vessels = env.find_available_vessels(state=state)
            # If some vessels are available, select actions for them
            if len(available_vessels) > 0:
                actions = {}
                decision_basis_states = {}
                decision_basis_state = env.custom_deep_copy_of_state(state)
                for vessel in available_vessels:
                    corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                    decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                    
                    legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel)
                    
                    action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=False)
                    actions[vessel] = action

                # Perform the operation and routing actions and update the state based on this
                state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
                
            # Make consumption ports consume (1 timeperiod worth of consume) regardless if any actions were performed
            state = env.consumption(state)
            
            # Check if state is infeasible or terminal
            state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent)
            
            # agent.update_policy(state, action, reward, next_state)
            done = state['done']
            
            if done:
                if not state['infeasible']:
                    return experience_path
                break
            
            
def evaluate_agent(env, agent):
    experience_path = []
    state = env.reset()
    done = False
    decision_basis_states = {vessel.number: state for vessel in state['vessels']}
    for vessel in state['vessels']:
        decision_basis_states[vessel.number] = env.custom_deep_copy_of_state(decision_basis_states[vessel.number])
    actions = {}
    for vessel in state['vessels']:
        legal_actions_for_vessel =  env.find_legal_actions_for_vessel(state=state, vessel=vessel)
        actions[vessel] = legal_actions_for_vessel[0]
    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
    
    while not done:
        # Increase time and make production ports produce.
        state = env.increment_time_and_produce(state=state)
        # Check if state is infeasible or terminal        
        state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
        # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
        env.update_vessel_status(state=state)
        # Find the vessels that are available to perform an action
        available_vessels = env.find_available_vessels(state=state)
        # If some vessels are available, select actions for them
        if len(available_vessels) > 0:
            actions = {}
            decision_basis_states = {}
            decision_basis_state = env.custom_deep_copy_of_state(state)
            for vessel in available_vessels:
                corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                
                legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel)
                
                action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=True)
                actions[vessel] = action

            # Perform the operation and routing actions and update the state based on this
            state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            
        # Make consumption ports consume (1 timeperiod worth of consume) regardless if any actions were performed
        state = env.consumption(state)
        
        # Check if state is infeasible or terminal
        state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
        
        # agent.update_policy(state, action, reward, next_state)
        done = state['done']
        
        if done:
            first_infeasible_time = env.log_episode(1, total_reward_for_path, experience_path)
            return first_infeasible_time
    

def convert_path_to_MIRPSO_solution(env, experience_path):
    # Create a dict with vesselnumber as key, and an empty list as value
    active_arcs = {}
    operational_nodes_for_vessel = {}
    
    for vessel in env.VESSELS:
        active_arcs[vessel.number] = []
        operational_nodes_for_vessel[vessel.number] = []
        
    for exp in experience_path:
        state, action, vessel, reward, next_state, earliest_vessel, feasible_path, first_infeasible_time = exp
        vessel_number, operation_type, quantity, arc = action
        active_arcs[vessel.number].append(arc)
        operation_node = arc.origin_node
        operational_nodes_for_vessel[vessel.number].append((operation_node, quantity))
    return active_arcs, operational_nodes_for_vessel

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
def train_from_pre_populated_buffer(env, agent, num_episodes):
    TRAINING_FREQUENCY = 1
    TARGET_UPDATE_FREQUENCY = 10
    EVALUATION_FREQUENCY = 1
    first_infeasible_times = []
    infeasibility_counters = []
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create another axis that shares the same x-axis
    
    for episode in range(1, num_episodes + 1):
        # Training the agent's network with a batch from the replay buffer.
        if episode % TRAINING_FREQUENCY == 0:
            agent.train_main_network(env)

        # Updating the target network
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
            print('Target network updated')
        
        if episode % EVALUATION_FREQUENCY == 0:
            first_inf_time, infeasibility_counter = evaluate_agent(env=env, agent=agent)
            if first_inf_time is None:
                first_inf_time = env.TIME_PERIOD_RANGE[-1]
            
            first_infeasible_times.append(first_inf_time)
            infeasibility_counters.append(infeasibility_counter)
            # Update live histogram plot
            # Clear the current plot's output and display the updated plot
            # Clear the current plot's output and display the updated plot
            clear_output(wait=True)  # Clear the previous output
            # plt.plot(first_infeasible_times, '-o', label='First Infeasible Time')
            ax1.plot(first_infeasible_times, '-ob', label='First Infeasible Time')
            ax2.plot(infeasibility_counters, '-or', label='Infeasibility Counter')
            if episode == EVALUATION_FREQUENCY:  # Add the legend only once
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
            # if episode == EVALUATION_FREQUENCY:  # Add the legend only once
                # plt.legend()
                
            ax1.set_title('Development of First Infeasible Times and Infeasibility Counters Over Episodes')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('First Infeasible Time', color='b')
            ax2.set_ylabel('Infeasibility Counter', color='r')
            
            display(plt.gcf())  # Display the current figure
            plt.pause(0.001)  # Pause for a short period to allow the plot to update

    plt.close()  # Close the figure to prevent additional output

    # plt.close()  # Close the figure to prevent additional output
            
def unpack_problem_data(problem_data):
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    non_operational = problem_data['non_operational']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    OPERATING_COST = problem_data['OPERATING_COST']
    OPERATING_SPEED = problem_data['OPERATING_SPEED']
    NODES = problem_data['NODES']
    NODE_DICT = problem_data['NODE_DICT']
    return vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES, NODE_DICT

def unpack_env_data(env_data):
    vessels = env_data['vessels']
    vessel_arcs = env_data['vessel_arcs']
    regularNodes = env_data['regularNodes']
    ports = env_data['ports']
    TIME_PERIOD_RANGE = env_data['TIME_PERIOD_RANGE']
    non_operational = env_data['non_operational']
    sourceNode = env_data['sourceNode']
    sinkNode = env_data['sinkNode']
    waiting_arcs = env_data['waiting_arcs']
    OPERATING_COST = env_data['OPERATING_COST']
    OPERATING_SPEED = env_data['OPERATING_SPEED']
    ports_dict = env_data['ports_dict']
    NODE_DICT = env_data['node_dict']
    vessel_dict = env_data['vessel_dict']
    return vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, ports_dict, NODE_DICT, vessel_dict


                
        
def main(FULLSIM):
    # Set a higher recursion limit (be cautious with this)
    sys.setrecursionlimit(5000) 
    random.seed(0)
    gc.enable()
    
    INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V9a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V11a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12a'
    # INSTANCE = 'LR1_2_DR1_3_VC2_V6a'
    # INSTANCE = 'LR1_2_DR1_3_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12b'
    # INSTANCE = 'LR2_11_DR2_22_VC3_V6a'
    # INSTANCE = 'LR2_11_DR2_33_VC4_V11a'
    
    TRAINING_FREQUENCY = 1
    TARGET_UPDATE_FREQUENCY = 100
    NON_RANDOM_ACTION_EPISODE_FREQUENCY = 25
    BATCH_SIZE = 256

    problem_data = build_problem(INSTANCE)
    vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES, NODE_DICT = unpack_problem_data(problem_data)
    simp_model, env_data = build_simplified_RL_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES, NODE_DICT)
    #Vessel arcs are the only thing that changes between the simplified and the full model
    vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, ports_dict, NODE_DICT, vessel_dict = unpack_env_data(env_data)
    
    env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode, NODE_DICT)
    replay = ReplayMemory(3000)
    agent = DQNAgent(ports = ports, vessels=vessels, TRAINING_FREQUENCY = TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY, NON_RANDOM_ACTION_EPISODE_FREQUENCY = NON_RANDOM_ACTION_EPISODE_FREQUENCY, BATCH_SIZE = BATCH_SIZE, replay = replay)
    # '''Load main and target model.'''
    # agent.main_model.load_state_dict(torch.load('main_model.pth'))
    # agent.target_model.load_state_dict(torch.load('target_model.pth'))
    
    if not FULLSIM:
        replay = agent.load_replay_buffer(file_name= 'replay_buffer.pkl')
        agent.memory = replay
        train_from_pre_populated_buffer(env, agent, 1000)
        
    else:
        NUM_EPISODES = 10000
        # profiler = cProfile.Profile()
        for episode in range(1, NUM_EPISODES):
            if episode % NON_RANDOM_ACTION_EPISODE_FREQUENCY == 0:
                exploit = True
                print(f"NON Random Action Episode: {episode}")
            else:
                exploit = False
            
            if episode % agent.TRAINING_FREQUENCY == 0:
                agent.train_main_network(env)
                
            if episode % agent.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print('Target network updated')
                gc.collect()
                replay.clean_up()
                        
            experience_path = []
                
            # Resetting the state of the environment
            state = env.reset()
            done = False
            # Directly create and fill decision_basis_states with custom deep-copied states for each vessel
            decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
            # We know that each vessel have only one legal action in the initial state
            actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
            state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            
            # All vessels have made their initial action.
            while not done:
                # Increase time and make production ports produce.
                state = env.increment_time_and_produce(state=state)
                
                # Check if state is infeasible or terminal        
                state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent)
                if state['done']:
                    if episode % NON_RANDOM_ACTION_EPISODE_FREQUENCY == 0:
                        env.log_episode(episode, total_reward_for_path, experience_path)
                    break
                
                # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
                env.update_vessel_status(state=state)
                # Find the vessels that are available to perform an action
                available_vessels = env.find_available_vessels(state=state)
            
                # If some vessels are available, select actions for them
                if available_vessels:
                    actions = {}
                    decision_basis_states = {}
                    decision_basis_state = env.custom_deep_copy_of_state(state)
                    for vessel in available_vessels:
                        corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                        decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                        legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel)
                        action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=exploit)
                        actions[vessel] = action
                    # Perform the operation and routing actions and update the state based on this
                    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
                    
                # Make consumption ports consume regardless if any actions were performed
                state = env.consumption(state)
            
        # agent.save_replay_buffer(file_name=f"replay_buffer_.pkl")
                
            # if episode % MODEL_SAVING_FREQUENCY == 0:
            #     torch.save(agent.main_model.state_dict(), 'main_model.pth')
            #     torch.save(agent.target_model.state_dict(), 'target_model.pth')
            
        # When agent is done training. Let the agent solve the problem, and return the solution
        experience_path = evaluate_agent(env, agent, replay)
        
    # Evaluate the agent one last time to get a solution
    experience_path = evaluate_agent_until_solution_is_found(env, agent, replay)
    active_arcs, operational_nodes_for_vessel = convert_path_to_MIRPSO_solution(experience_path)
    # Save the variables to a file
    
    
import sys

if __name__ == "__main__":
    FULL_SIM = True
    # FULL_SIM = False
    
    main(FULL_SIM)