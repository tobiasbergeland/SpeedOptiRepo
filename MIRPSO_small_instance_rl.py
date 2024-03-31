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
# from MIRPSO_M import (build_problem, build_simplified_RL_model, build_model, visualize_network_for_vessel, solve_model)
from MIRP_GROUP_2 import (build_problem, build_model, solve_model, rearrange_arcs)
from proximity_search import perform_proximity_search


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
            'time' : -1,
            'done' : False,
            'infeasible' : False,
        }
        
    def reset(self):
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
            'time' : -1,
            'done' : False,
            'infeasible' : False,
        }
        return self.state
   
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
                'capacity': vessel.capacity,
                'inventory': vessel.inventory,
                'number': vessel.number,
                'vessel_class': vessel.vessel_class,
                'isFinished': vessel.isFinished,
                
                #If position is None put None, else put the position
                'position': vessel.position.number if vessel.position is not None else None,
                'in_transit_towards': None if vessel.in_transit_towards is None else {
                    'destination_port_number': vessel.in_transit_towards[0].number,
                    'destination_time': vessel.in_transit_towards[1],
                }
            }
        return new_state
        
        
    def encode_state(self, state, vessel_simp):
        # Skip the sink port
        port_dict = {k: port for k, port in state['port_dict'].items() if port['capacity'] is not None}
        vessel_dict = state['vessel_dict']
        
        port_critical_times = np.array([
        math.floor(current_inventory / rate) if port['isLoadingPort'] != 1 else math.floor((port['capacity'] - current_inventory) / rate)
        for port in port_dict.values()
        for rate, current_inventory in [(port['rate'], port['inventory'])]
])

        # port_critical_times = np.array([
        #     max(0, math.floor(current_inventory / rate)) if port['isLoadingPort'] != 1 else max(0, math.floor((port['capacity'] - current_inventory) / rate))
        #     for port in port_dict.values()
        #     for rate, current_inventory in [(port['rate'], port['inventory'])]])
        vessel_inventories = np.array([vessel['inventory'] / vessel['capacity'] for vessel in vessel_dict.values()])
        current_vessel_number = np.array([vessel_simp['number']])
        current_vessel_class = np.array([vessel_simp['vessel_class']])
        inventory_effect_on_ports = np.array([(vessel_simp['capacity'] / port['rate']) for port in port_dict.values()])
        
        travel_times = np.array(self.TRAVEL_TIME_DICT[vessel_simp['position']])
        vessel_positions = np.array([v['position'] if v['position'] else v['in_transit_towards']['destination_port_number'] for v in vessel_dict.values()])
        vessel_in_transit = np.array([v['in_transit_towards']['destination_time'] - state['time'] if v['in_transit_towards'] else 0 for v in vessel_dict.values()])
        encoded_state = np.concatenate([port_critical_times, vessel_inventories, vessel_positions, vessel_in_transit, travel_times, current_vessel_number, current_vessel_class, inventory_effect_on_ports])
        return encoded_state
    
    def find_legal_actions_for_vessel(self, state, vessel):
        # Determine the operation type based on the vessel's position and the state's time
        legal_arcs = self.get_legal_arcs(state=state, vessel=vessel)
        
        # We now know that if we have a legal arc, the full load/unload is possible
        legal_actions = []
        for arc in legal_arcs:
            if arc.origin_node == self.SOURCE_NODE:
                operation_type = 0
                quantity = 0
                action = (vessel.number, operation_type, quantity, arc)
                legal_actions.append(action)
                return legal_actions
            if arc.is_waiting_arc:
                # Generate legal actions for waiting arcs
                operation_type = 0
                quantity = 0
                action = (vessel.number, operation_type, quantity, arc)
                legal_actions.append(action)
            elif arc.destination_node.port.isLoadingPort != 1:
                # We are at a loading port and loading the vessel
                operation_type = 1
                quantity = vessel.capacity
                action = (vessel.number, operation_type, quantity, arc)
                legal_actions.append(action)
            else:
                # We are at a discharging port and unloading the vessel
                operation_type = 2
                quantity = vessel.capacity
                action = (vessel.number, operation_type, quantity, arc)
                legal_actions.append(action)
                
        return legal_actions
    
    
    def sim_find_legal_actions_for_vessel(self, state, vessel):
        legal_arcs = self.sim_get_legal_arcs(state=state, vessel=vessel)
        if not legal_arcs:
            print('No legal arcs found')
        
        legal_actions = []
        for arc in legal_arcs:
            if arc.origin_node == self.SOURCE_NODE:
                operation_type = 0
                quantity = 0
                action = (vessel['number'], operation_type, quantity, arc)
                legal_actions.append(action)
                return legal_actions
            if arc.is_waiting_arc:
                # Generate legal actions for waiting arcs
                operation_type = 0
                quantity = 0
                action = (vessel['number'], operation_type, quantity, arc)
                legal_actions.append(action)
            elif arc.origin_node.port.isLoadingPort == 1:
                # We are at a loading port and loading the vessel
                operation_type = 1
                quantity = vessel['capacity']
                action = (vessel['number'], operation_type, quantity, arc)
                legal_actions.append(action)
            else:
                # We are at a discharging port and unloading the vessel
                operation_type = 2
                quantity = vessel['capacity']
                action = (vessel['number'], operation_type, quantity, arc)
                legal_actions.append(action)
        
        return legal_actions
    
    # def sim_find_legal_actions_for_vessel(self, state, vessel):
    #     # Initialize the operation type and vessel ID
    #     position_port_number = vessel['position']
    #     position_port = state['port_dict'][position_port_number]
    #     # Determine the operation type based on the vessel's position and the state's time
    #     operation_type = 0 if state['time'] == 0 else 1 if position_port['isLoadingPort'] == 1 else 2
    #     legal_quantity = self.sim_get_legal_quantities(operation_type=operation_type, vessel=vessel, position_port=position_port)
    #     legal_arcs = self.sim_get_legal_arcs(state=state, vessel=vessel, quantity=legal_quantity)
    #     legal_actions = [(vessel['number'], operation_type, legal_quantity, arc) for arc in legal_arcs]
    #     return legal_actions
        
        
    def sim_get_legal_arcs(self, state, vessel):
            current_node_key = (vessel['position'], state['time'])
            current_node = self.NODE_DICT[current_node_key]
            
            if current_node == self.SOURCE_NODE:
                return [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
            
            
            # Assuming TRAVEL_TIME_DICT is a dictionary where each value is a list of travel times
            travel_time_matrix = self.TRAVEL_TIME_DICT

            non_sim_vessel = self.VESSEL_DICT[vessel['number']]
            # Pre-filter arcs that originate from the current node
            potential_arcs = [arc for arc in self.VESSEL_ARCS[non_sim_vessel] if arc.origin_node == current_node]
            # Filter based on vessel state and destination port characteristics
            # All waiting arcs are legal
            # legal_arcs = [arc for arc in potential_arcs if arc.is_waiting_arc]
            legal_arcs = []
            port = state['port_dict'][vessel['position']]
            if port['isLoadingPort'] == 1:
                # Rate is added as to account for alpha slack
                # if port['inventory'] + port['rate'] >= vessel['capacity']:
                if port['inventory'] >= vessel['capacity']:
                    # Append arcs to all discharging ports to the legal arcs list
                    legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1
                                   or (arc.destination_node.port == self.SINK_NODE.port and state['time'] >= self.TIME_PERIOD_RANGE[-1] - self.TIME_PERIOD_RANGE[-1]/4)]
            else:
                # Rate is added as to account for alpha slack
                # if port['inventory'] + vessel['inventory'] <= port['capacity'] + port['rate']:
                if port['inventory'] + vessel['inventory'] <= port['capacity']:
                    # Append arcs to all loading ports to the legal arcs list
                    legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == 1
                                   or (arc.destination_node.port == self.SINK_NODE.port and state['time'] >= self.TIME_PERIOD_RANGE[-1] - self.TIME_PERIOD_RANGE[-1]/4)]
                    
            if not legal_arcs or state['time'] >= self.TIME_PERIOD_RANGE[-1] - self.TIME_PERIOD_RANGE[-1]/4:
                # Check first that a waiting arc is legal (only for loading ports)
                if port['isLoadingPort'] == 1:
                    if port['inventory'] <= port['capacity']:
                        legal_arcs += [arc for arc in potential_arcs if arc.is_waiting_arc]
                else:
                    legal_arcs += [arc for arc in potential_arcs if arc.is_waiting_arc]
                    
            if not legal_arcs:
                legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port == self.SINK_NODE.port]
                # Add sink arc if there are no other legal arcs
            
            # Check if there are multiple legal arcs that lead to the sink
            legal_arcs_to_sink = [arc for arc in legal_arcs if arc.destination_node.port == self.SINK_NODE.port]
            if len(legal_arcs_to_sink) > 1:
                print('Here')                
            return legal_arcs
        

    
    # def get_legal_arcs(self, state, vessel, quantity):
    #     # Find the node the vessel is currently at
    #     current_node_key = (vessel.position.number, state['time'])
    #     current_node = self.NODE_DICT[current_node_key]
    #     inventory_after_operation = vessel.inventory + quantity if current_node.port.isLoadingPort == 1 else vessel.inventory - quantity
    #     # Pre-filter arcs that originate from the current node
    #     potential_arcs = [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
    #     # Filter based on vessel state and destination port characteristics
    #     if inventory_after_operation == 0:
    #         # Vessel is empty, can only travel to loading ports or sink
    #         legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node == self.SINK_NODE}
    #     elif inventory_after_operation == vessel.max_inventory:
    #         # Vessel is full, can only travel to consumption ports or sink
    #         legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1 or arc.destination_node == self.SINK_NODE}
    #     else:
    #         # Vessel can travel anywhere except back to the same port
    #         legal_arcs = {arc for arc in potential_arcs if arc.destination_node.port != arc.origin_node.port}
    #     # Remove the sink arc if there are other legal arcs
    #     if len(legal_arcs) > 1:
    #         legal_arcs = {arc for arc in legal_arcs if arc.destination_node.port != self.SINK_NODE.port}
    #     return legal_arcs
    
    
    def get_legal_arcs(self, state, vessel):
        current_node = self.NODE_DICT[(vessel.position.number, state['time'])]
        
        if current_node == self.SOURCE_NODE:
            # Return the source arc
            return [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]

        potential_arcs = [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
        # All waiting arcs are legal
        legal_arcs = [arc for arc in potential_arcs if arc.is_waiting_arc]
        port = vessel.position
        if port.isLoadingPort == 1:
            # Rate is added as to account for alpha slack
            if port.inventory + port.rate >= vessel.capacity:
                # Append arcs to all discharging ports to the legal arcs list
                legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1]
        else:
            # Rate is added as to account for alpha slack
            if port.inventory + vessel.inventory <= port.capacity + port.rate:
                # Append arcs to all loading ports to the legal arcs list
                legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == 1]
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
                
                
    def update_vessel_status(self, state):
        time = state['time']
        for vessel in state['vessels']:
            destination_port = vessel.in_transit_towards[0]
            destination_time = vessel.in_transit_towards[1]
            if time == destination_time:
                # Vessel has reached its destination. Update the vessel's position
                self.update_vessel_position(vessel=vessel, new_position=destination_port)
                
    def sim_update_vessel_status(self, state):
        time = state['time']
        for vessel in state['vessel_dict'].values():
            destination_port = vessel['in_transit_towards']['destination_port_number']
            destination_time = vessel['in_transit_towards']['destination_time']
            if time == destination_time:
                # Vessel has reached its destination. Update the vessel's position
                self.sim_update_vessel_position(vessel=vessel, new_position=destination_port)
            if time == self.TIME_PERIOD_RANGE[-1]:
                vessel['isFinished'] = True
                
    def update_vessel_in_transition_and_inv_for_state(self, state, vessel, destination_port, destination_time, origin_port, quantity, operation_type):
        v = state['vessel_dict'][vessel['number']]
        in_transit_towards = {
            'destination_port_number': destination_port.number,
            'destination_time': destination_time}
        v['in_transit_towards'] = in_transit_towards
        v['position'] = None
        origin_port = state['port_dict'][origin_port.number]
        destination_port = state['port_dict'][destination_port.number]
        if destination_port == self.SINK_NODE.port:
            v['isFinished'] = True
        
        if operation_type == 1:
            origin_port['inventory'] -= quantity
            v['inventory'] += quantity
        else:
            origin_port['inventory'] += quantity
            v['inventory'] -= quantity
        return state
        
    def is_infeasible(self, state):
        for port in state['ports']:
            if port.isLoadingPort == 1:
                # Rate is added as to account for alpha slack
                if port.inventory < 0 or port.inventory - port.rate > port.capacity:
                    return True
            else:
                # Rate is added as to account for alpha slack
                if port.inventory + port.rate < 0 or port.inventory > port.capacity:
                    return True
        return False
    
    def sim_is_infeasible(self, state):
        infeasible_state = False
        infeasible_ports = []
        
        for port in state['port_dict'].values():
            if port['capacity'] is not None:
                if port['isLoadingPort'] == 1:
                    # Rate is added as to account for alpha slack
                    if port['inventory'] < 0 or port['inventory'] - port['rate'] > port['capacity']:
                        infeasible_state = True
                        infeasible_ports.append(port)
                else:
                    # Rate is added as to account for alpha slack
                    if port['inventory'] + port['rate'] < 0 or port['inventory'] > port['capacity']:
                        infeasible_state = True
                        infeasible_ports.append(port)
        return infeasible_state, infeasible_ports
                

    def is_terminal(self, state):
        if state['time'] == len(self.TIME_PERIOD_RANGE):
            return True
        
    def check_state(self, state, experience_path, replay, agent):
        '''Evaluates the state and returns status and reward.'''
        total_reward_for_path  = 0
        if self.is_terminal(state):
            experience_path = self.update_rewards_in_experience_path(experience_path, agent)
            cum_q_vals_main_net = 0
            cum_q_vals_target_net = 0
            for exp in experience_path:
                action = exp[1]
                if action is not None:
                    arc = action[3]
                    
                    # Use the main model to predict the Q-values for the current state
                    vessel_simp = exp[0]['vessel_dict'][exp[2].number]
                    encoded_state = self.encode_state(exp[0], vessel_simp)
                    
                    # Select the Q-value corresponding to the executed action
                    cum_q_vals_main_net += np.sum(agent.main_model(torch.FloatTensor(encoded_state)).detach().numpy())
                    cum_q_vals_target_net += np.sum(agent.target_model(torch.FloatTensor(encoded_state)).detach().numpy())
                    
                    destination_port = arc.destination_node.port.number
                    if destination_port == self.SINK_NODE.port.number:
                        continue
                    replay.push(exp, self)
                    
                rew = exp[3]
                total_reward_for_path += rew
            state['done'] = True
            state['infeasible'] = self.is_infeasible(state=state)
            return state, total_reward_for_path, cum_q_vals_main_net, cum_q_vals_target_net
        return state, None, None, None
    
    def log_episode(self, episode, total_reward_for_path, experience_path, state, cum_q_vals_main_net, cum_q_vals_target_net):
        infeasibility_counter = 0
        infeasibility_dict = {}
        for exp in experience_path:
            current_state = exp[0]
            # result_state = exp[4]
            time = current_state['time']
            if current_state['infeasible']:
                if time not in infeasibility_dict.keys():
                    infeasibility_dict[time] = True
                    
        # Check also the final state
        if state['infeasible']:
            infeasibility_dict[len(self.TIME_PERIOD_RANGE)] = True
            
        infeasibility_counter = len(infeasibility_dict.keys())
        print(f"Episode {episode}: Total Reward = {total_reward_for_path}\nMain Net Cumulative Q-Vals = {cum_q_vals_main_net}\nTarget Net Cumulative Q-Vals = {cum_q_vals_target_net}\nInfeasibility Counter = {infeasibility_counter}")
        if infeasibility_counter > 0:
            # Sort the dict by time
            infeasibility_dict = dict(sorted(infeasibility_dict.items()))
            print('Infeasible time periods:', infeasibility_dict.keys())
        print('-----------------------------------')
        first_infeasible_time = experience_path[0][7]
        return first_infeasible_time, infeasibility_counter
    
    def apply_penalty(self, exp, feasible_path, first_infeasible_time, reward):
        exp[3] = reward
        exp[6] = feasible_path
        exp[7] = first_infeasible_time
        
        
    def update_rewards_in_experience_path(self, experience_path, agent):
        feasible_path = True
        first_infeasible_time = None
        horizon = len(self.TIME_PERIOD_RANGE)
        
        # most_recent_visits_for_ports = {port.number : 0 for port in self.PORTS}
        for exp in experience_path:
            current_state, action, _, _, next_state, _, _, fi_time, terminal_flag = exp
            current_state_is_infeasible, infeasible_ports = self.sim_is_infeasible(current_state)
            if current_state_is_infeasible:
                feasible_path = False
                first_infeasible_time = current_state['time']
                break  # Exit the loop early if any infeasible state is found
            
            # if action is not None:
            #     arc = action[3]
            #     destination_port_number = arc.destination_node.port.number
            #     visit_time = arc.destination_node.time
            #     most_recent_visits_for_ports[destination_port_number] = visit_time
                        

        # For port in infeasible ports, find all actions that could have led to the infeasible port but was avoided
        
        # for inf_port in infeasible_ports:
        #     for exp in experience_path:
        #         current_state, action, _, _, _, _, _, _, _ = exp
        #         if current_state['time'] < first_infeasible_time:
        #             if action is not None:
        #                 arc = action[3]
        #                 origin_port = arc.origin_node.port
        #                 travel_time = self.TRAVEL_TIME_DICT[origin_port.number][inf_port['number'] - 1]
                        
        #                 if inf_port['isLoadingPort'] == 1:
        #                     # if arc is going from a discharging port and not to the infeasible port
        #                     if origin_port.isLoadingPort != 1 and arc.destination_node.port.number != inf_port['number'] and current_state['time'] == first_infeasible_time - travel_time:
        #                         # find the travel time from the origin port to the infeasible port
        #                         self.apply_penalty(exp, feasible_path, first_infeasible_time, -1)
                                   
        #                     # If a waiting arc was taken in the previous time step and the vessel is at the infeasible port, set that reward to 0
        #                     elif current_state['time'] == first_infeasible_time - 1 and arc.is_waiting_arc and origin_port.number == inf_port['number']:
        #                         self.apply_penalty(exp, feasible_path, first_infeasible_time, -1)
        #                 else:
        #                     # inf port is a discharging port
        #                     if origin_port.isLoadingPort == 1 and arc.destination_node.port.number != inf_port['number'] and current_state['time'] == first_infeasible_time - travel_time:
        #                         self.apply_penalty(exp, feasible_path, first_infeasible_time, -1)
        # '''Now we have set the reward to 0 for all states that could potentially have avoided the infeasible port with better actions.
        #     We will check if the reward is set to 0 in the part below.'''                       
            
        extra_reward_for_feasible_path = horizon
        for exp in experience_path:
            current_state, action, vessel, reward_ph, next_state, earliest_vessel, feasible_path_ph, fi_time, terminal_flag = exp
            cs_time = current_state['time']
            
            reward = reward_ph if reward_ph is not None else 0
            
            # Immediate reward 
            # current_state_is_infeasible, _ = self.sim_is_infeasible(current_state)
            if not current_state['infeasible']:
                reward += 1
            else:
                reward -= horizon - cs_time
            # reward += 1 if not current_state['infeasible'] else -1
                
            # Future reward for the next state if the next state is not terminal
            if not terminal_flag and action is not None:
                vessel_simp = next_state['vessel_dict'][earliest_vessel['number']]
                encoded_next_state = self.encode_state(next_state, vessel_simp)
                # Use the target model to predict the future Q-values for the next state
                future_rewards = agent.target_model(torch.FloatTensor(encoded_next_state)).detach().numpy()
                # Select the maximum future Q-value as an indicator of the next state's potential
                max_future_reward = np.max(future_rewards)
                # Clip the future reward to be within the desired range, e.g., [-1, 1]
                max_future_reward = np.clip(max_future_reward, -100000, 100000)
                # Update the reward using the clipped future reward
                reward += max_future_reward
            
            # If the whole path is feasible, add a huge reward discounted reward
            if feasible_path:
                current_time = current_state['time']
                time_until_terminal = horizon - current_time
                feasibility_reward = extra_reward_for_feasible_path * agent.gamma ** time_until_terminal
                reward += feasibility_reward
                    
            self.apply_penalty(exp, feasible_path, first_infeasible_time, reward)
            
        if feasible_path:
            print('Feasible path')
        return experience_path
    
    def sim_all_vessels_finished(self, state):
        for vessel in state['vessel_dict'].values():
            if not vessel['isFinished']:
                return False
        return True

    def all_vessels_finished(self, state):
        vessels = state['vessels']
        return all(v.isFinished for v in vessels)

    def execute_action(self, vessel, action):
        # Action is on the form (vessel_id, operation_type, quantity, arc)
        # Execute the action and update the state
        _, operation_type, quantity, arc = action
        port = arc.origin_node.port
        if operation_type == 1:
            #Loading
            vessel.inventory += quantity
            port.inventory -= quantity
        elif operation_type == 2:
            #Unloading
            vessel.inventory -= quantity
            port.inventory += quantity
        
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
        vessels_performing_actions = actions.keys()
        
        # Perform operations and update vessel positions for the actual state
        for vessel in vessels_performing_actions:
            self.execute_action(vessel=vessel, action=actions[vessel])
            
            
        # Take a copy of the state which contains the routing and the operations of the vessels performing actions. Will consume and produce goods in this state.
        simulation_state = self.custom_deep_copy_of_state(state)
        
        # Find the next time a vessel reaches a port
        earliest_time = math.inf
        earliest_vessel = None
        for vessel in simulation_state['vessel_dict'].values():
            time_to_port = vessel['in_transit_towards']['destination_time']
            if time_to_port < earliest_time:
                earliest_time = time_to_port
                earliest_vessel = vessel
        
        # Consume first in the current time period
        simulation_state = self.sim_consumption(simulation_state)
        
        while simulation_state['time'] < earliest_time:
            simulation_state = self.sim_increment_time_and_produce(simulation_state)
            if simulation_state['time'] == earliest_time:
                # Only update vessel positions, do not consume
                self.sim_update_vessel_status(simulation_state)
            else:
                simulation_state = self.sim_consumption(simulation_state)
        
        # Save this experience
        for vessel, action in actions.items():
            origin_port_number = action[3].origin_node.port.number
            # Save if the action is not to the sink or from the source
            if origin_port_number != self.SOURCE_NODE.port.number:
                decision_basis_state = decision_basis_states[vessel.number]
                db_state_is_infeasible, _ = self.sim_is_infeasible(decision_basis_state)
                decision_basis_state['infeasible'] = db_state_is_infeasible
                sim_state_is_infeasible, _ = self.sim_is_infeasible(simulation_state)
                simulation_state['infeasible'] = sim_state_is_infeasible
                
                all_vessels_finished = self.sim_all_vessels_finished(simulation_state)
                terminal_flag = all_vessels_finished or simulation_state['time'] == len(self.TIME_PERIOD_RANGE)
                feasible_path = None
                exp = [decision_basis_state, action, vessel, None, simulation_state, earliest_vessel, feasible_path, None, terminal_flag]
                experience_path.append(exp)
        return state
    
    def simple_step(self, state, experience_path):
        original_state = self.custom_deep_copy_of_state(state)
        consumed_state = self.sim_consumption(original_state)
        consumed_state_is_infeasible, _ = self.sim_is_infeasible(consumed_state)
        consumed_state['infeasible'] = consumed_state_is_infeasible
        terminal_flag = self.sim_all_vessels_finished(consumed_state) or consumed_state['time'] == len(self.TIME_PERIOD_RANGE)
        exp = [original_state, None, None, None, consumed_state, None, None, None, terminal_flag]
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
        self.fc1 = nn.Linear(state_size, 24)     # First fully connected layer
        self.fc2 = nn.Linear(24, 24)             # Second fully connected layer
        self.fc3 = nn.Linear(24, 24)             # Third fully connected layer, newly added
        self.fc4 = nn.Linear(24, action_size)    # Output layer
        self.relu = nn.ReLU()                    # ReLU activation

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))               # Activation for the newly added layer
        return self.fc4(x)
    
class ReplayMemory:
    def __init__(self, capacity, feasibility_priority=0.5):
        self.capacity = capacity  # Total capacity of the memory
        self.feasibility_priority = feasibility_priority  # Priority for feasible experiences
        self.memory = deque()  # Memory for infeasible experiences
        self.feasible_memory = deque()  # Memory for feasible experiences

    def push(self, exp, env):
        state, action, vessel, reward, next_state, earliest_simp_vessel, is_feasible, first_infeasible_time, terminal_flag = exp
        if first_infeasible_time:
            if state['time'] > first_infeasible_time:
                return # Do not add experiences that occur after the first infeasible time
        vessel_simp = state['vessel_dict'][vessel.number]
        encoded_state = env.encode_state(state, vessel_simp)
        
        # Important action data is quantity and destination port
        vnum, operation_type, quantity, arc = action
        origin_port_number = arc.origin_node.port.number
        destination_port = arc.destination_node.port.number
        
        # Convert encoded_state to a tuple for hashing
        encoded_state_tuple = tuple(encoded_state.flatten())
        # Create a unique identifier for the experience
        exp_id = hash((encoded_state_tuple, operation_type, origin_port_number, destination_port))
        
        if is_feasible:
            # Delete existing similar experience in both feasible and infeasible memory if it exists
            self.feasible_memory = deque([(id, exp) for id, exp in self.feasible_memory if id != exp_id])
            self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
            self.feasible_memory.append((exp_id, exp))
        else:
            # Check if the experience is already in the feasible memory
            if not any(id == exp_id for id, _ in self.feasible_memory):
                # If not, add it to the infeasible memory. Substitute if similar exp already exists in infeasible memory
                self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
                self.memory.append((exp_id, exp))
        
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
        state_size = len(ports) + 3 * len(vessels) + 1 + len(ports)*2 + 1
        action_size = len(ports) + 1
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
                new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                return action, new_state
            
        # If there is only one legal action, choose it
        if len(legal_actions) == 1:
            action = legal_actions[0]
            arc = action[3]
            # new_state = copy.deepcopy(state)
            new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
            return action, new_state
        
        # Encode state and add vessel number
        encoded_state = env.encode_state(state, vessel_simp)
        # Convert the state to a tensor
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.main_model(state_tensor).detach().numpy()
        q_values = q_values[0]
        # Sort the q-values, but keep track of the original indices
        q_values = [(q_value, index +1) for index, q_value in enumerate(q_values)]
        q_values.sort(reverse=True)
        a = 10
        # Choose the action with the highest q-value that is legal
        for q_value, destination_port_number in q_values:
            for action in legal_actions:
                arc = action[3]
                if arc.destination_node.port.number == destination_port_number:
                    new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                    return action, new_state
                
                
    def train_main_network(self, env):
        if len(self.memory) < self.BATCH_SIZE:
            return  # Not enough samples to train
        
        total_loss = 0
        minibatch = self.memory.sample(self.BATCH_SIZE)
        for _, exp in minibatch:
            state, action, vessel, reward, next_state, _, _, _, terminal_flag= exp
            vessel_simp = state['vessel_dict'][vessel.number]
            encoded_state = env.encode_state(state, vessel_simp)
            encoded_state = torch.FloatTensor(encoded_state).unsqueeze(0)
            _, _, _, arc = action
            destination_port = arc.destination_node.port
            if destination_port.number == env.SINK_NODE.port.number:
                continue
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
    
def evaluate_agent_until_solution_is_found(env, agent):
    agent.epsilon = 0.1
    attempts = 0
    while True:
        experience_path = []
        state = env.reset()
        done = False
        port_inventory_dict = {}
        vessel_inventory_dict = {}
        decision_basis_states = {vessel.number: state for vessel in state['vessels']}
        for vessel in state['vessels']:
            decision_basis_states[vessel.number] = env.custom_deep_copy_of_state(decision_basis_states[vessel.number])
        actions = {}
        for vessel in state['vessels']:
            legal_actions_for_vessel =  env.find_legal_actions_for_vessel(state=state, vessel=vessel)
            actions[vessel] = legal_actions_for_vessel[0]
        state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
        
        # Init port inventory is the inventory at this time
        port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
        # Init vessel inventory is the inventory at this time
        vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
        
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
                    action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=False)
                    actions[vessel] = action

                # Perform the operation and routing actions and update the state based on this
                state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
                
            # Make consumption ports consume (1 timeperiod worth of consume) regardless if any actions were performed
            state = env.consumption(state)
            
            # Save the inventory levels for the ports and vessels at this time
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            # Check if state is infeasible or terminal
            state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
            # agent.update_policy(state, action, reward, next_state)
            done = state['done']
            if done:
                feasible_path = experience_path[0][6]
                if feasible_path:
                    first_infeasible_time, infeasibility_counter = env.log_episode(1, total_reward_for_path, experience_path)
                    return first_infeasible_time,infeasibility_counter, experience_path, port_inventory_dict, vessel_inventory_dict
                else:
                    attempts += 1
                    print(f"Attempt {attempts} failed. Retrying...")
                break
            
            
def evaluate_agent(env, agent):
    experience_path = []
    state = env.reset()
    done = False
    
    # Create a dictionary with time as key and a list with inventory levels for each port as value
    
    port_inventory_dict = {}
    vessel_inventory_dict = {}
    decision_basis_states = {vessel.number: state for vessel in state['vessels']}
    for vessel in state['vessels']:
        decision_basis_states[vessel.number] = env.custom_deep_copy_of_state(decision_basis_states[vessel.number])
    actions = {}
    for vessel in state['vessels']:
        legal_actions_for_vessel =  env.find_legal_actions_for_vessel(state=state, vessel=vessel)
        actions[vessel] = legal_actions_for_vessel[0]
    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
    
    # Init port inventory is the inventory at this time
    port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
    # Init vessel inventory is the inventory at this time
    vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
    
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
        
        # Save the inventory levels for the ports and vessels at this time
        port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
        vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
        
        # Check if state is infeasible or terminal
        state, total_reward_for_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
        
        # agent.update_policy(state, action, reward, next_state)
        done = state['done']
        
        if done:
            first_infeasible_time, infeasibility_counter = env.log_episode(1, total_reward_for_path, experience_path)
            return first_infeasible_time,infeasibility_counter, experience_path, port_inventory_dict, vessel_inventory_dict
    

def convert_path_to_MIRPSO_solution(env, experience_path, port_inventory_dict, vessel_inventory_dict):
    # Create a dict with vesselnumber as key, and an empty list as value
    active_arcs = {vessel.number: [] for vessel in env.VESSELS}
    
    #add source arcs at the beginning of the each vessel's active arcs
    for vessel in env.VESSELS:
        # Find the source arc for the vessel
        arcs = env.VESSEL_ARCS[vessel]
        for arc in arcs:
            if arc.origin_node == env.SOURCE_NODE and arc.destination_node != env.SINK_NODE:
                source_arc = arc
                active_arcs[vessel.number].append(source_arc)
                break
        
    active_O_and_Q = {}
    for exp in experience_path:
        state, action, vessel, reward, next_state, earliest_vessel, feasible_path, first_infeasible_time = exp
        vessel_number, operation_type, quantity, arc = action
        active_arcs[vessel.number].append(arc)
        operation_node = arc.origin_node
        active_O_and_Q[(operation_node.port.number, operation_node.time, vessel)] = quantity
        
    active_X_keys = []
    for vessel in env.VESSELS:
        for arc in active_arcs[vessel.number]:
            active_X_keys.append(((arc.origin_node, arc.destination_node), vessel))
            
    S_values = {}
    for time, invs_at_time in port_inventory_dict.items():
        for port in env.PORTS:
            S_values[(port.number, time)] = invs_at_time[port.number]
            
    W_values = {}
    for time, invs_at_time in vessel_inventory_dict.items():
        for vessel in env.VESSELS:
            W_values[(time, vessel)] = invs_at_time[vessel.number]
        
    return active_O_and_Q, active_X_keys, S_values, W_values

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
            
            first_inf_time,infeasibility_counter, _, _, _ = evaluate_agent(env=env, agent=agent)
            
            if first_inf_time is None:
                first_inf_time = env.TIME_PERIOD_RANGE[-1]
            
            first_infeasible_times.append(first_inf_time)
            infeasibility_counters.append(infeasibility_counter)
            clear_output(wait=True)  # Clear the previous output
            ax1.plot(first_infeasible_times, '-ob', label='First Infeasible Time')
            ax2.plot(infeasibility_counters, '-or', label='Infeasibility Counter')
            if episode == EVALUATION_FREQUENCY:  # Add the legend only once
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
            ax1.set_title('Development of First Infeasible Times and Infeasibility Counters Over Episodes')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('First Infeasible Time', color='b')
            ax2.set_ylabel('Infeasibility Counter', color='r')
            display(plt.gcf())  # Display the current figure
            plt.pause(0.001)  # Pause for a short period to allow the plot to update
    plt.close()  # Close the figure to prevent additional output
            
def unpack_problem_data(problem_data):
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    arc_dict = problem_data['arc_dict']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    NODES = problem_data['NODES']
    NODE_DICT = problem_data['NODE_DICT']
    VESSEL_CLASSES = problem_data['VESSEL_CLASSES']
    vessel_class_capacities = problem_data['vessel_class_capacities']
    return vessels, vessel_arcs, arc_dict, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, waiting_arcs, NODES, NODE_DICT, VESSEL_CLASSES, vessel_class_capacities

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



def warm_start_model(m, active_O_and_Q, active_X_keys, S_values, W_values):
    # Initialize all 'x', 'o', and 'q' variables to 0 to ensure a complete warm start
    for var in m.getVars():
        if var.VarName.startswith('x') or var.VarName.startswith('o') or var.VarName.startswith('q'):
            var.Start = 0  # Default start value for all variables not explicitly set
    
    # Setting initial values for 'o' and 'q' variables based on active_O_and_Q
    for (port_number, time, vessel), q_value in active_O_and_Q.items():
        o_var_name = f"o[{port_number},{time},{vessel}]"
        q_var_name = f"q[{port_number},{time},{vessel}]"
        
        o_var = m.getVarByName(o_var_name)
        q_var = m.getVarByName(q_var_name)
        
        if o_var is not None:
            o_var.Start = 1
        if q_var is not None:
            q_var.Start = q_value
            
    # Setting initial values for 'x' variables based on active_X_keys
    for (arc_tuple, vessel) in active_X_keys:
        x_var_name = f"x[{arc_tuple},{vessel}]"
        x_var = m.getVarByName(x_var_name)
        if x_var is not None:
            x_var.Start = 1
    
    # For 's' and 'w' variables, since you believe all values are set already, we maintain your original logic
    for (port_number, time), s_value in S_values.items():
        s_var_name = f"s[{port_number},{time}]"
        s_var = m.getVarByName(s_var_name)
        if s_var is not None:
            s_var.Start = s_value
            
    for (time, vessel), w_value in W_values.items():
        w_var_name = f"w[{time},{vessel}]"
        w_var = m.getVarByName(w_var_name)
        if w_var is not None:
            w_var.Start = w_value
    
    # Finally, update the model to apply these start values
    m.update()
    
    # Optionally, print start values for verification or debugging
    for var in m.getVars():
        if var.VarName.startswith('q') and var.Start <= 300:  # Adjusted condition for clarity
            print(f"{var.VarName}: {var.Start}")
    
    x_solution = {v.VarName: v.Start for v in m.getVars() if v.VarName.startswith('x')}

    return x_solution, m


def main(FULLSIM):
    # Set a higher recursion limit (be cautious with this)
    sys.setrecursionlimit(5000) 
    random.seed(0)
    gc.enable()
    
    # INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V9a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V11a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12a'
    # INSTANCE = 'LR1_2_DR1_3_VC2_V6a'
    # INSTANCE = 'LR1_2_DR1_3_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12b'
    # INSTANCE = 'LR2_11_DR2_22_VC3_V6a'
    # INSTANCE = 'LR2_11_DR2_33_VC4_V11a'
    
    INSTANCE = 'LR1_DR02_VC01_V6a'
    # INSTANCE = 'LR1_DR02_VC02_V6a'
    
    TRAINING_FREQUENCY = 1
    TARGET_UPDATE_FREQUENCY = 300
    NON_RANDOM_ACTION_EPISODE_FREQUENCY = 25
    BATCH_SIZE = 256
    BUFFER_SAVING_FREQUENCY = 1000

    problem_data = build_problem(INSTANCE)
    
    vessels, vessel_arcs, arc_dict, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, waiting_arcs, NODES, NODE_DICT, VESSEL_CLASSES, vessel_class_capacities = unpack_problem_data(problem_data)
    
    origin_node_arcs, destination_node_arcs, vessel_class_arcs = rearrange_arcs(arc_dict=arc_dict)
    
    # simp_model, env_data = build_simplified_RL_model(vessels, all_vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES, NODE_DICT)
    #Vessel arcs are the only thing that changes between the simplified and the full model
    # vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, ports_dict, NODE_DICT, vessel_dict = unpack_env_data(env_data)
    
    model = build_model(vessels = vessels,
                        regularNodes=regularNodes,
                        ports = ports,
                        TIME_PERIOD_RANGE = TIME_PERIOD_RANGE,
                        sourceNode = sourceNode,
                        sinkNode = sinkNode,
                        vessel_classes = VESSEL_CLASSES,
                        origin_node_arcs = origin_node_arcs,
                        destination_node_arcs = destination_node_arcs,
                        vessel_class_arcs = vessel_class_arcs,
                        NODE_DICT = NODE_DICT,
                        vessel_class_capacities = vessel_class_capacities)
    
    env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode, NODE_DICT)
    replay = ReplayMemory(3000)
    agent = DQNAgent(ports = ports, vessels=vessels, TRAINING_FREQUENCY = TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY, NON_RANDOM_ACTION_EPISODE_FREQUENCY = NON_RANDOM_ACTION_EPISODE_FREQUENCY, BATCH_SIZE = BATCH_SIZE, replay = replay)
    
    # '''Load main and target model.'''
    # agent.main_model.load_state_dict(torch.load('main_model.pth'))
    # agent.target_model.load_state_dict(torch.load('target_model.pth'))
    
    if not FULLSIM:
        replay = agent.load_replay_buffer(file_name= 'replay_buffer_new_reward_policy_7000.pkl')
        agent.memory = replay
        
        agent.main_model.load_state_dict(torch.load('main_model_7000.pth'))
        agent.target_model.load_state_dict(torch.load('target_model_7000.pth'))
        # train_from_pre_populated_buffer(env, agent, 1000)
        
    else:
        NUM_EPISODES = 10001
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
                if episode < 5000:
                    # After 3000 episodes, the target net is good, so keep all the experiences in the replay buffer
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
                if state['time'] in TIME_PERIOD_RANGE:
                    state = env.increment_time_and_produce(state=state)
                else:
                    #Only increment the time
                    state['time'] += 1
                    
                
                # Check if state is infeasible or terminal        
                state, total_reward_for_path, cum_q_vals_main_net, cum_q_vals_target_net = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent)
                if state['done'] or state['infeasible']:
                    if episode % NON_RANDOM_ACTION_EPISODE_FREQUENCY == 0:
                        env.log_episode(episode, total_reward_for_path, experience_path, state, cum_q_vals_main_net, cum_q_vals_target_net)
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
                        _, _, _, _arc = action
                        actions[vessel] = action
                    # Perform the operation and routing actions and update the state based on this
                    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
                else:
                    # Should check the feasibility of the state, even though no actions were performed. 
                    state = env.simple_step(state, experience_path)
                # Make consumption ports consume regardless if any actions were performed
                state = env.consumption(state)
                
                
                
            
            if episode % BUFFER_SAVING_FREQUENCY == 0:
                agent.save_replay_buffer(file_name=f"replay_buffer_new_reward_policy_{episode}.pkl")
                torch.save(agent.main_model.state_dict(), f'main_model_{episode}.pth')
                torch.save(agent.target_model.state_dict(), f'target_model_{episode}.pth')
            
    # When agent is done training. Let the agent solve the problem, and return the solution
    # first_infeasible_time, infeasibility_counter, experience_path, port_inventory_dict, vessel_inventory_dict  = evaluate_agent(env, agent)
    first_infeasible_time, infeasibility_counter, experience_path, port_inventory_dict, vessel_inventory_dict  = evaluate_agent_until_solution_is_found(env, agent)
    active_O_and_Q, active_X_keys, S_values, W_values = convert_path_to_MIRPSO_solution(env, experience_path, port_inventory_dict, vessel_inventory_dict)
    env.reset()
    main_model, costs = build_model(vessels, all_vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST)
    x_initial_solution, main_model = warm_start_model(main_model, active_O_and_Q, active_X_keys, S_values, W_values)

    #print the initial solution
    #print("Initial solution:", x_initial_solution)

    ps_data = {'model': main_model, 'initial_solution':x_initial_solution,'costs': costs, 'regularNodes': regularNodes, 'vessels': vessels, 'operating_cost':OPERATING_COST, 'vessel_arcs': vessel_arcs}

    # Perform the proximity search using the initial solution
    improved_solution, obj_value = perform_proximity_search(ps_data)
    #Perform proximity search with simulated annealing
    #improved_solution, obj_value = perform_proximity_search_with_simulated_annealing(ps_data)
    # print("Final solution:", improved_solution)
    print("Objective value:", obj_value)
    #solve_model(main_model)
    
    
    
import sys

if __name__ == "__main__":
    FULL_SIM = True
    # FULL_SIM = False
    
    main(FULL_SIM)