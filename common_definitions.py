import math
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import copy


class MIRPSOEnv():
    def __init__(self, PORTS, VESSELS, VESSEL_ARCS, NODES, TIME_PERIOD_RANGE, SOURCE_NODE, SINK_NODE, NODE_DICT, special_sink_arcs, special_nodes_dict):
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
        self.SPECIAL_SINK_ARCS = special_sink_arcs
        self.SPECIAL_NODES_DICT = special_nodes_dict
        self.TRAVEL_TIME_DICT = {}
        self.current_checkpoint = 0
        self.inf_counter_updates = 0
        self.current_best_IC = TIME_PERIOD_RANGE[-1]
            
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
            # If random initial start, set it here
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
                'rate': port.rate,
                'berthLimit': port.berth_limit
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
    
    # Used to guide the vessel to ports needing the operation more.
    def eoh_calculations(self, state):
        leave_alone_ports = {}
        vessel_dict = state['vessel_dict']
        port_dict = state['port_dict']
        time_left = self.TIME_PERIOD_RANGE[-1] - state['time'] + 1 # +1 to account for the sink check
        for port_number, port_simp in port_dict.items():
            if port_simp['capacity'] is not None:
                if port_simp['isLoadingPort'] == 1:
                    projected_inv = port_simp['inventory']
                    for vessel_number, vessel_simp in vessel_dict.items():
                        if vessel_simp['position'] == port_number:
                            projected_inv -= vessel_simp['capacity']
                        elif vessel_simp['in_transit_towards'] is not None:
                            if vessel_simp['in_transit_towards']['destination_port_number'] == port_number:
                                projected_inv -= vessel_simp['capacity']
                    if 0 <= projected_inv + port_simp['rate'] * time_left <= port_simp['capacity'] + port_simp['rate']: # + rate to account for alpha slack
                        # Port does not need more operations. It can be left alone, but it might still have capacity to be operated at.
                        leave_alone_ports[port_number] = port_simp
                else:
                    projected_inv = port_simp['inventory']
                    for vessel_number, vessel_simp in vessel_dict.items():
                        if vessel_simp['position'] == port_number:
                            projected_inv += vessel_simp['capacity']      
                        elif vessel_simp['in_transit_towards'] is not None:
                            if vessel_simp['in_transit_towards']['destination_port_number'] == port_number:
                                projected_inv += vessel_simp['capacity']
                    if port_simp['capacity'] >= projected_inv - port_simp['rate'] * time_left >= 0 - port_simp['rate']:
                        # Port does not need more operations. It can be left alone, but it might still have capacity to be operated at.
                        leave_alone_ports[port_number] = port_simp
        return leave_alone_ports
    
    
    # Used to filter out arcs that are not feasible for the vessel
    def ports_that_must_be_avoided(self, state):
        avoid_ports = {}
        vessel_dict = state['vessel_dict']
        port_dict = state['port_dict']
        time_left = self.TIME_PERIOD_RANGE[-1] - state['time'] + 1
        minimum_vessel_capacity = min([vessel['capacity'] for vessel in vessel_dict.values()])
        
        for port_number, port_simp in port_dict.items():
            if port_simp['capacity'] is not None:
                if port_simp['isLoadingPort'] == 1:
                    projected_inv = port_simp['inventory']
                    for vessel_number, vessel_simp in vessel_dict.items():
                        if vessel_simp['position'] == port_number:
                            projected_inv -= vessel_simp['capacity']
                        elif vessel_simp['in_transit_towards'] is not None:
                            if vessel_simp['in_transit_towards']['destination_port_number'] == port_number:
                                projected_inv -= vessel_simp['capacity']
                    if projected_inv + port_simp['rate'] * time_left - minimum_vessel_capacity < 0:
                        avoid_ports[port_number] = port_simp
                        
                else:
                    projected_inv = port_simp['inventory']
                    for vessel_number, vessel_simp in vessel_dict.items():
                        if vessel_simp['position'] == port_number:
                            projected_inv += vessel_simp['capacity']
                        elif vessel_simp['in_transit_towards'] is not None:
                            if vessel_simp['in_transit_towards']['destination_port_number'] == port_number:
                                projected_inv += vessel_simp['capacity']
                    if projected_inv - port_simp['rate'] * time_left + minimum_vessel_capacity > port_simp['capacity']:
                        avoid_ports[port_number] = port_simp
        return avoid_ports
                            
    def encode_state(self, state, vessel_simp):
        # Skip the sink port
        # time = state['time']
        # time = np.array([time])
        port_dict = {k: port for k, port in state['port_dict'].items() if port['capacity'] is not None}
        vessel_dict = state['vessel_dict']
        
        port_critical_times = np.array([
        math.floor(current_inventory / rate) if port['isLoadingPort'] != 1 else math.floor((port['capacity'] - current_inventory) / rate)
        for port in port_dict.values()
        for rate, current_inventory in [(port['rate'], port['inventory'])]])

        # vessel_inventories = np.array([vessel['inventory'] / vessel['capacity'] for vessel in vessel_dict.values()])
        # current_vessel_number = np.array([vessel_simp['number']])
        current_vessel_position = np.array([vessel_simp['position']])
        current_vessel_class = np.array([vessel_simp['vessel_class']])
        # inventory_effect_on_ports = np.array([(vessel_simp['capacity'] / port['rate']) for port in port_dict.values()])
        
        inventory_effect_on_ports = []
        current_port_is_loading = port_dict[vessel_simp['position']]['isLoadingPort'] == 1
        for port in port_dict.values():
            # If both current and iterated ports are loading ports, or both are not, the effect is 0.
            # Otherwise, calculate the effect based on the vessel's capacity and the port's rate.
            effect = 0 if (current_port_is_loading == (port['isLoadingPort'] == 1)) else (vessel_simp['capacity'] // port['rate'])
            inventory_effect_on_ports.append(effect)
        inventory_effect_on_ports = np.array(inventory_effect_on_ports)

                
        
        travel_times = np.array(self.TRAVEL_TIME_DICT[vessel_simp['position']])
        # Change to -1 for ports the vessel cannot travel to, but keep 0 for the current position for the vessel.
        current_vessel_position = vessel_simp['position']
        # Change all entries that are 0 to -1, except for the current position in travel times
        travel_times = np.where(travel_times == 0, -1, travel_times)
        travel_times[current_vessel_position - 1] = 0
        
        
         
        
        vessel_positions = np.array([v['position'] if v['position'] else v['in_transit_towards']['destination_port_number'] for v in vessel_dict.values()])
        vessel_in_transit = np.array([v['in_transit_towards']['destination_time'] - state['time'] if v['in_transit_towards'] else 0 for v in vessel_dict.values()])
        
        # Find the indices of the ports that have > 0 in travel times
        legal_ports = np.where(travel_times > 0)[0]
        inv_at_arrival = [entry for entry in port_critical_times]
        for lp_idx in legal_ports:
            tt = travel_times[lp_idx]
            # Subtract the travel time from inv_at_arrival for the port
            inv_at_arrival[lp_idx] -= tt
            # Iterate through the indices and ports in vessel_positions and vessel_in_transit
            for v_idx, port_num in enumerate(vessel_positions):
                if port_num == lp_idx + 1:
                    # I know that the vessel is in transit towards the port
                    if vessel_in_transit[v_idx] <= tt:
                        # I know that the vessel will arrive at the port before the target vessel
                        effect_on_port = state['vessel_dict'][v_idx+1]['capacity'] // port_dict[lp_idx+1]['rate']
                        inv_at_arrival[lp_idx] += effect_on_port
                        
        inv_at_arrival = np.array(inv_at_arrival)
                    
            
        encoded_state = np.concatenate([port_critical_times, vessel_positions, vessel_in_transit, travel_times, inv_at_arrival, inventory_effect_on_ports])
        # current_vessel_class = np.array([vessel_simp['vessel_class']])        
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
    
    
    def sim_find_legal_actions_for_vessel(self, state, vessel, queued_actions, RUNNING_WPS):
        if RUNNING_WPS:
            legal_arcs = self.sim_get_legal_arcs_for_ps(state=state, vessel=vessel, special_sink_arcs=self.SPECIAL_SINK_ARCS, special_node_dict=self.SPECIAL_NODES_DICT, queued_actions=queued_actions)
        else:
            legal_arcs = self.sim_get_legal_arcs(state=state, vessel=vessel, special_sink_arcs=self.SPECIAL_SINK_ARCS, special_node_dict=self.SPECIAL_NODES_DICT, queued_actions=queued_actions)
            
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

    
    def all_ports_should_be_avoided(self, ports_to_avoid):
        if len(ports_to_avoid) == len(self.PORTS) - 1: # The vessel position port is removed from ports_to_avoid already. Therefore, we subtract 1
            return True
        return False
    
    def can_operate_at_port_now(self, vessel, port, queued_actions):
        # Have to check the berth limit for the port and the queue of actions. 
        berth_limit = port['berthLimit']
        
        operating_vessels_in_queued_actions = 0
        
        for action in queued_actions.values():
            arc = action[3]
            # Check if arc is interregional arc
            if arc.is_waiting_arc != 1 and arc.origin_node.port.number == port['number']:
                operating_vessels_in_queued_actions += 1
                
        if operating_vessels_in_queued_actions >= berth_limit:
            return False
        
        
        if port['isLoadingPort'] == 1:
            # The production has already been added to the inventory. Therefore, we only need to check if the inventory is less than the capacity
            if port['inventory'] >= vessel['capacity']:
                return True
        else:
            # Rate is subtracted to account for the consumption. It is the status at the end of the time period that is important.
            if port['inventory'] + vessel['inventory'] - port['rate'] <= port['capacity']:
                return True
        return False
    
    
    def sim_get_legal_arcs_for_ps(self, state, vessel, special_sink_arcs, special_node_dict, queued_actions):
            current_node_key = (vessel['position'], state['time'])
            current_node = self.NODE_DICT[current_node_key]
            
            # Only one possibility form source
            if current_node == self.SOURCE_NODE:
                return [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
            
            non_sim_vessel = self.VESSEL_DICT[vessel['number']]
            # Pre-filter arcs that originate from the current node
            return [arc for arc in self.VESSEL_ARCS[non_sim_vessel] if arc.origin_node == current_node]
            
            # Add the special sink arcs to the potential arcs if there are any
            # potential_arcs += [arc for arc in special_sink_arcs[non_sim_vessel] if arc.origin_node == current_node]
            # ports_that_must_be_avoided = self.ports_that_must_be_avoided(state)
            # Remove current port form the ports that must be avoided
            # if current_port['number'] in ports_that_must_be_avoided.keys():
            #     del ports_that_must_be_avoided[current_port['number']]
                
            # Remove all arcs going to a port that must be avoided
            # legal_arcs = [arc for arc in potential_arcs if arc.destination_node.port.number not in ports_that_must_be_avoided.keys() or arc.destination_node.time == self.TIME_PERIOD_RANGE[-1]+1]
            
            # if self.can_operate_at_port_now(vessel, current_port, queued_actions):
            #     # Check if the vessel can wait at the current port
            #     if self.waiting_arc_is_legal(current_port):
            #         if not legal_arcs:
            #             print('No legal arcs found')
            #             legal_arcs = [random.choice(potential_arcs)]
            #         return legal_arcs
            #     else:
            #         # Remove the waiting arc and return the rest
            #         legal_arcs = [arc for arc in legal_arcs if not arc.is_waiting_arc]
            #         if not legal_arcs:
            #             print('No legal arcs found')
            #             legal_arcs = [random.choice(potential_arcs)]
            #         return legal_arcs
            # else:
            #     # Waiting arc is the only legal arc
            #     legal_arcs = [arc for arc in legal_arcs if arc.is_waiting_arc]
            #     if not legal_arcs:
            #         print('No legal arcs found')
            #         # Take one random action from the potential arcs
            #         legal_arcs = [random.choice(potential_arcs)]
            #     return legal_arcs
        
        
    def sim_get_legal_arcs(self, state, vessel, special_sink_arcs, special_node_dict, queued_actions):
            current_node_key = (vessel['position'], state['time'])
            current_node = self.NODE_DICT[current_node_key]
            current_port = state['port_dict'][vessel['position']]
            
            # Only one possibility form source
            if current_node == self.SOURCE_NODE:
                return [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
            
            non_sim_vessel = self.VESSEL_DICT[vessel['number']]
            # Pre-filter arcs that originate from the current node
                    
            potential_arcs = [arc for arc in self.VESSEL_ARCS[non_sim_vessel] if arc.origin_node == current_node]
            # Remove the sink arc from the potential arcs
            potential_arcs = [arc for arc in potential_arcs if arc.destination_node.port != self.SINK_NODE.port]
            # Add the special sink arcs to the potential arcs if there are any
            potential_arcs += [arc for arc in special_sink_arcs[non_sim_vessel] if arc.origin_node == current_node and arc.destination_node.port.isLoadingPort != current_node.port.isLoadingPort]
            ports_that_must_be_avoided = self.ports_that_must_be_avoided(state)
            # Remove current port form the ports that must be avoided
            if current_port['number'] in ports_that_must_be_avoided.keys():
                del ports_that_must_be_avoided[current_port['number']]
                
            # Remove all arcs going to a port that must be avoided
            legal_arcs = [arc for arc in potential_arcs if arc.destination_node.port.number not in ports_that_must_be_avoided.keys() or arc.destination_node.time == self.TIME_PERIOD_RANGE[-1]+1]

            
            
            if self.can_operate_at_port_now(vessel, current_port, queued_actions):
                # Check if the vessel can wait at the current port
                if self.waiting_arc_is_legal(current_port):
                    if not legal_arcs:
                        legal_arcs = [random.choice(potential_arcs)]
                    return legal_arcs
                else:
                    # Remove the waiting arc and return the rest
                    legal_arcs = [arc for arc in legal_arcs if not arc.is_waiting_arc]
                    if not legal_arcs:
                        legal_arcs = [random.choice(potential_arcs)]
                    return legal_arcs
            else:
                # Waiting arc is the only legal arc
                legal_arcs = [arc for arc in legal_arcs if arc.is_waiting_arc]
                if not legal_arcs:
                    # Take one random action from the potential arcs
                    legal_arcs = [random.choice(potential_arcs)]
                return legal_arcs
            
            
    def waiting_arc_is_legal(self, current_port):
        if current_port['isLoadingPort'] == 1:
            if current_port['inventory'] + current_port['rate'] > current_port['capacity'] + current_port['rate']:
                return False
        else:
            if current_port['inventory'] - current_port['rate'] < - current_port['rate']:
                return False
        return True
    
    def get_legal_arcs(self, state, vessel):
        current_node = self.NODE_DICT[(vessel.position.number, state['time'])]
        if current_node == self.SOURCE_NODE:
            return [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]

        potential_arcs = [arc for arc in self.VESSEL_ARCS[vessel] if arc.origin_node == current_node]
        # All waiting arcs are legal
        legal_arcs = [arc for arc in potential_arcs if arc.is_waiting_arc]
        port = vessel.position
        if port.isLoadingPort == 1:
            # Rate is already added, since production is added to the inventory.
            if port.inventory >= vessel.capacity:
                # Append arcs to all discharging ports to the legal arcs list
                legal_arcs += [arc for arc in potential_arcs if arc.destination_node.port.isLoadingPort == -1]
        else:
            # Rate is subtracted to account for consumption.
            if port.inventory + vessel.inventory - port.rate <= port.capacity:
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
        
    def check_state(self, state, experience_path, replay, agent, INSTANCE, exploit):
        '''Evaluates the state and returns status and reward.'''
        total_reward_for_path  = 0
        if self.is_terminal(state):
            experience_path, feasible_path = self.update_rewards_in_experience_path(experience_path, agent, INSTANCE, exploit)
            for exp in experience_path:
                action = exp[1]
                time = exp[0]['time']
                first_infeasible_time = exp[7]
                if action is not None:
                    if first_infeasible_time is not None:
                        if time <= first_infeasible_time + 10:
                            replay.push(exp, self)
                    else:
                        replay.push(exp, self)
                    
                rew = exp[3]
                total_reward_for_path += rew
            state['done'] = True
            state['infeasible'] = self.is_infeasible(state=state)
            return state, total_reward_for_path, feasible_path
        # state['infeasible'] = self.is_infeasible(state=state)
        return state, None, None
    
    def log_episode(self, episode, total_reward_for_path, experience_path, state):
        infeasibility_counter = 0
        infeasibility_dict = {}
        for exp in experience_path:
            next_state = exp[4]
            # current_state = exp[0]
            # result_state = exp[4]
            time = next_state['time']
            if next_state['infeasible']:
                if time not in infeasibility_dict.keys():
                    infeasibility_dict[time] = True
                    
        # Check also the final state
        if state['infeasible']:
            infeasibility_dict[len(self.TIME_PERIOD_RANGE)] = True
            
        infeasibility_counter = len(infeasibility_dict.keys())
        print(f"Episode {episode}: Total Reward = {total_reward_for_path}\nInfeasibility Counter = {infeasibility_counter}")
        if infeasibility_counter > 0:
            # Sort the dict by time
            infeasibility_dict = dict(sorted(infeasibility_dict.items()))
            print('Infeasible time periods:', infeasibility_dict.keys())
        print('-----------------------------------')
        first_infeasible_time = experience_path[0][7]
        return first_infeasible_time, infeasibility_counter
    
    def log_window(self, episode, total_reward_for_path, experience_path, state, window_start, window_end):
        infeasibility_counter = 0
        infeasibility_dict = {}
        for exp in experience_path:
            next_state = exp[4]
            # current_state = exp[0]
            # result_state = exp[4]
            time = next_state['time']
            if next_state['infeasible']:
                if time not in infeasibility_dict.keys() and window_start <= time <= window_end:
                    infeasibility_dict[time] = True
                    
        # Check also the final state
        if state['infeasible']:
            infeasibility_dict[len(self.TIME_PERIOD_RANGE)] = True
            
        infeasibility_counter = len(infeasibility_dict.keys())
        print(f"Episode {episode}: Total Reward = {total_reward_for_path}\nInfeasibility Counter = {infeasibility_counter}")
        if infeasibility_counter > 0:
            # Sort the dict by time
            infeasibility_dict = dict(sorted(infeasibility_dict.items()))
            print('Infeasible time periods:', infeasibility_dict.keys())
        print('-----------------------------------')
        first_infeasible_time = experience_path[0][7]
        return first_infeasible_time, infeasibility_counter
    
    def apply_reward(self, exp, feasible_path, first_infeasible_time, reward):
        exp[3] = reward
        exp[6] = feasible_path
        exp[7] = first_infeasible_time
        
        
    def update_rewards_in_experience_path(self, experience_path, agent, INSTANCE, exploit):
        feasible_path = True
        first_infeasible_time = None
        horizon = len(self.TIME_PERIOD_RANGE)
        
        infeasibility_counter = 0
        
        infeasibility_dict = {}
        
        # for exp in experience_path:
        #     current_state, action, _, _, next_state, _, _, fi_time, terminal_flag = exp
        #     current_state_is_infeasible, infeasible_ports = self.sim_is_infeasible(current_state)
        #     # next_state_is_infeasible, _ = self.sim_is_infeasible(next_state)
        #     if current_state_is_infeasible:
        #         if first_infeasible_time is None:
        #             feasible_path = False
        #             first_infeasible_time = current_state['time']
        #         if current_state['time'] not in infeasibility_dict.keys():
        #             infeasibility_dict[current_state['time']] = True
        
        for exp in experience_path:
            current_state, action, _, _, next_state, _, _, fi_time, terminal_flag = exp
            next_state_is_infeasible, infeasible_ports = self.sim_is_infeasible(next_state)
            # next_state_is_infeasible, _ = self.sim_is_infeasible(next_state)
            if next_state_is_infeasible:
                if first_infeasible_time is None:
                    feasible_path = False
                    first_infeasible_time = next_state['time']
                if next_state['time'] not in infeasibility_dict.keys():
                    infeasibility_dict[next_state['time']] = True
                    
        infeasibility_counter = len(infeasibility_dict.keys())
        
        if first_infeasible_time is None:
            for exp in experience_path:
                current_state, action, _, _, next_state, _, _, fi_time, terminal_flag = exp
                next_state_is_infeasible, _ = self.sim_is_infeasible(next_state)
                if next_state_is_infeasible:
                    feasible_path = False
                    first_infeasible_time = next_state['time']
                    break
                
        if first_infeasible_time is None:
            print('yo')
        
        extra_reward_for_feasible_path = horizon
        
        checkpoint_rew = True
        checkpoint_step = 10
        horizon_checkpoints = [i for i in range(30, horizon, checkpoint_step)]
        # Find the maximum number in horizon_checkpoints that is lower than or equal to first_infeasible_time
        lowest_cp = 0
        
        if first_infeasible_time is not None:
            for idx, cp in enumerate(horizon_checkpoints):
                if cp <= first_infeasible_time and cp > lowest_cp:
                    lowest_cp = cp
                    cp_idx = idx
                    if lowest_cp > self.current_checkpoint:
                        self.current_checkpoint = lowest_cp
                else:
                    break
        else:
            # The path is feasible, so terminal reward is given instead
            checkpoint_rew = True
        
        if (lowest_cp == self.current_checkpoint or lowest_cp == self.current_checkpoint - checkpoint_step) and lowest_cp > 0:
        # if lowest_cp == self.current_checkpoint and lowest_cp > 0:
            # Find the index of lowest_cp in horizon_checkpoint
            extra_cp_reward = lowest_cp
        else:
            extra_cp_reward = 0
                
        
        for exp in experience_path:

            current_state, action, vessel, reward_ph, next_state, earliest_vessel, feasible_path_ph, fi_time, terminal_flag = exp
            current_state_time = current_state['time']
                
            '''Immediate Reward'''
            if not current_state['infeasible']:
                reward = 1
            else:
                reward = -1
            
            if current_state_time < lowest_cp and checkpoint_rew:
                reward += extra_cp_reward * agent.gamma ** (lowest_cp-current_state_time)     
                # reward += extra_cp_reward     
            
            # if first_infeasible_time:
            #     if current_state_time == first_infeasible_time:
            #         # reward -= (horizon-first_infeasible_time)
            #         reward -= horizon
            
            '''Future Reward'''
            # Future reward for the next state if the next state is not terminal
            if not terminal_flag and action is not None:
                vessel_simp = next_state['vessel_dict'][earliest_vessel['number']]
                encoded_next_state = self.encode_state(next_state, vessel_simp)
                # Use the target model to predict the future Q-values for the next state
                future_rewards = agent.target_model(torch.FloatTensor(encoded_next_state)).detach().numpy()
                # Select the maximum future Q-value as an indicator of the next state's potential
                max_future_reward = np.max(future_rewards)
                # Clip the future reward to be within the desired range, e.g., [-1, 1]
                if current_state['infeasible']:
                    max_future_reward = np.clip(max_future_reward, -10000, 0)
                if not current_state['infeasible']:
                    max_future_reward = np.clip(max_future_reward, -horizon, 1000)
                # Update the reward using the clipped future reward
                reward += max_future_reward
            
            '''Terminal Reward'''
            # If the whole path is feasible, add a huge reward discounted reward
            if feasible_path:
                current_time = current_state['time']
                time_until_terminal = horizon - current_time
                feasibility_reward = extra_reward_for_feasible_path * agent.gamma ** time_until_terminal
                # feasibility_reward = extra_reward_for_feasible_path
                if action is not None:
                    reward += feasibility_reward
            
            self.apply_reward(exp, feasible_path, first_infeasible_time, reward)
            
        if feasible_path:
            print('Feasible path')
            
        if (infeasibility_counter<self.current_best_IC and infeasibility_counter<=50 and exploit) or (infeasibility_counter<=10 and exploit):
            # save the main and target networks
            torch.save(agent.main_model.state_dict(), f'main_model_{INSTANCE}_INF_COUNTER_{infeasibility_counter}_{self.inf_counter_updates}.pth')
            torch.save(agent.target_model.state_dict(), f'target_model_{INSTANCE}_INF_COUNTER{infeasibility_counter}_{self.inf_counter_updates}.pth')
            with open(f'replay_buffer_{INSTANCE}_INF_COUNTER_{infeasibility_counter}_{self.inf_counter_updates}.pkl', 'wb') as f:
                pickle.dump(agent.memory, f)
            self.current_best_IC = infeasibility_counter
            self.inf_counter_updates += 1
            print(f'New best IC: {infeasibility_counter}')
        return experience_path, feasible_path
    
    def sim_all_vessels_finished(self, state):
        for vessel in state['vessel_dict'].values():
            if not vessel['isFinished']:
                return False
        return True

    def all_vessels_finished(self, state):
        vessels = state['vessels']
        return all(v.isFinished for v in vessels)

    def execute_action(self, vessel, 
                       action):
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
        if arc.destination_node.time == self.SINK_NODE.time:
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
    
    def produce(self, state):
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
    
class DQN(nn.Module):
    # def __init__(self, state_size, action_size):
    #     super(DQN, self).__init__()
    #     self.fc1 = nn.Linear(state_size, 64)     # First fully connected layer
    #     self.fc2 = nn.Linear(64, 128)             # Second fully connected layer
    #     self.fc3 = nn.Linear(128, 64)             # Third fully connected layer, newly added
    #     self.fc4 = nn.Linear(64, action_size)    # Output layer
    #     self.relu = nn.ReLU()                    # ReLU activation

    # def forward(self, state):
    #     x = self.relu(self.fc1(state))
    #     x = self.relu(self.fc2(x))
    #     x = self.relu(self.fc3(x))               # Activation for the newly added layer
    #     return self.fc4(x)
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)     # First fully connected layer
        self.fc2 = nn.Linear(64, 128)             # Second fully connected layer
        self.fc3 = nn.Linear(128, action_size)    # Output layer
        self.relu = nn.ReLU()                    # ReLU activation

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
        self.latest_infeasinble_time = 0

    def push(self, exp, env):
        state, action, vessel, reward, next_state, earliest_simp_vessel, is_feasible, first_infeasible_time, terminal_flag = exp
        # if first_infeasible_time:
        #     if state['time'] > first_infeasible_time:
        #         return # Do not add experiences that occur after the first infeasible time
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
        
        if is_feasible or first_infeasible_time>= env.current_checkpoint and env.current_checkpoint > 0:
            # Delete existing similar experience in both feasible and infeasible memory if it exists
            self.feasible_memory = deque([(id, exp) for id, exp in self.feasible_memory if id != exp_id])
            self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
            if len(self.feasible_memory) >= int(self.capacity/2):
                self.feasible_memory.popleft()
            self.feasible_memory.append((exp_id, exp))
            if first_infeasible_time > self.latest_infeasinble_time:
                self.latest_infeasinble_time = first_infeasible_time
        else:

            # Check if the experience is already in the feasible memory
            if not any(id == exp_id for id, _ in self.feasible_memory):
                # If not, add it to the infeasible memory. Substitute if similar exp already exists in infeasible memory
                self.memory = deque([(id, exp) for id, exp in self.memory if id != exp_id])
                if len(self.memory) >= self.capacity:
                    self.memory.popleft()
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
            max_num_regular = min(num_regular, len(self.memory))
            samples.extend(random.sample(self.memory, max_num_regular))
        return samples

    def __len__(self):
        return len(self.memory) + len(self.feasible_memory)

    def clean_up(self):
        # Calculate the target size for each memory based on 75% of their respective capacities
        # target_infeasible_size = int((self.capacity - int(self.capacity * self.feasibility_priority)) * 0.75)
        # target_feasible_size = int((self.capacity * self.feasibility_priority) * 0.75)
        
        # Reduce the size of each memory to the target size
        while len(self.memory) > self.capacity:
            self.memory.popleft()
        # while len(self.feasible_memory) > target_feasible_size:
        #     self.feasible_memory.popleft()
        return self

class DQNAgent:
    def __init__(self, ports, vessels, TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY, NON_RANDOM_ACTION_EPISODE_FREQUENCY, BATCH_SIZE, replay):
        # ports plus source and sink, vessel inventory, (vessel position, vessel in transit), time period, vessel_number
        state_size = len(ports) + 2 * len(vessels) + len(ports)*3
        action_size = len(ports)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = replay
        self.gamma = 0.99    # discount rate feasible paths
        self.sigma = 0.5     # discount rate infeasible paths
        self.epsilon = 0.75 # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.main_model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.main_model.parameters())
        self.TRAINING_FREQUENCY = TRAINING_FREQUENCY
        self.TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY
        self.NON_RANDOM_ACTION_EPISODE_FREQUENCY = NON_RANDOM_ACTION_EPISODE_FREQUENCY
        self.BATCH_SIZE = BATCH_SIZE
            
    def select_action(self, state, legal_actions,  env, vessel_simp, exploit):  
        
         # If there is only one legal action, choose it
        if len(legal_actions) == 1:
            action = legal_actions[0]
            arc = action[3]
            # new_state = copy.deepcopy(state)
            new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
            return action, new_state
        
        if not exploit:
            # If the time is close to the checkpoint increase the exploration rate
            if env.current_checkpoint > 0 and env.current_checkpoint - 20 <=state['time'] <= env.current_checkpoint:
                increased_epsilon = min(self.epsilon + 0.15, 0.75)
                if np.random.rand() < increased_epsilon:
                    action = random.choice(legal_actions)
                    arc = action[3]
                    new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                    return action, new_state
                
            if np.random.rand() < self.epsilon:
                action = random.choice(legal_actions)
                arc = action[3]
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
                
                
    def select_action_for_eval(self, state, legal_actions, env, vessel_simp):
        # If there is only one legal action, choose it
        if len(legal_actions) == 1:
            action = legal_actions[0]
            arc = action[3]
            # new_state = copy.deepcopy(state)
            new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
            return action, new_state
        
        if np.random.rand() < self.epsilon:
            action = random.choice(legal_actions)
            arc = action[3]
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
        # Find all unique destination port numbers
        unique_ports = {action[3].destination_node.port.number for action in legal_actions}
        # Choose the action with the highest q-value that is legal
        for q_value, destination_port_number in q_values:
            if destination_port_number in unique_ports:
                # Find all actions with the same destination port number in legal actions
                possible_actions = [action for action in legal_actions if action[3].destination_node.port.number == destination_port_number]
                action = max(possible_actions, key=lambda x: x[3].speed)
                arc = action[3]
                new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                return action, new_state
            
    def select_action_for_RH(self, state, legal_actions, env, vessel_simp, window_end, vessel_class_arcs):
        # If there is only one legal action, choose it
        if len(legal_actions) == 1:
            action = legal_actions[0]
            arc = action[3]
            # if arc.destination_node.time > window_end:
            #     a_0, a_1, a_2, arc = action
            #     # Find the sink arc from the same origin node
            #     vc = vessel_simp['vessel_class']
            #     for vc_arc in vessel_class_arcs[vc]:
            #         if arc.origin_node == vc_arc.origin_node and vc_arc.destination_node == env.SINK_NODE:
            #             action = (a_0, a_1, a_2, vc_arc)
            #             arc = vc_arc
            #             break
                
            new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
            return action, new_state
        
        if np.random.rand() < self.epsilon:
            action = random.choice(legal_actions)
            arc = action[3]
            # if arc.destination_node.time > window_end:
            #     a_0, a_1, a_2, arc = action
            #     # Find the sink arc from the same origin node
            #     vc = vessel_simp['vessel_class']
            #     for vc_arc in vessel_class_arcs[vc]:
            #         if arc.origin_node == vc_arc.origin_node and vc_arc.destination_node == env.SINK_NODE:
            #             action = (a_0, a_1, a_2, vc_arc)
            #             arc = vc_arc
            #             break
                    
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
        # Find all unique destination port numbers
        unique_ports = {action[3].destination_node.port.number for action in legal_actions}
        # Choose the action with the highest q-value that is legal
        for q_value, destination_port_number in q_values:
            if destination_port_number in unique_ports:
                # Find all actions with the same destination port number in legal actions
                possible_actions = [action for action in legal_actions if action[3].destination_node.port.number == destination_port_number]
                action = max(possible_actions, key=lambda x: x[3].speed)
                arc = action[3]
                # if arc.destination_node.time > window_end:
                #     a_0, a_1, a_2, arc = action
                #     # Find the sink arc from the same origin node
                #     vc = vessel_simp['vessel_class']
                #     for vc_arc in vessel_class_arcs[vc]:
                #         if arc.origin_node == vc_arc.origin_node and vc_arc.destination_node == env.SINK_NODE:
                #             action = (a_0, a_1, a_2, vc_arc)
                #             arc = vc_arc
                #             break
                new_state = env.update_vessel_in_transition_and_inv_for_state(state = state, vessel = vessel_simp, destination_port = arc.destination_node.port, destination_time = arc.destination_node.time, origin_port = arc.origin_node.port, quantity = action[2], operation_type = action[1])
                return action, new_state
                
                
    def select_action_for_ps(self, state, legal_actions, env, vessel_simp, RUNNING_MIRPSO):  
        
        if RUNNING_MIRPSO:
            if len(legal_actions) == 1:
                return legal_actions, True
        else:
            # If there is only one legal action, choose it
            if len(legal_actions) == 1:
                action = legal_actions[0]
                arc = action[3]
                return action
            
        # Encode state and add vessel number
        encoded_state = env.encode_state(state, vessel_simp)
        # Convert the state to a tensor
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.main_model(state_tensor).detach().numpy()
        q_values = q_values[0]
        # Sort the q-values, but keep track of the original indices
        q_values = [(q_value, index +1) for index, q_value in enumerate(q_values)]
        q_values.sort(reverse=True)
        
        if RUNNING_MIRPSO:
            possible_actions = []
            # Choose the action with the highest q-value that is legal
            for q_value, destination_port_number in q_values:
                for action in legal_actions:
                    arc = action[3]
                    if arc.destination_node.port.number == destination_port_number:
                        possible_actions.append(action)
                # If there are legal actions for the first choice of the agent, return them
                if possible_actions:
                    return possible_actions, destination_port_number
            return None, None
        
        else:       
            # Choose the action with the highest q-value that is legal
            for q_value, destination_port_number in q_values:
                for action in legal_actions:
                    arc = action[3]
                    if arc.destination_node.port.number == destination_port_number:
                        return action
                    
                
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
            target_q = torch.FloatTensor([reward]).to(encoded_state.device)
            # Predicted Q-values for the current state
            q_values = self.main_model(encoded_state)
            # Extract the Q-value for the action taken. This time keeping it connected to the graph.
            q_value = q_values.gather(1, torch.tensor([[action_idx]], dtype=torch.long).to(encoded_state.device)).squeeze()
            # Use the adjusted_reward directly as the target
            target_q = target_q.squeeze()
            # Compute loss
            loss = F.mse_loss(q_value, target_q)
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