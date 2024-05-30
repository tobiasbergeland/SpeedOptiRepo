import re
import copy
import math

def get_current_x_solution_vars(model):
    return {v.varName : v for v in model.getVars() if v.VarName.startswith('x')}

def get_current_x_solution_vals(model):
    return {v.varName : v.x for v in model.getVars() if v.VarName.startswith('x')}

def get_current_alpha_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('a')}

def get_current_alpha_solution_vals(model):
    return {v.varName: v.x for v in model.getVars() if v.VarName.startswith('a')}

def get_current_s_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('s')}

def get_current_s_solution_vals(model):
    return {v.varName: int(round(v.x)) for v in model.getVars() if v.VarName.startswith('s')}


def find_corresponding_arcs(current_solution_x, vessel_class_arcs):
    active_arcs = {vc: [] for vc in vessel_class_arcs.keys()}
    
    for vessel_class, arcs in vessel_class_arcs.items():
        for arc in arcs:
            # Use the name of the variable to find the corresponding arc
            # Create the same variable name as in the model
            varname = convert_key_to_varname((arc.tuple, arc.vessel_class))
            if 1 - 1e-3 <= current_solution_x[varname]:
                # append to the active arcs with the vessel class as key if it is not present already
                if arc not in active_arcs[vessel_class]:
                    active_arcs.setdefault(arc.vessel_class, []).append(arc)
                # active_arcs.setdefault(arc.vessel_class, []).append(arc)
    return active_arcs

def convert_key_to_varname(key):
    arc_tuple = key[0]
    origin_node = arc_tuple[0]
    destination_node = arc_tuple[1]
    vc = key[1]
    
    # Check if the arc is an interregional arc or not
    # If origin port is the same as the destination port, it's a waiting arc, thus not a interregional arc
    if origin_node.port.number == destination_node.port.number or origin_node.time == -1:
        varname = f"x_non_interregional_{arc_tuple}_{vc}"
    else:
        varname = f"x_interregional_{arc_tuple}_{vc}"
    return varname



def create_alpha_variable_dict(non_zero_alphas, zero_alpha_keys):
    alpha_dict = {f'alpha[{key[0]},{key[1]}]': value for key, value in non_zero_alphas.items()}
    # Update the dictionary with zero alpha values
    for key in zero_alpha_keys:
        alpha_dict[f'alpha[{key[0]},{key[1]}]'] = 0
    return alpha_dict


# def varname_to_key(varname):
#     # Example parsing logic that needs to adapt to your actual expected format
#     if varname.startswith('x_non_interregional') or varname.startswith('x_interregional'):
#         try:
#             prefix, rest = varname.split('_', 2)[1:]  # Split to ignore the first part
#             arc_tuple, vessel_class = rest.rsplit('_', 1)  # Separate last underscore part for vessel class
#             arc_tuple = eval(arc_tuple)  # Convert string tuple to actual tuple, safely only if needed
#             return (arc_tuple, int(vessel_class))
#         except Exception as e:
#             raise ValueError(f"Error parsing variable name {varname}: {str(e)}")
#     else:
#         raise ValueError("Variable name does not match expected format")
    
def varname_to_key(var_name):
    """
    Parses the variable name to extract the tuple and vessel class more generically.
    
    Args:
    - var_name (str): The variable name formatted, e.g., 'x_interregional_((1, 0), (2, 7))_0'.
    
    Returns:
    - tuple: A tuple containing the arc tuple and vessel class as integer.
    """
    # Generalized regex to accommodate any type of prefix before the tuple
    pattern = r"x_[\w]+_\(\((-?\d{1,3}),\s*(-?\d{1,3})\),\s*\((-?\d{1,3}),\s*(-?\d{1,3})\)\)_(\d+)"
    match = re.search(pattern, var_name)
    
    if match:
        # Extracting individual numbers and creating tuples
        tuple_1 = (int(match.group(1)), int(match.group(2)))
        tuple_2 = (int(match.group(3)), int(match.group(4)))
        vessel_class = int(match.group(5))  # Convert the vessel class part to an integer
        
        arc_tuple = (tuple_1, tuple_2)
        
        return (arc_tuple, vessel_class)
    else:
        raise ValueError("Variable name format is incorrect.")

# def varname_to_key(varname):
#     # Regex updated to extract nested tuples and a final integer vc
#     match = re.search(r"x_(non_)?interregional_\(\((\d+), (\d)\), \((\d+), (\d+)\)\)_(\d+)", varname)
#     if not match:
#         raise ValueError("Variable name does not match expected format")

#     # Extract values from the regex match groups
#     origin_node_id = (int(match.group(2)), int(match.group(3)))
#     destination_node_id = (int(match.group(4)), int(match.group(5)))
#     vc = int(match.group(6))
    
#     # Reconstruct the arc tuple
#     arc_tuple = (origin_node_id, destination_node_id)
    
#     # Return the reconstructed key
#     return (arc_tuple, vc)

def key_to_node_info(key):
    arc_tuple, vc = key
    origin_node_id, destination_node_id = arc_tuple
    origin_port, origin_time  = origin_node_id
    destination_port, destination_time = destination_node_id
    return (origin_port, origin_time, destination_port, destination_time, vc)


# def extract_time_period_from_s_or_alpha_var_name(var_name):
#     simple_match = re.search(r'\\[(\\d+),(\\d+)\\]', var_name)
#     if simple_match:
#         return int(simple_match.group(2))  # Returns the second number as an integer, representing the time period
#     else:
#         return None
    
def extract_time_period_from_s_or_alpha_var_name(var_name):
    # Use regular expression to find the pattern '[(digits),(digits)]'
    match = re.search(r'\[(\d+),(\d+)\]', var_name)
    if match:
        # Return the second number as an integer
        return int(match.group(2))
    else:
        return None

def extract_time_period_from_x_var_name(var_name):
    complex_matches = re.findall(r'\\((-?\\d+),\\s*(-?\\d+)\\)', var_name)
    if complex_matches and len(complex_matches) == 2:
        dep_time = int(complex_matches[0][1])  # Extract departure time from the first tuple
        arr_time = int(complex_matches[1][1])  # Extract arrival time from the second tuple
        return (dep_time, arr_time)
    else:
        return None
    
def find_vessel_destinations(vessels_in_vc, vc_active_arcs, end_time):
    destination_nodes = {}
    # Find all arcs crossing the end time
    arcs_crossing_end_time = [arc for arc in vc_active_arcs if arc.origin_node.time < end_time and arc.destination_node.time >= end_time]
    crossing_traveling_arcs = [arc for arc in arcs_crossing_end_time if arc.origin_node.port != arc.destination_node.port]
    crossing_waiting_arcs = [arc for arc in arcs_crossing_end_time if arc.origin_node.port == arc.destination_node.port]
    taken_traveling_arcs = []
    vessels_assigned = []
    for v in vessels_in_vc:
        for arc in crossing_traveling_arcs:
            if arc not in taken_traveling_arcs:
                destination_nodes[v] = arc.destination_node
                taken_traveling_arcs.append(arc)
                vessels_assigned.append(v)
                break
    
    for v in vessels_in_vc:
        if v not in vessels_assigned:
            for arc in crossing_waiting_arcs:
                destination_nodes[v] = arc.destination_node
                vessels_assigned.append(v)
                break
            
        
    if len(vessels_assigned) != len(vessels_in_vc):
        raise ValueError("Not all vessels have been assigned a destination node")
    if len(destination_nodes) != len(vessels_in_vc):
        raise ValueError("Not all vessels have been assigned a destination node")
    if len(crossing_traveling_arcs) != len(taken_traveling_arcs):
        raise ValueError("Not all vessels have been assigned a destination node")
            
    return destination_nodes
    
    
    
def construction_heuristic_for_window(env, agent, window_start, window_end, combined_solution, INSTANCE, experience_path, port_inventory_dict, vessel_inventory_dict, current_solution_vars_x, current_solution_vals_x, active_arcs, VESSEL_CLASSES, vessel_class_arcs, model, current_solution_vals_s):
    best_solution = None
    best_inf_counter = env.TIME_PERIOD_RANGE[-1]
    solutions = []
    # Need some randomization in order for the agent to produce different results
    agent.epsilon = 0.3
    
    for i in range(20):
        if not combined_solution:
            # Window start is 0, so we need to start from scratch
            experience_path = []
            state = env.reset()
            done = False
            port_inventory_dict = {}
            vessel_inventory_dict = {}
            acc_alpha = {port.number : 0 for port in state['ports'] if port.rate}
            alpha_dict = {time: {port.number: 0 for port in state['ports']} for time in range(env.TIME_PERIOD_RANGE[-1] + 1)}
            decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
            actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
            state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            # increment time only
            state['time'] += 1
            # state = env.increment_time_and_produce(state=state)
            # Init port inventory is the inventory at this time. Time is 0 after the increment.
            # port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            # # Init vessel inventory is the inventory at this time
            # vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            
        else:
            
            # We need to recreate the state from the combined solution and start from the window start
            state = env.reset()
            # s_solution = get_current_s_solution_vars(model)
            experience_path = []
            # Reset the values in the port_inventory_dict but keep all keys
            port_inventory_dict = {time: {port.number: 0 for port in state['ports']} for time in range(env.TIME_PERIOD_RANGE[-1])}
            alpha_dict = {time: {port.number: 0 for port in state['ports']} for time in range(env.TIME_PERIOD_RANGE[-1] + 1)}
            acc_alpha = {port.number : 0 for port in state['ports'] if port.rate}
            
            # port_inventory_dict = {}
            
            # Set the inventory levels for the ports and vessels at the window start
            # for port_number, port_inventory in port_inventory_dict[window_start+1].items(): #S_31
            for port in state['ports']:
                for time in range(window_start):
                    port_inventory_dict[time][port.number] = current_solution_vals_s[f's[{port.number},{time}]']
            
            for port in state['ports']:
                port_number = port.number
                # Get the value of the s variable from current_solution_vals_s
                s_var_value = current_solution_vals_s[f's[{port_number},{window_start}]']
                # s_var_name = f"s[{port_number},{window_start+1}]"
                # s_var = model.getVarByName(s_var_name)
                # s_var_value = s_var.X
                port = env.PORTS[port_number -1]
                port.inventory = s_var_value
            # for vessel_number, vessel_inventory in vessel_inventory_dict[window_start].items():
            #     vessel = env.VESSELS[vessel_number -1]
            #     vessel.inventory = vessel_inventory
            # Set the time to the window start
            state['time'] = window_start
            state['done'] = False
            done = state['done']
            
            for vc in VESSEL_CLASSES:
                vessels_in_class = [vessel for vessel in state['vessels'] if vessel.vessel_class == vc]
                vc_active_arcs = active_arcs[vc]
                destination_nodes = find_vessel_destinations(vessels_in_class, vc_active_arcs, window_start)
                for v, node in destination_nodes.items():
                    v.position = None
                    v.in_transit_towards = (node.port, node.time)
                    if node.port.isLoadingPort == 1:
                        v.inventory = 0
                    else:
                        v.inventory = v.capacity
                        
            # Should check the feasibility of the state, even though no actions were performed. 
            # state = env.simple_step(state, experience_path)
            
            # port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            # vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            
        while not done:
            # Check if state is infeasible or terminal
            state, total_reward_for_path, feasible_path, alpha_register = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent, INSTANCE=INSTANCE, exploit = False, port_inventory_dict=port_inventory_dict)
            
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            
            acc_alpha = adjust_for_alpha(acc_alpha, alpha_dict, port_inventory_dict, state)
            
            # vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            
            if state['done']:
                first_infeasible_time, infeasibility_counter = env.log_window(None, total_reward_for_path, experience_path, state, window_start, window_end, alpha_dict)
                feasible_path = experience_path[0][6]
                
                # Check that the port inventories in port_inventory_dict are within the boundaries
                
                # if (infeasibility_counter < best_inf_counter):
                # best_inf_counter = infeasibility_counter
                best_solution = (experience_path, port_inventory_dict)
                solutions.append(best_solution)
                break
            
            # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
            env.update_vessel_status(state=state)
            # Find the vessels that are available to perform an action
            available_vessels = env.find_available_vessels(state=state)
            
            if available_vessels:
                    actions = {}
                    decision_basis_states = {}
                    actions_to_make = {}
                    decision_basis_state = env.custom_deep_copy_of_state(state)
                    for vessel in available_vessels:
                        corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                        decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                        legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel, queued_actions=actions_to_make, RUNNING_WPS = False, acc_alpha = acc_alpha)
                        action, decision_basis_state = agent.select_action_for_RH(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, window_end=window_end, vessel_class_arcs=vessel_class_arcs)
                        _, _, _, _arc = action
                        actions[vessel] = action
                        actions_to_make[vessel] = action
                    # Perform the operation and routing actions and update the state based on this
                    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            else:
                # Should check the feasibility of the state, even though no actions were performed. 
                state = env.simple_step(state, experience_path)
                
            # Increase time and make production ports produce.
            state = env.produce(state)
            
            # Make consumption ports consume regardless if any actions were performed
            state = env.consumption(state)
                
            state['time'] += 1
                    
    # experience_path = best_solution[0]
    # port_inventory_dict = best_solution[1]
    
    return solutions
    
# This is called just after logging. All inventories should be within their limits.
def adjust_for_alpha(acc_alpha, alpha_dict, port_inventory_dict, state):
    time = state['time']
    
    for port_number, inv in port_inventory_dict[time].items():
        port = state['port_dict'][port_number]
        if port.isLoadingPort == 1:
            if inv -acc_alpha[port_number] > port.capacity:
                # If inv-acc_alpha is less than 0
                if inv - acc_alpha[port_number] < 0:
                    print(f'Inv at port {port_number} at time {time} is {inv} and acc_alpha is {acc_alpha[port_number]}')
                alpha = inv - port.capacity - acc_alpha[port_number]
                acc_alpha[port_number] += alpha
                alpha_dict[time-1][port_number] = alpha
        else:
            if inv + acc_alpha[port_number] < 0:
                if inv + acc_alpha[port_number] > port.capacity:
                    print(f'Inv at port {port_number} at time {time} is {inv} and acc_alpha is {acc_alpha[port_number]}')
                alpha = -(inv + acc_alpha[port_number])
                acc_alpha[port_number] += alpha
                alpha_dict[time-1][port_number] = alpha
    return acc_alpha
    
    
def evaluate_agent(env, agent, INSTANCE):
    best_solution = None
    best_inf_counter = env.TIME_PERIOD_RANGE[-1]
    # Need some randomization in order for the agent to produce different results
    agent.epsilon = 0.1
    
    for i in range(10):
        
        experience_path = []
        state = env.reset()
        done = False
        
        port_inventory_dict = {}
        vessel_inventory_dict = {}
        decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
        
        actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
        state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
        
        # increment time only
        state['time'] += 1
        # state = env.increment_time_and_produce(state=state)
        # Init port inventory is the inventory at this time. Time is 0 after the increment.
        port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
        # Init vessel inventory is the inventory at this time
        vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
        
        
        while not done:
            # Check if state is infeasible or terminal
            state, total_reward_for_path, feasible_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent, INSTANCE=INSTANCE, exploit = False)
            
            if state['done']:
                first_infeasible_time, infeasibility_counter = env.log_episode(None, total_reward_for_path, experience_path, state)
                feasible_path = experience_path[0][6]
                # state = env.consumption(state)
                
                # port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
                # vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
                
                if infeasibility_counter < best_inf_counter:
                    best_inf_counter = infeasibility_counter
                    best_solution = (experience_path, port_inventory_dict, vessel_inventory_dict)
                break
                # return experience_path, port_inventory_dict, vessel_inventory_dict
                
            # Increase time and make production ports produce.
            state = env.produce(state)
            
            # port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            # vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
                
                
            # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
            env.update_vessel_status(state=state)
            # Find the vessels that are available to perform an action
            available_vessels = env.find_available_vessels(state=state)
            
            if available_vessels:
                    actions = {}
                    decision_basis_states = {}
                    actions_to_make = {}
                    decision_basis_state = env.custom_deep_copy_of_state(state)
                    for vessel in available_vessels:
                        corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                        decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                        legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel, queued_actions=actions_to_make, RUNNING_WPS = False)
                        action, decision_basis_state = agent.select_action_for_eval(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel)
                        _, _, _, _arc = action
                        actions[vessel] = action
                    # Perform the operation and routing actions and update the state based on this
                    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            else:
                # Should check the feasibility of the state, even though no actions were performed. 
                state = env.simple_step(state, experience_path)
            # Make consumption ports consume regardless if any actions were performed
            state = env.consumption(state)
            # state = env.consumption(state)
            
            # # # Save the inventory levels for the ports and vessels at this time
            # if state['time'] in env.TIME_PERIOD_RANGE and state['time'] >= 1:
            #     port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            #     vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
                
            state['time'] += 1
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
                
    experience_path = best_solution[0]
    port_inventory_dict = best_solution[1]
    vessel_inventory_dict = best_solution[2]
    return experience_path, port_inventory_dict, vessel_inventory_dict
            
            
def convert_path_to_MIRPSO_solution(env, solutions, window_end, window_start):
    best_solution = None
    for solution in solutions:
        experience_path, port_inventory_dict = solution
        
        # Create a dict with vesselnumber as key, and an empty list as value
        active_arcs_from_exp_path = {vessel.number: [] for vessel in env.VESSELS}
        
        if window_start == 0:
            for vessel in env.VESSELS:
                # Find the source arc for the vessel
                arcs = env.VESSEL_ARCS[vessel]
                for arc in arcs:
                    if arc.origin_node == env.SOURCE_NODE and arc.destination_node != env.SINK_NODE:
                        source_arc = arc
                        active_arcs_from_exp_path[vessel.number].append(source_arc)
                        break
                
        # active_X_keys = []
        for exp in experience_path:
            state, action, vessel, reward, next_state, earliest_vessel, feasible_path, first_infeasible_time, terminal_flag, alpha_state = exp
            if action is None:
                continue
                
            vessel_number, operation_type, quantity, arc = action
            if arc.destination_node.time == env.SINK_NODE.time:
                # Change to the sink arc
                for vessel_arc in env.VESSEL_ARCS[vessel]:
                    if vessel_arc.origin_node == arc.origin_node and vessel_arc.destination_node == env.SINK_NODE:
                        arc = vessel_arc
                        break
            active_arcs_from_exp_path[vessel_number].append(arc)
            # active_X_keys.append(((arc.tuple), vessel.vessel_class))
            
            
        active_X_keys = {}
        checkup_arcs = {}
        for vessel in env.VESSELS:
            for arc in active_arcs_from_exp_path[vessel.number]:
                
                if ((arc.tuple), vessel.vessel_class) not in active_X_keys.keys():
                    active_X_keys[((arc.tuple), vessel.vessel_class)] = 1
                else:
                    active_X_keys[((arc.tuple), vessel.vessel_class)] += 1
                    
                if (arc, vessel.vessel_class) not in checkup_arcs.keys():
                    checkup_arcs[(arc, vessel.vessel_class)] = 1
                else:
                    checkup_arcs[(arc, vessel.vessel_class)] += 1
                    
        feasible_x = True
        for key, counter in checkup_arcs.items():
            arc = key[0]
            if not arc.is_waiting_arc and counter > 1:
                feasible_x = False
                break
        if not feasible_x:
            continue
                
                    
        # Sum up the counters in the active_X_keys
        sumcounter = 0
        for key, counter in active_X_keys.items():
            sumcounter += counter
            
        # Check if similar to the number of arcs in the active_arcs_from_exp_path
        sum_aaiep = 0
        for vessel in env.VESSELS:
            active_arcs_from_exp_path_for_v = active_arcs_from_exp_path[vessel.number]
            sum_aaiep += len(active_arcs_from_exp_path_for_v)
        
        # print(f'The sum of the counters in active_X_keys is {sumcounter}. The sum of the number of arcs in active_arcs_from_exp_path is {sum_aaiep}')
        if sumcounter != sum_aaiep:
            raise ValueError('The number of arcs in active_arcs_from_exp_path is not equal to the sum of the counters in active_X_keys')
            
        
            
        
                
        alpha_states = set()
        infeasibility_counter = 0
        acc_alpha = {}
        alpha_values = {}
        S_values = {}
        alpha_register = {time : {} for time in range(len(port_inventory_dict))}
        for port in env.PORTS:
            acc_alpha[port.number] = 0
            # alpha_register[port.number] = {}
            for time in range(len(env.TIME_PERIOD_RANGE) + 1):
                alpha_register[time][port.number] = 0
                inv = port_inventory_dict[time][port.number]
                
                if port.isLoadingPort == 1:
                    if inv - acc_alpha[port.number] > port.capacity:
                        alpha = inv - acc_alpha[port.number] - port.capacity
                        acc_alpha[port.number] += alpha
                        alpha_register[time][port.number] = alpha
                        alpha_values[(port.number, time-1)] = alpha
                        if time <= window_end:
                            infeasibility_counter += 1
                        alpha_states.add(time)
                        S_values[(port.number, time)] = int(port.capacity)
                        
                        if first_infeasible_time is None:
                            first_infeasible_time = time
                        # if alpha > port.rate:
                        #     raise ValueError('Alpha is greater than the rate')
                    else:
                        the_s_val = int(inv - acc_alpha[port.number])
                        if the_s_val < 0:
                            print(f'The s value is negative ({the_s_val}) in a loading port at time {time}')
                        S_values[(port.number, time)] = int(inv - acc_alpha[port.number])
                        alpha_values[(port.number, time-1)] = 0
                            
                else:
                    if inv + acc_alpha[port.number] < 0:
                        alpha = - (inv + acc_alpha[port.number])
                        acc_alpha[port.number] += alpha
                        alpha_register[time][port.number] = alpha
                        alpha_values[(port.number, time-1)] = int(alpha)
                        if time <= window_end:
                            infeasibility_counter += 1
                        alpha_states.add(time)
                        S_values[(port.number, time)] = 0
                        
                        if first_infeasible_time is None:
                            first_infeasible_time = time
                        # if int(alpha) > port.rate:
                        #     raise ValueError('Alpha is greater than the rate')
                    else:
                        the_s_val_c = int(inv + acc_alpha[port.number])
                        if the_s_val_c > port.capacity:
                            print(f'The s value ({the_s_val}) is greater than the capacity ({port.capacity}) in a consumption port at time {time}')
                        S_values[(port.number, time)] = int(inv + acc_alpha[port.number])
                        alpha_values[(port.number, time-1)] = 0
                        
        feasible_solution = True
        for port in env.PORTS:
            for time in range(len(env.TIME_PERIOD_RANGE) + 1):
                if time > window_end:
                    break
                if S_values[(port.number, time)] > port.capacity or S_values[(port.number, time)] < 0:
                    feasible_solution = False
            
                    
        # if feasible_solution:
        #     best_solution = (active_X_keys, S_values, alpha_values, infeasibility_counter, feasible_solution, port_inventory_dict)
        #     return best_solution, active_arcs_from_exp_path
        
        if best_solution is None:
            best_solution = (active_X_keys, S_values, alpha_values, infeasibility_counter, feasible_solution, port_inventory_dict)
        elif not best_solution[4] and feasible_solution:
            best_solution = (active_X_keys, S_values, alpha_values, infeasibility_counter, feasible_solution, port_inventory_dict)
        elif not best_solution[4] and not feasible_solution and infeasibility_counter < best_solution[3]:
            best_solution = (active_X_keys, S_values, alpha_values, infeasibility_counter, feasible_solution, port_inventory_dict)
        elif best_solution[4] and feasible_solution and infeasibility_counter < best_solution[3]:
            best_solution = (active_X_keys, S_values, alpha_values, infeasibility_counter, feasible_solution, port_inventory_dict)
                    
    return best_solution, active_arcs_from_exp_path


def warm_start_model(m, active_X_keys, S_values, alpha_values, source_node, window_end):
    warm_start_vars = 0
    vars_set = set()
    # Initialize all 'x', 'a' variables to 0 to ensure a complete warm start
    for var in m.getVars():
        # if var.VarName.startswith('x') or var.VarName.startswith('o') or var.VarName.startswith('q'):
        if var.VarName.startswith('x'):
            # if var.VarName.startswith('x_non_interregional_((0, -1)'):
                # print('Yeah')
            # print(var.VarName)
            # if var is not fixed:
            if var.lb != var.ub:
                var.Start = 0
            else:
                var.Start = var.lb
                vars_set.add(var.VarName)
        elif var.VarName.startswith('alpha'):
            if var.lb != var.ub:
                var.Start = 0
            else:
                var.Start = var.lb
                vars_set.add(var.VarName)
        elif var.VarName.startswith('s'):
            if var.lb != var.ub:
                var.Start = 0
            else:
                var.Start = var.lb
                vars_set.add(var.VarName)
    m.update()
            
    waiting_arcs_taken_counter = {}
    traveling_taken = {}
    # Setting initial values for 'x' variables based on active_X_keys
    for (arc_tuple, vessel_class), counter in active_X_keys.items():
        # Check if the arc is interregional or not
        origin_node, destination_node = arc_tuple
        if origin_node.port == destination_node.port or origin_node == source_node:
            # Arc is not interregional, name is therefore f"x_non_interregional_{arc.tuple}_{arc.vessel_class}"
            x_var_name = f"x_non_interregional_{arc_tuple}_{vessel_class}" 
        else:
            x_var_name = f"x_interregional_{arc_tuple}_{vessel_class}"
            
        # x_var_name = f"x[{arc_tuple},{vessel}]"
        x_var = m.getVarByName(x_var_name)
        # Set the start value to 0 if the variable is found
        if x_var is not None and (x_var.lb != x_var.ub) and x_var_name not in vars_set:
            x_var.Start = round(counter)
            vars_set.add(x_var_name)
        elif x_var is not None and (x_var.lb == x_var.ub):
            x_var.Start = x_var.lb
            vars_set.add(x_var_name)
        m.update()
    
    for (port_number, time), s_value in S_values.items():
        s_var_name = f"s[{port_number},{time}]"
        s_var = m.getVarByName(s_var_name)
        if s_var is not None and (s_var.lb != s_var.ub) and s_var_name not in vars_set:
            s_var.Start = round(s_value)
            warm_start_vars += 1
            vars_set.add(s_var_name)
            # print(f'Variable {s_var_name} has been set to {s_value}')
        m.update()
            
            
    for (port_number, time), alpha_value in alpha_values.items():
        alpha_var_name = f"alpha[{port_number},{time}]"
        alpha_var = m.getVarByName(alpha_var_name)
        if alpha_var is not None and (alpha_var.lb != alpha_var.ub) and alpha_var_name not in vars_set:
            alpha_var.Start = round(alpha_value)
            warm_start_vars += 1
            vars_set.add(alpha_var_name)
        m.update()
            
    # Finally, update the model to apply these start values
    m.update()
    
    m.setParam('LogFile', 'gurobi_warmstart.log')
    m.setParam('IntFeasTol', 1e-4)
    m.setParam('FeasibilityTol', 1e-4)
    m.setParam('OptimalityTol', 1e-4)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    m.update()
    
    x_solution = {v.VarName: v.Start for v in m.getVars() if v.VarName.startswith('x')}
    
    print(f'Warm start solution has {warm_start_vars} variables set. The total number of variables in the model is {len(m.getVars())}')
    
    # Set the rest of the x_vars to 0
    
    vars_set_iteration_2 = set()
    # Print the vars that have not been set
    for var in m.getVars():
        if var.VarName not in vars_set and (x_var.lb != x_var.ub):
            if var.VarName.startswith('x_non') or var.VarName.startswith('x_inter'):
                # Set var to 0
                var.Start = 0
                vars_set_iteration_2.add(var.VarName)
                m.update()
                
    m.update()
    print(f'Warm start solution has {len(vars_set_iteration_2) + len(vars_set)} variables set. The total number of variables in the model is {len(m.getVars())}')
    
    # Write all variables and their start values to a file
    with open('warm_start_vars.txt', 'w') as f:
        for var in m.getVars():
            if var.VarName.startswith('x') and var.Start > 0:
                f.write(f'{var.VarName}: {var.Start}\n')
            elif var.VarName.startswith('alpha') or var.VarName.startswith('s'):
                f.write(f'{var.VarName}: {var.Start}\n')
                
                
    # Loop through the s vars
    for var in m.getVars():
        if var.VarName.startswith('s'):
            if var.lb > var.Start or var.ub < var.Start:
                print(f'Variable {var.VarName} has a start value of {var.Start} which is not within the bounds')
                # Extract the time from the variable name
                varname = var.VarName
                time = extract_time_period_from_s_or_alpha_var_name(varname)
                if time > window_end:
                    print(f'Variable {var.VarName} is outside the window So we can set the start value to a value within the bounds')
                    if var.Start > var.ub:
                        var.Start = var.ub
                    elif var.Start < var.lb:
                        var.Start = var.lb
                else:
                    print(f'Variable {var.VarName} is within the window and out of bounds. Trouble ahead')
                    
    
    warm_start_sol = {v.VarName: v.Start for v in m.getVars()}
    
    
    return x_solution, m, warm_start_sol

def time_passed(start_time):
    return time.time() - start_time


import gurobipy as gp
from gurobipy import GRB
import time

def perform_proximity_search(ps_data, RUNNING_NPS_AND_RH, window_end, TIME_LIMIT_PER_WINDOW, L):
    
    ch_obj = None
    model = ps_data['model']
    # print the amount of vars that have a start value
    start_time = time.time()
    
    # print the first 50 variables starting with x_non and their start values
    for var in model.getVars():
        if var.VarName.startswith('x_non_interregional_((0, -1)'):
            print(var.VarName, var.Start)
        
    
    # remove all y-variables
    for var in model.getVars():
        if var.VarName.startswith('y'):
            model.remove(var)
    
    # Make sure that no HammingDistance constraint is present
    for constr in model.getConstrs():
        if constr.ConstrName.startswith('y'):
            model.remove(constr)
    
    
    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            print("Feasible solution found")
            model.terminate()
            

    # Turn off heuristics.
    # model.setParam('Heuristics', 0)  
    # model.setParam('Presolve', 0)
    # model.update()
    
    model.optimize(callback)
    
    # Check if the model is infeasible or unbounded
    if model.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible")
        # Compute iis
        model.computeIIS()
        model.write('infeasible_window.ilp')
        
        return None
    elif model.status == gp.GRB.UNBOUNDED:
        print("Model is unbounded")
        return None
    
    else:
        print("Feasible solution found in Phase 1")
        ch_obj = model.objVal

        # Phase 2: Turn on heuristics and re-optimize with time limit
        model.setParam('Heuristics', 1)
    
        vessel_class_arcs = ps_data['vessel_class_arcs']
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        model.setParam(gp.GRB.Param.OutputFlag, 1)
        
        original_objective_function = model.getObjective()
        current_solution_vals_x = get_current_x_solution_vals(model)
        current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
        active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        current_solution_vals_alpha = get_current_alpha_solution_vals(model)
        current_solution_vals_s = get_current_s_solution_vals(model)
        current_best_obj = model.getObjective().getValue()

    
    PERCENTAGE_DECREASE = 0.1
    PERCENTAGE_CHANGE_FACTOR = 1.5 # Fixed
    PERCENTAGE_CHANGE_FACTOR_AFTER_INF = PERCENTAGE_CHANGE_FACTOR / 4
    # PERCENTAGE_DECREASE_AFTER_INFISIBILITY = PERCENTAGE_DECREASE/2
    # INFEASIBILITY_MULTIPLIER = 0.1
    cutoff_value = (PERCENTAGE_DECREASE * current_best_obj)
    
    y = model.addVars(current_solution_vars_x.keys(), vtype=GRB.BINARY, name="y")
    i = 0
    has_been_infeasible = False
    
    # Start loop here:
    while True:
        if has_been_infeasible:
            model.setParam(gp.GRB.Param.OutputFlag, 1)
        i += 1
        print_info = (i % 1 == 0)
            
        if print_info or has_been_infeasible:
            print(f'Proximity Search Iteration {i}')
        
        # Set the objective (back) to the original objective function
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()

        cutoff_value = (PERCENTAGE_DECREASE * current_best_obj)
        if print_info:
            print(f'Cutoff value in iteration {i} = {cutoff_value}. Percentage decrease is {PERCENTAGE_DECREASE*100}%')
        add_cutoff_constraint(model, current_best_obj, cutoff_value)

        update_objective_to_minimize_hamming_distance(model, y, current_solution_vars_x, current_solution_vals_x, None)
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        time_left = TIME_LIMIT_PER_WINDOW - time_passed(start_time)
        if cutoff_value < 1:
            model.setParam(gp.GRB.Param.TimeLimit, time_left if time_left >= 0 else 10)
        else:
            model.setParam(gp.GRB.Param.TimeLimit, min(L, time_left) if time_left >= 0 else 10)
        model.optimize()

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status == GRB.SOLUTION_LIMIT:
            last_feasible_solution_time = time_passed(start_time)
            if print_info or has_been_infeasible:
                print("Hamming Distance from previous solution:", model.objVal)
            if RUNNING_NPS_AND_RH:
                new_obj_value = evaluate_solution_with_original_objective_for_RH(model, ps_data, window_end)
            else:
                new_obj_value = evaluate_solution_with_original_objective(model, ps_data, print_info, has_been_infeasible)
            if print_info or has_been_infeasible:
                print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')
                
            current_best_obj = new_obj_value
            current_solution_vars_x = get_current_x_solution_vars(model)
            current_solution_vals_x = get_current_x_solution_vals(model)
            current_solution_vals_alpha = get_current_alpha_solution_vals(model)
            current_solution_vals_s = get_current_s_solution_vals(model)
            active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
            
            # if has_been_infeasible:
            #     PERCENTAGE_DECREASE = PERCENTAGE_DECREASE_AFTER_INFISIBILITY
            '''If feasible, increase the percentage change, by multiplying with PERCENTAGE CHANGE FACTOR (>1)'''
            PERCENTAGE_DECREASE = min(PERCENTAGE_DECREASE * PERCENTAGE_CHANGE_FACTOR, 0.2)

                
            current_time = time.time()
            time_diff = current_time - start_time
            if time_diff > TIME_LIMIT_PER_WINDOW:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs, ch_obj
            
        elif model.Status == GRB.TIME_LIMIT or model.Status == GRB.INFEASIBLE:
            has_been_infeasible = True
            
            PERCENTAGE_DECREASE *= PERCENTAGE_CHANGE_FACTOR_AFTER_INF
            # Return the best solution found. Time limit reached.
            print(f'Best objective value is {current_best_obj}.')
            if cutoff_value < 1:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs, ch_obj
            else:
                current_time = time.time()
                if start_time:
                    if current_time - start_time > TIME_LIMIT_PER_WINDOW:
                        remove_cutoff_constraints(model)
                        return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs, ch_obj
            
        
        
def remove_cutoff_constraints(model):
    cutoff_prefix = "cutoff_constraint"
    for constr in model.getConstrs():  # Iterate over all constraints in the model
        if constr.ConstrName.startswith(cutoff_prefix):
            model.remove(constr)
    model.update() 

def add_cutoff_constraint(model, current_best_obj, cutoff_value):
    # Prefix for cutoff constraint names
    cutoff_prefix = "cutoff_constraint"
    
    # Remove all existing cutoff constraints
    for constr in model.getConstrs():  # Iterate over all constraints in the model
        if constr.ConstrName.startswith(cutoff_prefix):
            model.remove(constr)
    
    # Ensure the model is updated after removals
    model.update()
    
    # Add the new cutoff constraint with a unique name to avoid name conflicts
    constraint_name = f"{cutoff_prefix}_{current_best_obj}_{cutoff_value}"
    cutoff_constr = model.addConstr(model.getObjective() <= current_best_obj - cutoff_value, name=constraint_name)
    
    # Update the model to add the new constraint
    model.update()
    
    return cutoff_constr


def update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution, weights):
  
    # Remove previous Hamming distance constraints
    for constr in model.getConstrs():
        if 'Hamming_distance' in constr.ConstrName:
            model.remove(constr)
    model.update()
    
    # Add new Hamming distance constraints
    for var_name, var in x_variables.items():
        if var_name not in y.keys():
            continue
        initial_value = current_solution[var_name]
        if initial_value == 0:
            model.addConstr(y[var_name] >= var - initial_value, name=f'y_{var_name}_Hamming_distance')
        else:
            model.addConstr(y[var_name] >= initial_value - var, name=f'y_{var_name}_Hamming_distance')
            
    # For all y-vars that do not have a weight, set the weight to 0.1
 
            
    if weights:
        for var_name in y.keys():
            
            if var_name not in weights.keys() or weights[var_name] == 0:
                weights[var_name] = 3
                
        weighted_hamming_distance = gp.quicksum(weights[var_name] * y[var_name] for var_name in x_variables.keys() if var_name in y.keys())
        model.setObjective(weighted_hamming_distance, GRB.MINIMIZE)
    else:
        model.setObjective(y.sum(), GRB.MINIMIZE)
    model.update()
    
    
def evaluate_solution_with_original_objective_for_RH(model, ps_data, window_end):
    costs = ps_data['costs']
    regularNodes = ps_data['regularNodes']
    P = ps_data['P']
    
    x = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
    alpha = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('a')}
    
    arc_costs = 0
    for key in costs:
        # Find the time of the arc
        arc_tuple = key[0]
        origin_node = arc_tuple[0]
        if origin_node.time <= window_end:
            arc_costs += costs[key] * x[convert_key_to_varname(key)]
    
    alpha_costs = 0
    for node in regularNodes:
        key = f'alpha[{node.port.number},{node.time}]'
        alpha_val = alpha[key]
        P_val = P[(node.port.number, node.time)]
        if node.time <= window_end:
            alpha_costs += alpha_val * P_val
        
    # print(f'Alpha_costs: {alpha_costs}')
    # print(f'Arc costs: {arc_costs}')
    
    new_obj_value = arc_costs + alpha_costs
    # print('New objective value:', new_obj_value)
    return new_obj_value

def evaluate_solution_with_original_objective(model, ps_data, print_info, has_been_infeasible):
    if print_info or has_been_infeasible:
        print("Evaluating solution with original objective")
    costs = ps_data['costs']
    regularNodes = ps_data['regularNodes']
    P = ps_data['P']
    
    # Directly calculate arc costs using comprehension
    x = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
    # Do the same fo alpha variables
    alpha = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('a')}
    
    arc_costs = sum(costs[key] * x[convert_key_to_varname(key)] for key in costs)
    
    alpha_costs = 0
    for node in regularNodes:
        key = f'alpha[{node.port.number},{node.time}]'
        alpha_val = alpha[key]
        P_val = P[(node.port.number, node.time)]
        alpha_costs += alpha_val * P_val
        
    if print_info or has_been_infeasible:
        print(f'Alpha_costs: {alpha_costs}')
        print(f'Arc costs: {arc_costs}')
    
    new_obj_value = arc_costs + alpha_costs
    if print_info or has_been_infeasible:
        print('New objective value:', new_obj_value)
    return new_obj_value
    
