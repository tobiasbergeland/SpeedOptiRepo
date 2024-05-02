import re
import copy

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
    return {v.varName: v.x for v in model.getVars() if v.VarName.startswith('s')}


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


def extract_time_period_from_s_or_alpha_var_name(var_name):
    simple_match = re.search(r'\\[(\\d+),(\\d+)\\]', var_name)
    if simple_match:
        return int(simple_match.group(2))  # Returns the second number as an integer, representing the time period
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
    
    
def evaluate_agent(env, agent):
    experience_path = []
    state = env.reset()
    done = False
    
    port_inventory_dict = {}
    vessel_inventory_dict = {}
    decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
    
    actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
    
    while not done:
        if state['time'] >= env.TIME_PERIOD_RANGE[-1]*(3/4):
            agent.epsilon = 0.25
            
        # Increase time and make production ports produce.
        if state['time'] in env.TIME_PERIOD_RANGE:
            state = env.increment_time_and_produce(state=state)
        else:
            #Only increment the time
            state['time'] += 1
                # Init port inventory is the inventory at this time. Time is 0 after the increment.
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            # Init vessel inventory is the inventory at this time
            vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
                
        # Check if state is infeasible or terminal
        state, total_reward_for_path, cum_q_vals_main_net, cum_q_vals_target_net, feasible_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
        
        if state['done']:
            first_infeasible_time, infeasibility_counter = env.log_episode(None, total_reward_for_path, experience_path, state, cum_q_vals_main_net, cum_q_vals_target_net)
            feasible_path = experience_path[0][6]
            state = env.consumption(state)
            
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            return experience_path, port_inventory_dict, vessel_inventory_dict
            
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
                    action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=True)
                    _, _, _, _arc = action
                    actions[vessel] = action
                # Perform the operation and routing actions and update the state based on this
                state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
        else:
            # Should check the feasibility of the state, even though no actions were performed. 
            state = env.simple_step(state, experience_path)
        # Make consumption ports consume regardless if any actions were performed
        state = env.consumption(state)
        
        # Save the inventory levels for the ports and vessels at this time
        if state['time'] in env.TIME_PERIOD_RANGE and state['time'] != 0:
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            vessel_inventory_dict[state['time']] = {vessel.number: vessel.inventory for vessel in state['vessels']}
            
            
def convert_path_to_MIRPSO_solution(env, experience_path, port_inventory_dict):
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
        
    for exp in experience_path:
        state, action, vessel, reward, next_state, earliest_vessel, feasible_path, first_infeasible_time, terminal_flag = exp
        if action is None:
            continue
            
        vessel_number, operation_type, quantity, arc = action
        if arc.destination_node.time == env.SINK_NODE.time:
            # Change to the sink arc
            for vessel_arc in env.VESSEL_ARCS[vessel]:
                if vessel_arc.origin_node == arc.origin_node and vessel_arc.destination_node == env.SINK_NODE:
                    arc = vessel_arc
                    break
        active_arcs[vessel_number].append(arc)
        
    active_X_keys = []
    for vessel in env.VESSELS:
        for arc in active_arcs[vessel.number]:
            active_X_keys.append(((arc.tuple), vessel.vessel_class))
            
    S_values = {}
    alpha_values = {}
    accumulated_alpha = {port.number : 0 for port in env.PORTS}
    for time, invs_at_time in port_inventory_dict.items():
        for port in env.PORTS:
            inventory = invs_at_time[port.number]
            if port.isLoadingPort == 1:
                if inventory > port.capacity:
                    alpha = inventory - port.capacity
                    alpha_values[(port.number, time-1)] = alpha
                    accumulated_alpha[port.number] += alpha
                S_values[(port.number, time)] = invs_at_time[port.number] - accumulated_alpha[port.number]
                # else:
                    # S_values[(port.number, time)] = invs_at_time[port.number]
            else:
                if inventory < 0:
                    alpha = abs(inventory)
                    alpha_values[(port.number, time-1)] = alpha
                    accumulated_alpha[port.number] += alpha
                    
                S_values[(port.number, time)] = invs_at_time[port.number] + accumulated_alpha[port.number]
                # else:    
                    # S_values[(port.number, time)] = invs_at_time[port.number]
            
    # W_values = {}
    # for time, invs_at_time in vessel_inventory_dict.items():
    #     for vessel in env.VESSELS:
    #         W_values[(time, vessel)] = invs_at_time[vessel.number]
        
    return active_X_keys, S_values, alpha_values


def warm_start_model(m, active_X_keys, S_values, alpha_values):
    # Initialize all 'x', 'a' variables to 0 to ensure a complete warm start
    for var in m.getVars():
        # if var.VarName.startswith('x') or var.VarName.startswith('o') or var.VarName.startswith('q'):
        if var.VarName.startswith('x'):
            # print(var.VarName)
            var.Start = 0  # Default start value for all variables not explicitly set
        elif var.VarName.startswith('alpha'):
            var.Start = 0
            
    # Setting initial values for 'x' variables based on active_X_keys
    for (arc_tuple, vessel_class) in active_X_keys:
        # Check if the arc is interregional or not
        origin_node, destination_node = arc_tuple
        if origin_node.port == destination_node.port:
            # Arc is not interregional, name is therefore f"x_non_interregional_{arc.tuple}_{arc.vessel_class}"
            x_var_name = f"x_non_interregional_{arc_tuple}_{vessel_class}" 
        else:
            x_var_name = f"x_interregional_{arc_tuple}_{vessel_class}"
            
        # x_var_name = f"x[{arc_tuple},{vessel}]"
        x_var = m.getVarByName(x_var_name)
        if x_var is not None:
            x_var.Start = 1
    
    # For 's' and 'w' variables, since you believe all values are set already, we maintain your original logic
    for (port_number, time), s_value in S_values.items():
        s_var_name = f"s[{port_number},{time}]"
        s_var = m.getVarByName(s_var_name)
        if s_var is not None:
            s_var.Start = s_value
            
    for (port_number, time), alpha_value in alpha_values.items():
        alpha_var_name = f"alpha[{port_number},{time}]"
        alpha_var = m.getVarByName(alpha_var_name)
        if alpha_var is not None:
            alpha_var.Start = alpha_value
    
    # Finally, update the model to apply these start values
    m.update()
    
    x_solution = {v.VarName: v.Start for v in m.getVars() if v.VarName.startswith('x')}
    warm_start_sol = {v.VarName: v.Start for v in m.getVars()}
    
    return x_solution, m, warm_start_sol