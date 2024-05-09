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
    
    actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel, random_start = False)[0] for vessel in state['vessels']}
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

def time_passed(start_time):
    return time.time() - start_time


import gurobipy as gp
from gurobipy import GRB
import time

def perform_proximity_search(ps_data, RUNNING_NPS_AND_RH, window_end, time_limit):
        
    model = ps_data['model']
    vessel_class_arcs = ps_data['vessel_class_arcs']
    model.setParam(gp.GRB.Param.SolutionLimit, 1)
    model.setParam(gp.GRB.Param.OutputFlag, 1)
    
    start_time = time.time()
    model.optimize()
    original_objective_function = model.getObjective()
    current_solution_vals_x = get_current_x_solution_vals(model)
    current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
    active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
    current_solution_alpha = get_current_alpha_solution_vals(model)
    current_solution_s = get_current_s_solution_vals(model)
    current_best_obj = model.getObjective().getValue()
    
    PERCENTAGE_DECREASE = 0.1
    PERCENTAGE_CHANGE_FACTOR = 1.1 #MAX 2
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
        time_left = time_limit - time_passed(start_time)
        model.setParam(gp.GRB.Param.TimeLimit, min(60, time_left) if time_left >= 0 else 10)
        model.optimize()

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status ==  GRB.SOLUTION_LIMIT:
            if print_info or has_been_infeasible:
                print("Hamming Distance from previous solution:", model.objVal)
            current_solution_vals_x = get_current_x_solution_vals(model)
            if RUNNING_NPS_AND_RH:
                new_obj_value = evaluate_solution_with_original_objective_for_RH(model, ps_data, window_end)
            else:
                new_obj_value = evaluate_solution_with_original_objective(model, ps_data, print_info, has_been_infeasible)
            if print_info or has_been_infeasible:
                print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')
                
            current_best_obj = new_obj_value
            current_solution_vars_x = get_current_x_solution_vars(model)
            current_solution_alpha = get_current_alpha_solution_vals(model)
            current_solution_s = get_current_s_solution_vals(model)
            # if has_been_infeasible:
            #     PERCENTAGE_DECREASE = PERCENTAGE_DECREASE_AFTER_INFISIBILITY
            '''If feasible, increase the percentage change, by multiplying with PERCENTAGE CHANGE FACTOR (>1)'''
            PERCENTAGE_DECREASE = min(PERCENTAGE_DECREASE * PERCENTAGE_CHANGE_FACTOR, 0.5)

                
            current_time = time.time()
            time_diff = current_time - start_time
            if time_diff > time_limit:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            
        elif model.Status == GRB.TIME_LIMIT or model.Status == GRB.INFEASIBLE:
            has_been_infeasible = True
            
            PERCENTAGE_DECREASE *= PERCENTAGE_CHANGE_FACTOR_AFTER_INF
            # Return the best solution found. Time limit reached.
            print(f'Best objective value is {current_best_obj}.')
            if cutoff_value < 1:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            else:
                current_time = time.time()
                if start_time:
                    if current_time - start_time > time_limit:
                        remove_cutoff_constraints(model)
                        return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            
        
        
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
        initial_value = current_solution[var_name]
        if initial_value == 0:
            model.addConstr(y[var_name] >= var - initial_value, name=f'y_{var_name}_Hamming_distance')
        else:
            model.addConstr(y[var_name] >= initial_value - var, name=f'y_{var_name}_Hamming_distance')
            
    if weights:
        weighted_hamming_distance = gp.quicksum(weights[var_name] * y[var_name] for var_name in x_variables.keys())
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
    
