import re
import gurobipy as gp

from optimization_utils import *
# from MIRP_GROUP_2 import perform_proximity_search
# from proximity_search import perform_proximity_search

# def get_current_x_solution_vars(model):
#     return {v.varName : v for v in model.getVars() if v.VarName.startswith('x')}

# def get_current_alpha_solution_vars(model):
#     return {v.varName: v for v in model.getVars() if v.VarName.startswith('a')}

# def get_current_s_solution_vars(model):
#     return {v.varName: v for v in model.getVars() if v.VarName.startswith('s')}


def extract_time_period_from_s_or_alpha_var_name(var_name):
    # First try to extract time period for variables like 's[i,t]' or 'alpha[i,t]'
    simple_match = re.search(r'\[(\d+),(\d+)\]', var_name)
    if simple_match:
        return int(simple_match.group(2))  # Returns the second number as an integer, representing the time period
    else:
        return None


def extract_time_period_from_x_var_name(var_name):
    # First try to extract time period for variables like 's[i,t]' or 'alpha[i,t]'
 
    # If the first pattern fails, then try to extract for variables like 'x_interregional_((i1, t1), (i2, t2))'
    # complex_matches = re.findall(r'\((\d+), (\d+)\)', var_name)
    complex_matches = re.findall(r'\((-?\d+),\s*(-?\d+)\)', var_name)

    if complex_matches and len(complex_matches) == 2:
        dep_time = int(complex_matches[0][1])  # Extract departure time from the first tuple
        arr_time = int(complex_matches[1][1])  # Extract arrival time from the second tuple
        return (dep_time, arr_time)
    else:
        return None

        
def set_warm_start(model, warm_start_solution, window_start):
    x_solution, s_solution, alpha_solution, _, _, _ = get_var_data(model)
    """
    Set the warm start for the model based on a given solution.
    """
    num_vars_fixed = 0
    for varname, var in x_solution.items():
        time = extract_time_period_from_x_var_name(varname)[0]
        val = round(warm_start_solution[varname])
        if time < window_start:
            # Round the value to the nearest integer
            var.start = val
            var.lb = val
            var.ub = val
            num_vars_fixed += 1
    for varname, var in s_solution.items():
        time = extract_time_period_from_s_or_alpha_var_name(varname)
        val = round(warm_start_solution[varname])
        if time < window_start:
            var.start = val
            var.lb = val
            var.ub = val
            num_vars_fixed += 1
    for varname, var in alpha_solution.items():
        time = extract_time_period_from_s_or_alpha_var_name(varname)
        val = round(warm_start_solution[varname])
        if time < window_start:
            var.start = val
            var.lb = val
            var.ub = val
            num_vars_fixed += 1
    model.update()
    return model
   
def prepare_window(model, window_start, window_end, x_vars, x_bounds, s_vars, s_bounds, alpha_vars, alpha_bounds):
    
    # Fix variables to their solution values if they are outside the current window
    for var_name, var in x_vars.items():
        time_period = extract_time_period_from_x_var_name(var_name)[0] # Assuming that the time period is the second element of the tuple key
        if time_period >= window_start:
            # fix_variable(var)
        # else:
            var.ub = x_bounds[var_name]  # Variable is free for the current window
            var.lb = 0
    
    for var_name, var in s_vars.items():
        time_period = extract_time_period_from_s_or_alpha_var_name(var_name)
        if time_period >= window_start:
            # fix_variable(var)
        # else:
            # Assuming we have a dictionary 's_bounds' that contains the upper bounds for 's' variables
            var.ub = s_bounds[var_name]
            var.lb = 0  # Assuming storage cannot be negative
            
    for var_name, var in alpha_vars.items():
        time_period = extract_time_period_from_s_or_alpha_var_name(var_name)
        if time_period >= window_start:
            # fix_variable(var)
        # else:
            # Assuming we have a dictionary 's_bounds' that contains the upper bounds for 's' variables
            var.ub = alpha_bounds[var_name]
            var.lb = 0  # Assuming storage cannot be negative
    model.update()
    return model
    
        
def solve_window(model, ps_data, TIME_LIMIT_PER_WINDOW):
    vessel_class_arcs = ps_data['vessel_class_arcs']
    """
    Solve the optimization model for a specific time window.
    """
    # Solve the model for the current window
    model.setParam('TimeLimit', TIME_LIMIT_PER_WINDOW)
    model.optimize()
    
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
    
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Time limit reached")
        current_solution_vars_x = get_current_x_solution_vars(model)
        current_solution_vals_x = get_current_x_solution_vals(model)
        current_solution_vals_s = get_current_s_solution_vals(model)
        current_solution_vals_alpha = get_current_alpha_solution_vals(model)
        active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        return model.objVal, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs
        
    elif model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found")
        current_solution_vars_x = get_current_x_solution_vars(model)
        current_solution_vals_x = get_current_x_solution_vals(model)
        current_solution_vals_s = get_current_s_solution_vals(model)
        current_solution_vals_alpha = get_current_alpha_solution_vals(model)
        active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        # Extract the solution for the current window
        return model.objVal, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs

def fix_variable(var):
    """
    Lock in decisions by fixing variables based on the solution.
    """
    var.ub = var.x  # Fix the variable at its current value
    var.lb = var.x

def get_var_data(model):
    """
    Retrieve the solution from the model after solving.
    """
    x_solution = get_current_x_solution_vars(model)
    s_solution = get_current_s_solution_vars(model)
    alpha_solution = get_current_alpha_solution_vars(model)
    x_bounds  = {varname: var.ub for varname, var in x_solution.items()}
    s_bounds = {varname: var.ub for varname, var in s_solution.items()}
    alpha_bounds = {varname: var.ub for varname, var in alpha_solution.items()}
    return x_solution, s_solution, alpha_solution, x_bounds, s_bounds, alpha_bounds

def save_original_rhs(model):
    """
    Save the original right-hand side values for constraints.
    """
    return {constr.ConstrName: constr.RHS for constr in model.getConstrs()}

import time
def rolling_horizon_optimization(model, horizon_length, window_size, step_size, TIME_LIMIT_PER_WINDOW, ps_data, agent, env, RUNNING_WPS_AND_RH, RUNNING_NPS_AND_RH, proximity_search_using_agent, RUNNING_MIRPSO):
    
    # First save all constraints, but remove time-dependent constraints from the model.
    time_constraints = store_and_remove_time_constraints(model)
    
    combined_solution = None
    for window_start in range(0, horizon_length + 1, step_size):
        
        # print('Stats after removing time-dependent constraints')
        window_end = window_start + window_size
        if window_end > horizon_length:
            window_end = horizon_length
        
        if combined_solution:
            # Apply warm start based on previous solution
            set_warm_start(model, combined_solution, window_start)
            
            x_solution, s_solution, alpha_solution, x_bounds, s_bounds, alpha_bounds = get_var_data(model)
            # Prepare the model for the current window
            model = prepare_window(model, window_start, window_end, x_solution, x_bounds, s_solution, s_bounds, alpha_solution, alpha_bounds)
            
        # Adjust constraints and objective function for the current window
        adjust_constraints_for_window(model, window_end, time_constraints)
        set_objective_for_window(model, window_start, window_end, ps_data)
        
        if RUNNING_WPS_AND_RH:
            ps_data['model'] = model
            # Solve the model for the current window with WPS
            current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs = proximity_search_using_agent(ps_data=ps_data, agent=agent, env=env, RUNNING_WPS_AND_RH=RUNNING_WPS_AND_RH, window_end=window_end, RUNNING_MIRPSO = RUNNING_MIRPSO, time_limit = TIME_LIMIT_PER_WINDOW)
            combined_solution = {**current_solution_vals_x, **current_solution_vals_s, **current_solution_vals_alpha}
        elif RUNNING_NPS_AND_RH:
            ps_data['model'] = model
            # Solve the model for the current window with NPS
            current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs = perform_proximity_search(ps_data=ps_data, RUNNING_NPS_AND_RH=RUNNING_NPS_AND_RH, window_end=window_end, time_limit=TIME_LIMIT_PER_WINDOW)
            combined_solution = {**current_solution_vals_x, **current_solution_vals_s, **current_solution_vals_alpha}
            
        else:
            # Solve the model for the current window
            current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs = solve_window(model, ps_data, TIME_LIMIT_PER_WINDOW)
            combined_solution = {**current_solution_vals_x, **current_solution_vals_s, **current_solution_vals_alpha}
        
        if window_end >= horizon_length:
            break
        else:
            print('New iteration')
            
    return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_vals_s, current_solution_vals_alpha, active_arcs
    
        
def extract_current_solution(model):
    return {var.VarName: var.X for var in model.getVars()}


def adjust_constraints_for_window(model, window_end, time_constraints):
    constraints_to_remove_from_time_constraints = []
    # Find a set of active constraints in the entire model
    for name, info in time_constraints.items():
        time = int(name.split('_Time')[1])
        if time <= window_end:
            # Add the constraint
            lhs, sense, rhs, name = info
            c = model.addConstr(lhs, sense, rhs, name)
            # remove the constraint from the time_constraints dictionary
            constraints_to_remove_from_time_constraints.append(name)
            
    for name in constraints_to_remove_from_time_constraints:
        del time_constraints[name]
        
    print(len(constraints_to_remove_from_time_constraints))
        
    model.update()

def store_and_remove_time_constraints(model):
    # Dictionary to store time-dependent constraints
    time_constraints = {}
    # List all constraints and identify time-dependent ones
    for constr in model.getConstrs():
        if 'Time' in constr.ConstrName:  # Check if the constraint name includes 'Time'
            lhs, sense, rhs, name = model.getRow(constr), constr.Sense, constr.RHS, constr.ConstrName
            time_constraints[name] =[lhs, sense, rhs, name]
            model.remove(constr)  # Remove the constraint from the model
    
    model.update()  # Always update the model after making changes
    print(f"Removed {len(time_constraints)} time-dependent constraints")
    return time_constraints
    
def restore_constraints(model, original_rhs):
    """
    Restore all constraints to their original RHS values after optimization.
    """
    for constr in model.getConstrs():
        constr.RHS = original_rhs[constr.ConstrName]
    model.update()

    
def set_objective_for_window(model, window_start, window_end, ps_data):
    costs_namekey = ps_data['costs_namekey']
    P = ps_data['P']
    regular_nodes = ps_data['regularNodes']
    
    # P = {(node.port.number, node.time): node.port.spotMarketPricePerUnit * (node.port.spotMarketDiscountFactor ** (node.time)) for node in regularNodes}
    # original_obj = gp.quicksum(costs[key]*x[key] for key in costs) + gp.quicksum(alpha[node.port.number, node.time] * P[(node.port.number, node.time)] for node in regularNodes)
    
    x_vars = get_current_x_solution_vars(model)
    alpha_vars = get_current_alpha_solution_vars(model)
    
    obj = gp.LinExpr()
    for varname, var in x_vars.items():
        time = extract_time_period_from_x_var_name(varname)[0]
        if time <= window_end:
            obj += var * costs_namekey[varname] # Assuming get_cost returns the cost coefficient for the variable
            
    for node in regular_nodes:
        # if window_start <= node.time <= window_end:
        if node.time <= window_end:
            obj += alpha_vars[f'alpha[{node.port.number},{node.time}]'] * P[(node.port.number, node.time)]

    model.setObjective(obj, gp.GRB.MINIMIZE)  # Set the objective to minimize; adjust accordingly
    model.update()


