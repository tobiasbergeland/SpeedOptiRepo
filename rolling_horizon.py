import re
import gurobipy as gp

def get_current_x_solution_vars(model):
    return {v.varName : v for v in model.getVars() if v.VarName.startswith('x')}

def get_current_alpha_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('a')}

def get_current_s_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('s')}


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


# def solve_model_RHH(model, costs, P, RUNNING_MIRPSO, INSTANCE, start_period, end_period, warm_start_solution=None, arc_time_periods=None):
#     if warm_start_solution is None:
#         warm_start_solution = {}

#     else:
#         # Apply warm start: load previous solution into model
#         for var in model.getVars():
#             if var.VarName in warm_start_solution:
#                 var.start = warm_start_solution[var.VarName]

    
#     # Fix variables as needed, including x variables spanning the fixed boundary
#     for var in model.getVars():
#         if var.VarName.startswith('x'):
#             if arc_time_periods and var.VarName in arc_time_periods:
#                 dep_time, arr_time = arc_time_periods[var.VarName]
#                 if dep_time < start_period + 30:  # Check if the sailing starts before the fixed period ends
#                     var.lb = var.ub = warm_start_solution.get(var.VarName, var.x)
#         else:
#             time_period = extract_time_period_from_var_name(var.VarName)
#             if time_period != -1 and time_period < start_period + 30:
#                 if var.VarName in warm_start_solution:
#                     var.lb = var.ub = warm_start_solution[var.VarName]

#     model.optimize()

#     # After solving, save the new solution to use as the next warm start
#     new_solution = {var.VarName: var.x for var in model.getVars()}

#     return new_solution

"""
    new_sol = None

    roll = 30
    for i in range(0, HORIZON, roll):
        print(f"Solving model for time periods {i} to {i+roll}")
        if i == 0:
            warm_start_solution = None
            arc_time_periods = None
        else:
            warm_start_solution = new_sol
            arc_time_periods = {v.VarName: extract_time_period_from_var_name(v.VarName) for v in model.getVars() if v.VarName.startswith('x')}
        new_sol = solve_model_RHH(model, costs, P, RUNNING_MIRPSO, INSTANCE, i, i+roll, warm_start_solution, arc_time_periods)
        #for var_name, value in new_sol.items():
        #    print(f"{var_name}: {value}")
    """
    
    
# def solve_window(model, costs, P, RUNNING_MIRPSO, INSTANCE, window_size=60, fix_period=30):
#     # Solve for 60 days at a time
#     # Fix the first 30 days from the last window

#     for i in range(0, TIME_PERIOD_RANGE, window_size - fix_period):  # Notice the step size
#         start_period = i
#         end_period = i + window_size

#         if start_period == 0:
#             new_sol = None
#             arc_time_periods = None
#         else:
#             arc_time_periods = {v.VarName: extract_time_period_from_var_name(v.VarName) for v in model.getVars() if v.VarName.startswith('x')}

#         print(f"Solving model for time periods {start_period} to {end_period}")
#         new_sol = solve_model_RHH(model, costs, P, RUNNING_MIRPSO, INSTANCE, start_period, end_period, new_sol, arc_time_periods)
        

        
        
        
def set_warm_start(model, warm_start_solution, window_start):
    x_solution, s_solution, alpha_solution, _, _, _ = get_var_data(model)
    """
    Set the warm start for the model based on a given solution.
    """
    for varname, var in x_solution.items():
        time = extract_time_period_from_x_var_name(varname)[0]
        val = warm_start_solution[varname]
        if time < window_start:
            var.start = warm_start_solution[varname]
            var.lb = val
            var.ub = val
            
    for varname, var in s_solution.items():
        time = extract_time_period_from_s_or_alpha_var_name(varname)
        val = warm_start_solution[varname]
        if time < window_start:
            var.start = warm_start_solution[varname]
            var.lb = val
            var.ub = val
            
    for varname, var in alpha_solution.items():
        time = extract_time_period_from_s_or_alpha_var_name(varname)
        val = warm_start_solution[varname]
        if time < window_start:
            var.start = warm_start_solution[varname]
            var.lb = val
            var.ub = val
    model.update()
    return model
    
   
    # Apply warm start: load previous solution into model
    
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
    
        
def solve_window(model):
    """
    Solve the optimization model for a specific time window.
    """
    # Solve the model for the current window
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
    
    # Extract the solution for the current window
    return extract_current_solution(model)

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


def rolling_horizon_optimization(model, horizon_length, window_size, step_size, ps_data):
    print('Stats for the original model')
    model.printStats()
    # Print all the bounds for the variables
    for var in model.getVars():
        print(f"{var.VarName}: {var.LB} - {var.UB}")
    # First save all constraints, but remove time-dependent constraints from the model.
    time_constraints = store_and_remove_time_constraints(model)
    print('Stats after removing time-dependent constraints')
    model.printStats()
    
    current_solution = None
    for window_start in range(0, horizon_length + 1, step_size):
        window_end = window_start + window_size
        if window_end > horizon_length:
            window_end = horizon_length
        
        if window_start > 0:
            # Apply warm start based on previous solution
            set_warm_start(model, current_solution, window_start)
            
        if current_solution:
            x_solution, s_solution, alpha_solution, x_bounds, s_bounds, alpha_bounds = get_var_data(model)
        
            # Prepare the model for the current window
            model = prepare_window(model, window_start, window_end, x_solution, x_bounds, s_solution, s_bounds, alpha_solution, alpha_bounds)
            
        # Adjust constraints and objective function for the current window
        adjust_constraints_for_window(model, window_end, time_constraints)
        set_objective_for_window(model, window_start, window_end, ps_data)
        
        # Solve the model for the current window
        current_solution = solve_window(model)
        
        # Print the non zero variables
        for varname, value in current_solution.items():
            if value > 0:
                print(f"{varname}: {value}")
        print('New iteration')
        
    print('Stats after solving all windows')
    model.printStats()
    for var in model.getVars():
        print(f"{var.VarName}: {var.LB} - {var.UB}")
        

def extract_current_solution(model):
    return {var.VarName: var.X for var in model.getVars()}


def adjust_constraints_for_window(model, window_end, time_constraints):
    constraints_to_remove = []
    # Find a set of active constraints in the entire model
    for name, info in time_constraints.items():
        time = int(name.split('_Time')[1])
        if time <= window_end:
            # Add the constraint
            lhs, sense, rhs, name = info
            c = model.addConstr(lhs, sense, rhs, name)
            # remove the constraint from the time_constraints dictionary
            constraints_to_remove.append(name)
            
    for name in constraints_to_remove:
        del time_constraints[name]
        
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
        if time < window_end:
            obj += var * costs_namekey[varname] # Assuming get_cost returns the cost coefficient for the variable
            
    for node in regular_nodes:
        if window_start <= node.time < window_end:
            obj += alpha_vars[f'alpha[{node.port.number},{node.time}]'] * P[(node.port.number, node.time)]

    model.setObjective(obj, gp.GRB.MINIMIZE)  # Set the objective to minimize; adjust accordingly
    model.update()


