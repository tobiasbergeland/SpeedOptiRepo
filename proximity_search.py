import gurobipy as gp    
from gurobipy import GRB

def perform_proximity_search(ps_data):  # Add other parameters
    # Set initial solution and objective value
    current_solution = ps_data['initial_solution']
    model = ps_data['model']
    current_best_obj = model.getObjective().getValue()
    original_objective_function = model.getObjective()

    # Iterative improvement parameters
    max_iterations = 10  # Set according to your requirements
    iteration = 0
    
    while iteration < max_iterations:
        print("Proximity Search Iteration:", iteration)
        print("Current best objective value:", current_best_obj)
        print('---------------------------------')
        # Modify the model for Proximity Search
        model = change_to_proximity_search_model(model, current_solution, current_best_obj, original_objective_function, cutoff_value=10000)
        
        # set new solution limit for model
        model.setParam(gp.GRB.Param.SolutionLimit, iteration + 1)
        model.update()

        # Solve the modified model
        model.optimize()
        print(f'Objective value: {model.objVal}')

        # Check if a new solution is found
        if model.Status == GRB.OPTIMAL:
            # new_solution = [var.X for var in model.getVars()]
            new_solution = {v.VarName: v.X for v in model.getVars()}
            
            # Evaluate the new solution using the original objective
            
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)
            

            # Update current solution if improvement is found
            if new_obj_value < current_best_obj:
                print("Found a better solution.")
                print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')
                current_solution = new_solution
                current_best_obj = new_obj_value

        iteration += 1

    return current_solution, current_best_obj

def evaluate_solution_with_original_objective(model, ps_data):
    print("Evaluating solution with original objective")
    # Evaluate the new solution using the original objective
    # Get the values from the variables called x
    costs = ps_data['costs']
    adjusted_costs = {}
    regularNodes = ps_data['regularNodes']
    vessels = ps_data['vessels']
    OPERATING_COST = ps_data['operating_cost']
    
    x = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
    o = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('o')}
    
    for key in costs:
        cost = costs[key]
        adjusted_key = convert_key_to_varname(key)
        adjusted_costs[adjusted_key] = cost

  
    arc_costs = 0
    for adjusted_key in adjusted_costs.keys():
        arc_costs += adjusted_costs[adjusted_key] * x[adjusted_key]    
    print(f'Arc costs: {arc_costs}')
    
    operation_costs = 0
    for node in regularNodes:
        for vessel in vessels:
            operation_costs += o[f'o[{node.port.number},{node.time},{vessel}]'] * OPERATING_COST
    print(f'Operating costs: {operation_costs}')
    
    new_obj_value = arc_costs + operation_costs
    
    print('New objective value:', new_obj_value)

    return new_obj_value

def convert_key_to_varname(key):
    """
    Convert a key in the form of nested tuples to a string that matches the variable naming convention.

    Args:
        key (tuple): A key from the costs dictionary.

    Returns:
        str: A string representation of the key suitable for variable name lookup in the Gurobi model.
    """
    
    inner = str(key[0])  # Convert the inner tuple to string and remove spaces
    vessel = key[1]
    return f"x[{inner},{vessel}]"


def change_to_proximity_search_model(model, current_solution, current_best_obj, original_objective_function, cutoff_value=1000):
    """
    Modifies the given model for Proximity Search.
    Args:
        model: The optimization model.
        current_solution: The current solution (list of variable values).
        current_best_obj: The objective value of the current best solution.
        cutoff_value: The minimum improvement required for the new solution.
    Returns:
        The modified model.
    """
    
    variables = model.getVars()
    
    # Remove all variables that are not x variables
    variables = [var for var in variables if var.VarName.startswith('x')]
    
    hamming_distance_obj = gp.quicksum(1 if var.X != current_solution[var.VarName] else 0
                                   for var in variables)
    
    # Set the objective back to the original objective function
    model.setObjective(original_objective_function, GRB.MINIMIZE)
    model.update()    
    

    # Add a new cutoff constraint
    cutoff_constr = model.addConstr(model.getObjective() <= current_best_obj - cutoff_value, f'cutoff: Current_best_obj:{current_best_obj} - cutoff_value{cutoff_value}  = {current_best_obj - cutoff_value}')
    model.update()
    print(cutoff_constr.ConstrName)
    
    
    model.setObjective(hamming_distance_obj, GRB.MINIMIZE)
    
    # Update the model
    model.update()
    print('Model updated for Proximity Search')
    return model