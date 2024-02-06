import gurobipy as gp    
from gurobipy import GRB

def get_current_x_solution(model):
    return {v.VarName: v.x for v in model.getVars() if v.VarName.startswith('x')}


def add_cutoff_constraint(model, current_best_obj, cutoff_value):
    cutoff_constr = model.addConstr(model.getObjective() <= current_best_obj - cutoff_value,
                                    f'cutoff: Current_best_obj: {current_best_obj} - cutoff_value: {cutoff_value}  = {current_best_obj - cutoff_value}')
    model.update()
    return cutoff_constr


def update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution):
    for var_name, var in x_variables.items():
        initial_value = current_solution[var_name]
        if initial_value == 0:
            model.addConstr(y[var_name] >= var, name=f'y_{var_name}_Hamming_distance')
        else:
            model.addConstr(y[var_name] >= 1 - var, name=f'y_{var_name}_Hamming_distance')
    model.setObjective(y.sum(), GRB.MINIMIZE)
    model.update()
    

def evaluate_solution_with_original_objective(model, ps_data):
    print("Evaluating solution with original objective")
    costs = ps_data['costs']
    regularNodes = ps_data['regularNodes']
    vessels = ps_data['vessels']
    OPERATING_COST = ps_data['operating_cost']
    
    # Directly calculate arc costs using comprehension
    x = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
    arc_costs = sum(costs[key] * x[convert_key_to_varname(key)] for key in costs)
    print(f'Arc costs: {arc_costs}')
    
    # Directly calculate operation costs using comprehension
    o = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('o')}
    operation_costs = sum(o[f'o[{node.port.number},{node.time},{vessel}]'] * OPERATING_COST for node in regularNodes for vessel in vessels)
    print(f'Operating costs: {operation_costs}')
    
    new_obj_value = arc_costs + operation_costs
    print('New objective value:', new_obj_value)
    return new_obj_value


def convert_key_to_varname(key):
    inner = str(key[0])  # Convert the inner tuple to string and remove spaces
    vessel = key[1]
    return f"x[{inner},{vessel}]"


def perform_proximity_search(ps_data):  # Add other parameters as needed
    model = ps_data['model']
    original_objective_function = model.getObjective()
    cutoff_value = 100000
    current_solution = get_current_x_solution(model)
    
    current_best_obj = model.getObjective().getValue()
    x_variables = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
    
    y = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="y")
    
    # Start loop here:
    for i in range(1,3):
        print(f'Proximity Search Iteration {i}')
        
        # Set the objective (back) to the original objective function
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()
        
        add_cutoff_constraint(model, current_best_obj, cutoff_value)
        
        update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)

        # Solve the modified problem
        model.optimize()

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status == GRB.OPTIMAL:
            print("New objective value:", model.objVal)
            
            new_solution = {v.VarName: v.X for v in model.getVars()}
            
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)

            # Update current solution if improvement is found
            if new_obj_value < current_best_obj:
                print("Found a better solution.")
                print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')
                current_solution = new_solution
                current_best_obj = new_obj_value
                
    solution = {v.VarName: v for v in model.getVars()}

    return solution, new_obj_value