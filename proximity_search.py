import gurobipy as gp    
from gurobipy import GRB
import time

def get_current_x_solution(model):
    return {v.VarName: v.x for v in model.getVars() if v.VarName.startswith('x')}


def add_cutoff_constraint(model, current_best_obj, cutoff_value):
    cutoff_constr = model.addConstr(model.getObjective() <= current_best_obj - cutoff_value,
                                    f'cutoff: Current_best_obj: {current_best_obj} <= {current_best_obj - int(cutoff_value)}')
    model.update()
    return cutoff_constr

"""
def update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution):
    for var_name, var in x_variables.items():
        initial_value = current_solution[var_name]
        if initial_value == 0:
            model.addConstr(y[var_name] >= var, name=f'y_{var_name}_Hamming_distance')
        else:
            model.addConstr(y[var_name] >= 1 - var, name=f'y_{var_name}_Hamming_distance')
    model.setObjective(y.sum(), GRB.MINIMIZE)
    model.update()
    """

def update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution):
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
    model.setObjective(y.sum(), GRB.MINIMIZE)
    model.update()

def find_alternative_solution(model, x_variables, current_solution):

    # Correct approach for removing constraints based on their names
    to_remove = [constr for constr in model.getConstrs() if 'force_flip_' in constr.ConstrName or 'no_flip_' in constr.ConstrName or constr.ConstrName == "ensure_at_least_one_flip"]
    for constr in to_remove:
        model.remove(constr)
    model.update()
    # Implement logic for finding a better solution without focusing on minimizing Hamming distance
    # This could involve adjusting your model's constraints/objective to encourage exploration
    print("Finding alternative solution without minimizing Hamming distance...")
    # Remove previous Hamming distance constraints
    for constr in model.getConstrs():
        if 'Hamming_distance' in constr.ConstrName:
            model.remove(constr)
    model.update()
    
    flip_indicators = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="flip_indicator")
    
    # Add constraints to link flip indicators with actual variable changes
    for var_name, var in x_variables.items():
        initial_value = current_solution[var_name]
        # Constraint to link flip_indicator with the actual variable change
        model.addGenConstrIndicator(flip_indicators[var_name], True, var, GRB.EQUAL, 1 - initial_value, name=f'force_flip_{var_name}')
        model.addGenConstrIndicator(flip_indicators[var_name], False, var, GRB.EQUAL, initial_value, name=f'no_flip_{var_name}')

    # Set an objective that doesn't focus on the sum of flips, but ensures at least one flip
    # This can be a placeholder objective focusing on the original problem's goal, or you can add a constraint
    # to ensure at least one variable is flipped if the objective doesn't naturally encourage changes.
    # For example, ensure at least one flip_indicator is set to 1.
    model.addConstr(flip_indicators.sum() >= 1, "ensure_at_least_one_flip")
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
    #cutoff_value = 100000
    current_solution = get_current_x_solution(model)
    
    current_best_obj = model.getObjective().getValue()
    x_variables = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}

    initial_percentage_decrease = 0.0001
    cutoff_value = (initial_percentage_decrease * current_best_obj)
    
    y = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="y")

    t = time.time()
    i = 1
    
    # Start loop here:
    while time.time() - t < 900:
    #for i in range(1, 100):
        print(f'Proximity Search Iteration {i}')
        
        # Set the objective (back) to the original objective function
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()

        cutoff_value = (initial_percentage_decrease * current_best_obj)
        
        add_cutoff_constraint(model, current_best_obj, cutoff_value)
        
        update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)
        #find_alternative_solution_with_flip_limit(model, x_variables, current_solution, i, 100)

        #Stop optimization after finidng 1 solution
        # model.setParam(GRB.SolutionLimit, 1)
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        # Set time limit
        model.setParam(gp.GRB.Param.TimeLimit, 20)
        # Solve the modified problem
        model.optimize()
        i += 1

        if model.Status == GRB.TIME_LIMIT:
            print("Optimization stopped due to time limit.")
            find_alternative_solution(model, x_variables, current_solution)

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status ==  GRB.SOLUTION_LIMIT:
            print("Hamming Distance from previous solution:", model.objVal)
            
            new_solution = {v.VarName: v.X for v in model.getVars()}
            
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)
            print("Found a better solution.")
            print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')

            current_solution = new_solution
            current_best_obj = new_obj_value

            # Update current solution if improvement is found
            #    current_solution = new_solution
            #    current_best_obj = new_obj_value

                #cutoff_value = (initial_percentage_decrease * current_best_obj)
                #return current_solution, current_best_obj
            #    break
                
    solution = {v.VarName: v for v in model.getVars()}

    return solution, new_obj_value
"""
def perform_proximity_search(ps_data):  # Add other parameters as needed
    model = ps_data['model']
    original_objective_function = model.getObjective()
    current_solution = get_current_x_solution(model)
    current_best_obj = model.getObjective().getValue()
    x_variables = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
    
    initial_percentage_decrease = 0.001
    cutoff_value = (initial_percentage_decrease * current_best_obj)
    
    y = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="y")
    
    # Initialize improvement flag and iteration counter
    improvement = True
    iteration = 0
    
    # Start loop here:
    while improvement:
        iteration += 1
        print(f'Proximity Search Iteration {iteration}')
        
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()

        cutoff_value = (initial_percentage_decrease * current_best_obj)
        add_cutoff_constraint(model, current_best_obj, cutoff_value)
        update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)

        model.setParam(gp.GRB.Param.SolutionLimit, 1)  # Stop optimization after finding 1 solution
        model.optimize()

        if model.Status == GRB.SOLUTION_LIMIT:
            print("Hamming Distance from previous solution:", model.objVal)
            new_solution = {v.VarName: v.X for v in model.getVars()}
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)

            # Only update if a better solution is found
            print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')

            current_solution = new_solution
            current_best_obj = new_obj_value

            # Update current solution if improvement is found
            #    current_solution = new_solution
            #    current_best_obj = new_obj_value

                #cutoff_value = (initial_percentage_decrease * current_best_obj)
                #return current_solution, current_best_obj
            #    break
            improvement = True
        else:
            # If the model does not find a solution within the limit, it's time to stop the loop
            improvement = False

        # Increment iteration counter or add any other conditions you wish to use to break the loop
        
    solution = {v.VarName: v.X for v in model.getVars()}
    return solution, new_obj_value
    """