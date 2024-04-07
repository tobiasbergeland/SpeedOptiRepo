import gurobipy as gp    
from gurobipy import GRB
import time, random, math

def get_current_x_solution(model):
    return {v.VarName: v.x for v in model.getVars() if v.VarName.startswith('x')}

def get_current_x_solution_initial(model):
    return {v.VarName: v.Start for v in model.getVars() if v.VarName.startswith('x')}

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

def update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution):
    #to_remove = [constr for constr in model.getConstrs() if 'force_flip_' in constr.ConstrName or 'no_flip_' in constr.ConstrName or constr.ConstrName == "ensure_at_least_one_flip"]
    #for constr in to_remove:
    #    model.remove(constr)
    #model.update()
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

def evaluate_solution_with_original_objective(model, ps_data):
    print("Evaluating solution with original objective")
    costs = ps_data['costs']
    regularNodes = ps_data['regularNodes']
    vessels = ps_data['vessels']
    # OPERATING_COST = ps_data['operating_cost']
    
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

# Function to update the Tabu list
def update_tabu_list(tabu_list):
    for var in list(tabu_list.keys()):
        tabu_list[var] -= 1
        if tabu_list[var] <= 0:
            del tabu_list[var]

def perform_proximity_search(ps_data):  # Add other parameters as needed
    model = ps_data['model']
    model.setParam(gp.GRB.Param.SolutionLimit, 1)
    model.optimize()
    original_objective_function = model.getObjective()
    #cutoff_value = 100000
    current_solution = get_current_x_solution(model)
    #current_solution = get_current_x_solution_initial(model)
    #current_solution = ps_data['initial_solution']
    
    current_best_obj = model.getObjective().getValue()
    x_variables = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}

    initial_percentage_decrease = 0.0005
    discount_factor = 0.9
    cutoff_value = (initial_percentage_decrease * current_best_obj)
    
    y = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="y")

     # Solution pool configuration for initial broad search
    #model.setParam(GRB.Param.PoolSearchMode, 2)  # Enhanced search for diverse solutions
    #model.setParam(GRB.Param.PoolSolutions, 10)  # Store up to 10 diverse solutions

    # Initialize the counter at the beginning of the perform_proximity_search function
    tabu_list = {}
    tabu_tenure = 10

    t = time.time()
    i = 1
    #tabu_list = []
    
    # Start loop here:
    while time.time() - t < 300:
    #for i in range(1, 100):
        print(f'Proximity Search Iteration {i}')

        update_tabu_list(tabu_list)
        #if i % 10 == 0:
        #    initial_percentage_decrease *= 0.5
        
        # Set the objective (back) to the original objective function
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()

        cutoff_value = (initial_percentage_decrease * current_best_obj)*discount_factor
        
        add_cutoff_constraint(model, current_best_obj, cutoff_value)

        #add_tabu_constraints(model, tabu_list, x_variables)  # Add constraints to avoid tabu solutions
        update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)
        #find_alternative_solution_with_flip_limit(model, x_variables, current_solution, i, 100)

        #Stop optimization after finidng 1 solution
        # model.setParam(GRB.SolutionLimit, 1)
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        #model.setParam(gp.GRB.Param.MIPFocus, 1)
        # Set time limit
        #model.setParam(gp.GRB.Param.TimeLimit, 10)
        # Set heurstic exploration
        #model.setParam(gp.GRB.Param.Heuristics, 0.6)
        # Solve the modified problem
        model.optimize()
        i += 1

        if t == GRB.TIME_LIMIT:
            print("Optimization stopped due to time limit.")
            break

        #if model.Status == GRB.TIME_LIMIT:
            #diversified_restart(model, x_variables, method='perturb')
            #current_solution = get_current_x_solution(model)
            #update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)
            #model.update()
            #cutoff_value = (initial_percentage_decrease * current_best_obj)*0.9
        
            #add_cutoff_constraint(model, current_best_obj, cutoff_value)
            #print("Optimization stopped due to time limit.")
            #find_alternative_solution(model, x_variables, current_solution)
            #model.setObjective(original_objective_function, GRB.MINIMIZE)
            #model.setParam(gp.GRB.Param.TimeLimit, 900)
            #model.update()
            #model.optimize()
            
            #update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status ==  GRB.SOLUTION_LIMIT:
            print("Hamming Distance from previous solution:", model.objVal)
            
            new_solution = {v.VarName: v.X for v in model.getVars()}

            # Assuming new_solution has been determined after an optimization run
            for var_name, new_value in new_solution.items():
                if var_name.startswith('x') and current_solution.get(var_name, None) != new_value:
                    # Only update Tabu list for variables that have changed
                    # Ensure you're checking against a baseline where the variable exists
                    tabu_list[var_name] = tabu_tenure


            
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)
            print("Found a better solution.")
            print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')


            #abu_solution_representation = {var_name: int(new_solution[var_name]) for var_name, var in new_solution.items() if var_name.startswith('x')}
            
            # Add the new solution's representation to the tabu list if not already present
            # Note: This is a simple approach; depending on your problem, you might need to check for duplicates more carefully
            #if tabu_solution_representation not in tabu_list:
            #    tabu_list.append(tabu_solution_representation)
            #    print("Added new solution to tabu list.")


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
def add_cutoff_constraint(model, current_best_obj, cutoff_value):
    cutoff_constr = model.addConstr(model.getObjective() <= current_best_obj - cutoff_value,
                                    f'cutoff: Current_best_obj: {current_best_obj} <= {current_best_obj - int(cutoff_value)}')
    model.update()
    return cutoff_constr
"""

"""
def diversified_restart(model, x_variables, method='perturb', saved_solutions=None):
    if method == 'perturb':
        # Randomly perturb a subset of the variables
        for var_name, var in x_variables.items():
            if random.random() < 0.5:  # Adjust the probability as needed
                var.Start = 1 - var.Start  # Flip binary variable
"""

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

"""
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
    model.optimize()
"""    

"""
def add_tabu_constraints(model, tabu_list, x_variables):
    for i, tabu_solution in enumerate(tabu_list):
        constr_expr = gp.LinExpr()
        for var_name, var_value in tabu_solution.items():
            var = x_variables.get(var_name)
            if var is not None:
                # Correctly construct the constraint expression based on the variable's value
                if var_value == 1:
                    constr_expr += var
                else:
                    constr_expr += (1 - var)
        # Ensure the total expression does not equal the number of variables (to avoid repeating the tabu solution)
        model.addConstr(constr_expr <= len(tabu_solution) - 1, f'tabu_constraint_{i}')
    model.update()
"""
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
def add_conditional_cutoff_constraint(model, best_obj_value, improvement_factor=0.001):
    """Adds a cutoff constraint to the model based on the best objective value found so far."""
    cutoff_value = best_obj_value * (1 - improvement_factor)  # Seeking at least a 1% improvement
    model.addConstr(model.getObjective() <= cutoff_value, name="cutoff_constraint")
    model.update()

def acceptance_probability(old_cost, new_cost, temperature):
    """Calculates the probability of accepting a worse solution based on temperature."""
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)

def perform_proximity_search_with_simulated_annealing(ps_data):
    model = ps_data['model']
    original_objective_function = model.getObjective()

    current_solution = get_current_x_solution(model)
    current_best_obj = model.getObjective().getValue()

    x_variables = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}

    y = model.addVars(x_variables.keys(), vtype=GRB.BINARY, name="y")  # For Hamming distance calculation

    initial_temp = 50000  # Example starting temperature
    cooling_rate = 0.99  # Example cooling rate
    min_temp = 1  # Minimum temperature to halt the algorithm
    temperature = initial_temp

    initial_percentage_decrease = 0.001
    discount_factor = 0.9

    #best_solution = current_solution
    #best_obj_value = float('inf')  # Initialize with a high value
    
    iteration = 0
    while temperature > min_temp:
        iteration += 1
        temperature = initial_temp * (cooling_rate ** iteration)

        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        model.update()

        cutoff_value = (initial_percentage_decrease * current_best_obj)*discount_factor


        #if cutoff_constr is not None:
        #    model.remove(cutoff_constr)
        #cutoff_constr = add_conditional_cutoff_constraint(model, current_best_obj)
        
        #update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)
        #model.optimize()
        # Periodically apply the cutoff constraint
        #if iteration % 10 == 0:  # Adjust the frequency as needed
        #    if current_best_obj< float('inf'):
                #add_conditional_cutoff_constraint(model, current_best_obj)
        add_cutoff_constraint(model, current_best_obj, cutoff_value)
        
        update_objective_to_minimize_hamming_distance(model, y, x_variables, current_solution)
        model.optimize()

        if model.Status == GRB.SOLUTION_LIMIT:
            new_solution = {v.VarName: v.X for v in model.getVars()}
            
            new_obj_value = evaluate_solution_with_original_objective(model, ps_data)

            print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')

            if new_obj_value < current_best_obj or acceptance_probability(current_best_obj, new_obj_value, temperature) > random.random():
                current_solution = new_solution
                if new_obj_value < current_best_obj:
                    current_best_obj = new_obj_value
                    print(f"New best solution with objective {current_best_obj} at temperature {temperature}")
                #current_solution = new_solution  # Update current solution regardless of improvement for exploration
                print(f"Accepted new solution at temperature {temperature}")
            else:
                print(f"Rejected new solution at temperature {temperature}")

        # Optionally, remove the cutoff constraint for the next iteration to allow exploration
        #model.remove(model.getConstrByName("cutoff_constraint"))
        #model.update()

    return current_solution, current_best_obj
