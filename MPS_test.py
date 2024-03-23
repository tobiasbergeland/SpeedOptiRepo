import gurobipy as gp
from gurobipy import GRB
from gurobipy import read

INSTANCE = "LR1_DR02_VC02_V6a"

# Create a new model from the MPS file
model = read(f"Group_2/{INSTANCE}/{INSTANCE}_t120.mps")

# fixed_variables = {}
# for var in model.getVars():
#     if "endingSupplyInRegion" in var.VarName:
#         if var.lb == var.ub:  # If lower bound is equal to upper bound
#             fixed_variables[var.VarName] = var.lb

# # Print the fixed endingSupplyInRegion variables and their values
# for var_name, value in fixed_variables.items():
#     print(f"{var_name} is fixed to {value}")

x_vars_counter = 0
for var in model.getVars():
    if var.VarName.startswith("x"):
        x_vars_counter += 1
print(f"Number of x variables: {x_vars_counter}")

# Do same for a
a_vars_counter = 0
for var in model.getVars():
    if var.VarName.startswith("a"):
        a_vars_counter += 1
print(f"Number of a variables: {a_vars_counter}")

s_vars_counter = 0
for var in model.getVars():
    if var.VarName.startswith("s"):
        s_vars_counter += 1
print(f"Number of s variables: {s_vars_counter}")

ending_supply_vars_counter = 0
for var in model.getVars():
    if "endingSupplyInRegion" in var.VarName:
        ending_supply_vars_counter += 1
print(f"Number of endingSupplyInRegion variables: {ending_supply_vars_counter}")

# # print all vars not starting with x, a or s
# for var in model.getVars():
#     if not var.VarName.startswith("x") and not var.VarName.startswith("a") and not var.VarName.startswith("s"):
#         print(var.VarName)
        
# # Look for constraints that use a specific variable
variable_name = "endingSupplyInRegion"
constraints_using_variable = []

for constr in model.getConstrs():  # Loop over all constraints
    expr = model.getRow(constr)  # Get the linear expression for this constraint
    # Loop through each term in the linear expression
    for i in range(expr.size()):
        # Get the variable associated with this term
        var = expr.getVar(i)
        # Check if the variable name contains the variable_name we're looking for
        if variable_name in var.VarName:
            constraints_using_variable.append(constr.ConstrName)
            break  # Stop checking other terms in this constraint

# # Print the constraints that use the 'endingSupplyInRegion' variable
# print(constraints_using_variable)
    
# Print all constraints
model.printStats()

# Print all the categories of constraints
    

# Loop through the constraints and print them
for con in model.getConstrs():
    if con.ConstrName.startswith("flow"):
        continue
    print(f"Constraint: {con.ConstrName}")
    
    expr = model.getRow(con)  # Get the linear expression for the constraint
    # The getVar method returns the variable associated with a particular coefficient
    var_list = model.getVars()
    # expr.getCoeff(i) will get the coefficient for the ith variable in the expression
    for i in range(expr.size()):
        var = var_list[expr.getVar(i).index]  # Get the variable object
        coeff = expr.getCoeff(i)  # Get the coefficient for that variable
        print(f"   {var.VarName} {coeff}")


# Optimize the model
model.optimize()

# Check the optimization status
if model.Status == gp.GRB.OPTIMAL:
    print('Optimal solution found.')
    
    
with open(f'{INSTANCE}_t120.sol', 'w') as file:
    file.write("Variable Values:\n")
    x_vars_counter = 0
    for var in model.getVars():
        if var.VarName.startswith("x"):
            x_vars_counter += 1
        file.write(f"{var.varName}: {var.x}\n")
            
file.close()
    
