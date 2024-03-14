import gurobipy as gp
from gurobipy import GRB
from gurobipy import read


# Create a new model from the MPS file
model = read("/Users/tobiaskolstobergeland/Documents/IndØk/10.Semester/ProsjektOppgave/Repo/SpeedOptiRepo/Group_2/LR1_DR02_VC01_V6a/LR1_DR02_VC01_V6a_t120.mps")

# Optimize the model
model.optimize()

# Check the optimization status
if model.Status == gp.GRB.OPTIMAL:
    print('Optimal solution found.')
    
    
with open('LR1_DR02_VC01_V6a_t120.sol', 'w') as file:
    file.write("Variable Values:\n")
    for var in model.getVars():
        file.write(f"{var.varName}: {var.x}\n")
file.close()
    
