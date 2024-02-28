from MIRPSO_M import build_model
from MIRPSO_M import build_problem
from proximity_search import perform_proximity_search
from MIRPSO_M import find_initial_solution
from MIRPSO_M import solve_model
from MIRPSO_M import build_simplified_RL_model
from MIRPSO_M import visualize_network_for_vessel

import json
import gurobipy as gp


def save_solution(solution, filename='initial_solution.json'):
    with open(filename, 'w') as f:
        json.dump(solution, f)
        
def load_solution(filename='initial_solution.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V9¨a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V11a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12a'
    # INSTANCE = 'LR1_2_DR1_3_VC2_V6a'
    # INSTANCE = 'LR1_2_DR1_3_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12b'
    # INSTANCE = 'LR2_11_DR2_22_VC3_V6a'
    # INSTANCE = 'LR2_11_DR2_33_VC4_V11a'

    problem_data = build_problem(INSTANCE)
    
    # Unpack the problem data
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    non_operational = problem_data['non_operational']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    OPERATING_COST = problem_data['OPERATING_COST']
    
    model, costs = build_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST)

    x_initial_solution, model = find_initial_solution(model)
    # Print the size of the initial solution
    print("Initial solution size:", len(x_initial_solution))
    
    # Remove the solution limit
    model.setParam(gp.GRB.Param.SolutionLimit, 2000000)
    
    ps_data = {'model': model, 'initial_solution':x_initial_solution, 'costs': costs, 'regularNodes': regularNodes, 'vessels': vessels, 'operating_cost':OPERATING_COST, 'vessel_arcs': vessel_arcs}

    # Perform the proximity search using the initial solution
    improved_solution, obj_value = perform_proximity_search(ps_data)

    # print("Final solution:", improved_solution)
    print("Objective value:", obj_value)
    
    
def main_no_proximity_search():
    INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V9a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V11a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12a'
    # INSTANCE = 'LR1_2_DR1_3_VC2_V6a'
    # INSTANCE = 'LR1_2_DR1_3_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12b'
    # INSTANCE = 'LR2_11_DR2_22_VC3_V6a'
    # INSTANCE = 'LR2_11_DR2_33_VC4_V11a'

    problem_data = build_problem(INSTANCE)
    
    # Unpack the problem data
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    non_operational = problem_data['non_operational']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    OPERATING_COST = problem_data['OPERATING_COST']
    OPERATING_SPEED = problem_data['OPERATING_SPEED']
    NODES = problem_data['NODES']
    
    
    model, costs = build_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST)
    # simp_model, env_data = build_simplified_RL_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES)
    
    #ps_data = {'model': model, 'costs': costs, 'regularNodes': regularNodes, 'vessels': vessels, 'operating_cost':OPERATING_COST, 'vessel_arcs': vessel_arcs}

    # Perform the proximity search using the initial solution
    solve_model(model)

    # for v in vessels:
    #     visualize_network_for_vessel(v, env_data['vessel_arcs'], False, NODES, sinkNode)
    

    # solve_model(model)
    # solve_model(simp_model)

if __name__ == "__main__":
    main()
    #main_no_proximity_search()

