import copy
import gc
import sys
import torch

from common_definitions import MIRPSOEnv, DQNAgent, ReplayMemory, DQNAgent
from optimization_utils import *
from MIRP_GROUP_2 import (build_problem, build_model, solve_model, rearrange_arcs)

def evaluate_agent_until_solution_is_found(env, agent):
    agent.epsilon = 0.1
    attempts = 1
    exploit = False
    
    while True:
        experience_path = []
        state = env.reset()
        done = False
        
        port_inventory_dict = {}
        decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
        
        actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
        state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
        
        while not done:
            # Increment time and log
            state['time'] += 1
            port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
            
            state = env.produce(state)
                    
            # Check if state is infeasible or terminal        
            state, total_reward_for_path, feasible_path = env.check_state(state=state, experience_path=experience_path, replay=agent.memory, agent=agent)
            
            if state['done']:
                first_infeasible_time, infeasibility_counter = env.log_episode(attempts, total_reward_for_path, experience_path, state)
                feasible_path = experience_path[0][6]
                if not feasible_path:
                    break
                else:
                    return experience_path, port_inventory_dict
                
            # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
            env.update_vessel_status(state=state)
            # Find the vessels that are available to perform an action
            available_vessels = env.find_available_vessels(state=state)
            
            if available_vessels:
                actions = {}
                decision_basis_states = {}
                decision_basis_state = env.custom_deep_copy_of_state(state)
                for vessel in available_vessels:
                    corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                    decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                    legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel, queued_actions=actions)
                    action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=exploit)
                    _, _, _, _arc = action
                    actions[vessel] = action
                # Perform the operation and routing actions and update the state based on this
                state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            else:
                # Should check the feasibility of the state, even though no actions were performed. 
                state = env.simple_step(state, experience_path)
            
            state = env.consumption(state)
            
        print(f"Attempt {attempts} failed. Retrying...")
        attempts += 1
        

# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output
# def train_from_pre_populated_buffer(env, agent, num_episodes):
#     TRAINING_FREQUENCY = 1
#     TARGET_UPDATE_FREQUENCY = 100
#     EVALUATION_FREQUENCY = 100
#     first_infeasible_times = []
#     infeasibility_counters = []
#     plt.figure(figsize=(10, 6))
#     ax1 = plt.gca()  # Get the current axis
#     ax2 = ax1.twinx()  # Create another axis that shares the same x-axis
    
#     for episode in range(1, num_episodes + 1):
#         # Training the agent's network with a batch from the replay buffer.
#         if episode % TRAINING_FREQUENCY == 0:
#             agent.train_main_network(env)

#         # Updating the target network
#         if episode % TARGET_UPDATE_FREQUENCY == 0:
#             agent.update_target_network()
#             print('Target network updated')
        
#         if episode % EVALUATION_FREQUENCY == 0:
            
#             # first_inf_time,infeasibility_counter, _, _, _ = evaluate_agent(env=env, agent=agent)
#             experience_path, port_inventory_dict, vessel_inventory_dict = evaluate_agent(env=env, agent=agent)
            
#             state, action, vessel, reward, next_state, earliest_vessel, feasible_path, first_infeasible_time, terminal_flag = experience_path[0]
#             if feasible_path:
#                 print('Feasible path')
#             else:
#                 first_inf_time = first_infeasible_time
                
#             infeasibility_counter = 0
#             infeasibility_dict = {}
#             for exp in experience_path:
#                 current_state = exp[0]
#                 # result_state = exp[4]
#                 time = current_state['time']
#                 if current_state['infeasible']:
#                     if time not in infeasibility_dict.keys():
#                         infeasibility_dict[time] = True
                        
#             # Check also the final state
#             if state['infeasible']:
#                 infeasibility_dict[len(env.TIME_PERIOD_RANGE)] = True
                
#             infeasibility_counter = len(infeasibility_dict.keys())
            
#             if first_inf_time is None:
#                 first_inf_time = env.TIME_PERIOD_RANGE[-1]
            
#             first_infeasible_times.append(first_inf_time)
#             infeasibility_counters.append(infeasibility_counter)
#             clear_output(wait=True)  # Clear the previous output
#             ax1.plot(first_infeasible_times, '-ob', label='First Infeasible Time')
#             ax2.plot(infeasibility_counters, '-or', label='Infeasibility Counter')
#             if episode == EVALUATION_FREQUENCY:  # Add the legend only once
#                 ax1.legend(loc='upper left')
#                 ax2.legend(loc='upper right')
#             ax1.set_title('Development of First Infeasible Times and Infeasibility Counters Over Episodes')
#             ax1.set_xlabel('Episode')
#             ax1.set_ylabel('First Infeasible Time', color='b')
#             ax2.set_ylabel('Infeasibility Counter', color='r')
#             display(plt.gcf())  # Display the current figure
#             plt.pause(0.001)  # Pause for a short period to allow the plot to update
#     plt.close()  # Close the figure to prevent additional output
            
def unpack_problem_data(problem_data):
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    arc_dict = problem_data['arc_dict']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    NODES = problem_data['NODES']
    NODE_DICT = problem_data['NODE_DICT']
    VESSEL_CLASSES = problem_data['VESSEL_CLASSES']
    vessel_class_capacities = problem_data['vessel_class_capacities']
    special_sink_arcs = problem_data['special_sink_arcs']
    special_nodes_dict = problem_data['special_nodes_dict']
    return vessels, vessel_arcs, arc_dict, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, waiting_arcs, NODES, NODE_DICT, VESSEL_CLASSES, vessel_class_capacities, special_sink_arcs, special_nodes_dict

def unpack_env_data(env_data):
    vessels = env_data['vessels']
    vessel_arcs = env_data['vessel_arcs']
    regularNodes = env_data['regularNodes']
    ports = env_data['ports']
    TIME_PERIOD_RANGE = env_data['TIME_PERIOD_RANGE']
    non_operational = env_data['non_operational']
    sourceNode = env_data['sourceNode']
    sinkNode = env_data['sinkNode']
    waiting_arcs = env_data['waiting_arcs']
    OPERATING_COST = env_data['OPERATING_COST']
    OPERATING_SPEED = env_data['OPERATING_SPEED']
    ports_dict = env_data['ports_dict']
    NODE_DICT = env_data['node_dict']
    vessel_dict = env_data['vessel_dict']
    return vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, ports_dict, NODE_DICT, vessel_dict


import random
def main(FULLSIM, TRAIN_AND_SAVE_ONLY):
    LOG =True
    RUNNING_MIRPSO =False
    # RUNNING_MIRPSO =True
    
    # Set a higher recursion limit (be cautious with this)
    sys.setrecursionlimit(100000) 
    # random.seed(1)
    gc.enable()
    
    # INSTANCE = 'LR1_DR02_VC01_V6a'
    # INSTANCE = 'LR1_DR02_VC02_V6a'
    # INSTANCE = 'LR1_DR02_VC03_V7a'
    # INSTANCE = 'LR1_DR02_VC04_V8a'
    # INSTANCE = 'LR1_DR02_VC05_V8a'
    # INSTANCE = 'LR1_DR03_VC03_V10b'
    
    # 'Trene agent på desse. mange iterasjoner.'
    INSTANCE = 'LR1_DR02_VC03_V8a'
    # INSTANCE = 'LR1_DR05_VC05_V25a'
    # INSTANCE = 'LR1_DR08_VC10_V40a'
    
    
    TRAINING_FREQUENCY = 1
    TARGET_UPDATE_FREQUENCY = 50
    NON_RANDOM_ACTION_EPISODE_FREQUENCY = 5
    BATCH_SIZE = 500
    BUFFER_SAVING_FREQUENCY = 200
    problem_data = build_problem(INSTANCE, RUNNING_MIRPSO)
    
    vessels, vessel_arcs, arc_dict, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, waiting_arcs, NODES, NODE_DICT, VESSEL_CLASSES, vessel_class_capacities, special_sink_arcs, special_nodes_dict = unpack_problem_data(problem_data)
    
    for v in vessels:
        print(f'Vessel {v.number} start in {v.initial_port} at time {v.first_time_available}')
    
    origin_node_arcs, destination_node_arcs, vessel_class_arcs = rearrange_arcs(arc_dict=arc_dict)
    
    m, costs, P, costs_namekey = build_model(vessels = vessels, regularNodes=regularNodes, ports = ports, TIME_PERIOD_RANGE = TIME_PERIOD_RANGE, sourceNode = sourceNode, sinkNode = sinkNode,
                                             vessel_classes = VESSEL_CLASSES,origin_node_arcs = origin_node_arcs, destination_node_arcs = destination_node_arcs, vessel_class_arcs = vessel_class_arcs,
                                             NODE_DICT = NODE_DICT, vessel_class_capacities = vessel_class_capacities)
    
    env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode, NODE_DICT, special_sink_arcs, special_nodes_dict)
    replay = ReplayMemory(10000)
    agent = DQNAgent(ports = ports, vessels=vessels, TRAINING_FREQUENCY = TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY = TARGET_UPDATE_FREQUENCY, NON_RANDOM_ACTION_EPISODE_FREQUENCY = NON_RANDOM_ACTION_EPISODE_FREQUENCY, BATCH_SIZE = BATCH_SIZE, replay = replay)
    
    # '''Load main and target model.'''
    # agent.main_model.load_state_dict(torch.load('main_model.pth'))
    # agent.target_model.load_state_dict(torch.load('target_model.pth'))
    
    if not FULLSIM:
        replay = agent.load_replay_buffer(file_name= 'replay_buffer_8apr_nt3_50_1664.pkl')
        replay.capacity = 5000
        replay = replay.clean_up()
        # replay = agent.load_replay_buffer(file_name= 'replay_buffer_new_reward_policy_5000.pkl')
        agent.memory = replay
        
        agent.main_model.load_state_dict(torch.load('main_model_8apr_nt3_50_1664.pth'))
        agent.target_model.load_state_dict(torch.load('target_model_8_apr_nt3_50_1664.pth'))
        # train_from_pre_populated_buffer(env, agent, 5000)
        
    else:
        num_feasible_paths_with_random_actions = 0
        NUM_EPISODES = 2000
        # replay = agent.load_replay_buffer(file_name='replay_buffer_LR1_DR02_VC03_V8a_5000_NEW2.pkl')
        # replay.capacity = 5000
        # replay = replay.clean_up()
        # agent.memory = replay
        
        # agent.main_model.load_state_dict(torch.load('main_model_LR1_DR02_VC03_V8a_5000_NEW2.pth'))
        # agent.target_model.load_state_dict(torch.load('target_model_LR1_DR02_VC03_V8a_5000_NEW2.pth'))
        
        for episode in range(1, NUM_EPISODES):
            if episode % NON_RANDOM_ACTION_EPISODE_FREQUENCY == 0 or episode in {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}:
                exploit = True
                print(f"NON Random Action Episode: {episode}")
                num_feasible_paths_with_random_actions = 0
            else:
                exploit = False
            
            if episode % agent.TRAINING_FREQUENCY == 0:
                agent.train_main_network(env)
                
            if episode % agent.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print('Target network updated')
                gc.collect()
                
            port_inventory_dict = {}
            # vessel_inventory_dict = {}
                        
            experience_path = []
            state = env.reset()
            acc_alpha = {port.number : 0 for port in state['ports'] if port.rate}
            done = False
            
            # Directly create and fill decision_basis_states with custom deep-copied states for each vessel
            decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
            # We know that each vessel have only one legal action in the initial state
            actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel)[0] for vessel in state['vessels']}
            state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
            
            state['time'] += 1
            
            # All vessels have made their initial action.
            while not done:
                # Increase time and make production ports produce.
                # increment time only
                # Check if state is terminal
                # if env.is_infeasible(state):
                    # print(state['time'])
                
                state, total_reward_for_path, feasible_path, alpha_register = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent, INSTANCE=INSTANCE, exploit=exploit, port_inventory_dict=port_inventory_dict)
                
                port_inventory_dict[state['time']] = {port.number: port.inventory for port in state['ports']}
                
                acc_alpha = adjust_for_alpha(acc_alpha, port_inventory_dict, state)
                
                if state['done']:
                    if feasible_path:
                        num_feasible_paths_with_random_actions += 1
                    if episode % NON_RANDOM_ACTION_EPISODE_FREQUENCY == 0 or LOG:
                        env.log_episode(episode, total_reward_for_path, experience_path, state, port_inventory_dict, alpha_register)
                    break
                
                # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
                env.update_vessel_status(state=state)
                # Find the vessels that are available to perform an action
                available_vessels = env.find_available_vessels(state=state)
            
                # If some vessels are available, select actions for them
                if available_vessels:
                    # Randomize the order of the vessels
                    # random.shuffle(available_vessels)
                    actions = {}
                    decision_basis_states = {}
                    decision_basis_state = env.custom_deep_copy_of_state(state)
                    for vessel in available_vessels:
                        corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                        decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                        legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel, queued_actions=actions, RUNNING_WPS=False, acc_alpha=acc_alpha)
                        action, decision_basis_state = agent.select_action(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, exploit=exploit)
                        _, _, _, _arc = action
                        actions[vessel] = action
                    # Perform the operation and routing actions and update the state based on this
                    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
                else:
                    # Should check the feasibility of the state, even though no actions were performed. 
                    state = env.simple_step(state, experience_path)
                    
                state = env.produce(state)
                
                # Make consumption ports consume regardless if any actions were performed
                
                state = env.consumption(state)
                
                state['time'] += 1
            
            if episode % BUFFER_SAVING_FREQUENCY == 0:
                # agent.save_replay_buffer(file_name=f"replay_buffer_{INSTANCE}_{episode}_NEW2.pkl")
                torch.save(agent.main_model.state_dict(), f'main_model_{INSTANCE}_{episode}_NEW2.pth')
                torch.save(agent.target_model.state_dict(), f'target_model_{INSTANCE}_{episode}_NEW2.pth')
                
    if TRAIN_AND_SAVE_ONLY:
        return
    
    # When agent is done training. Let the agent solve the problem, and return the solution
    # first_infeasible_time, infeasibility_counter, experience_path, port_inventory_dict, vessel_inventory_dict  = evaluate_agent(env, agent)
    experience_path, port_inventory_dict  = evaluate_agent_until_solution_is_found(env, agent)
    active_X_keys, S_values, alpha_values = convert_path_to_MIRPSO_solution(env, experience_path, port_inventory_dict)
    env.reset()
    
    x_initial_solution, model, warm_start_sol = warm_start_model(model, active_X_keys, S_values, alpha_values)

    ps_data = {'model': model, 'initial_solution':x_initial_solution,'costs': costs, 'regularNodes': regularNodes, 'vessels': vessels, 'vessel_arcs': vessel_arcs}
    
    violated_constraints = []
    for constr in model.getConstrs():
        expr = model.getRow(constr)  # This gives you the LinExpr of the constraint
        constr_value = 0

        # Iterating directly over the LinExpr to get variables and coefficients
        for i in range(expr.size()):
            var = expr.getVar(i)  # Get variable at position i
            coeff = expr.getCoeff(i)  # Get coefficient at position i
            constr_value += coeff * warm_start_sol[var.VarName]

        # Now, compare constr_value with the constraint's RHS, considering its sense (<, >, =)
        if constr.Sense == '<':
            if constr_value > constr.RHS + model.Params.FeasibilityTol:  # Considering feasibility tolerance
                violated_constraints.append(constr.ConstrName)
        elif constr.Sense == '>':
            if constr_value < constr.RHS - model.Params.FeasibilityTol:
                violated_constraints.append(constr.ConstrName)
        elif constr.Sense == '=':
            if abs(constr_value - constr.RHS) > model.Params.FeasibilityTol:
                violated_constraints.append(constr.ConstrName)

    if violated_constraints:
        print("Violated Constraints:", violated_constraints)
    else:
        print("No violated constraints identified with the warm start solution.")
        
        
    # Print the values of all the alpha variables
    for var in model.getVars():
        # if value is not 0, print the value
        if var.Start > 0:
            print(f"{var.VarName}: {var.Start}")

    # model.setParam(gp.GRB.Param.SolutionLimit, 1)
    model.optimize()
    
    print('Solution found')
    print('Objective value:', model.objVal)
    
import sys

if __name__ == "__main__":
    FULL_SIM = False
    TRAIN_AND_SAVE_ONLY = False
    
    FULL_SIM = True
    TRAIN_AND_SAVE_ONLY = True
    
    
    
    main(FULL_SIM, TRAIN_AND_SAVE_ONLY)
    
    