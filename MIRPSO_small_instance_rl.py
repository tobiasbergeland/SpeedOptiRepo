import networkx as nx
import numpy as np
import math
import random
import copy

# sys.path.append('/Users/tobiaskolstobergeland/Documents/IndØk/10.Semester/ProsjektOppgave/Repo/SpeedOptiRepo/MIRPSO_M.py')
from MIRPSO_M import build_problem, build_simplified_RL_model, visualize_network_for_vessel


class MIRPSOEnv():
    def __init__(self, PORTS, VESSELS, VESSEL_ARCS, NODES, TIME_PERIOD_RANGE, SOURCE_NODE, SINK_NODE):
        # Ports have initial inventory and rate of consumption/production
        self.PORTS = PORTS
        self.VESSELS = VESSELS
        self.VESSEL_ARCS = VESSEL_ARCS
        self.NODES = NODES
        self.TIME_PERIOD_RANGE = TIME_PERIOD_RANGE
        self.SOURCE_NODE = SOURCE_NODE
        self.SINK_NODE = SINK_NODE
        
        self.INITIAL_PORT_INVENTORY= {}
        # Create a dictionary to store the initial inventory of each port. Use it later to reset the environment
        for port in self.PORTS:
            self.INITIAL_PORT_INVENTORY[port] = port.inventory
            
        # Do similarly for the vessels
        self.INITIAL_VESSEL_INVENTORY = {}
        for vessel in self.VESSELS:
            self.INITIAL_VESSEL_INVENTORY[vessel] = vessel.inventory
        
        self.vessel_paths = {}
        # Set the position of the vessels to the source node, and initialize an empty path for each vessel
        for vessel in self.VESSELS:
            vessel.position = self.SOURCE_NODE.port
            vessel.in_transit_towards = None
            self.vessel_paths[vessel] = vessel.action_path
            
        self.state = {
            'ports': self.PORTS, # Inventory
            'vessels': self.VESSELS, # Inventory, position, in_transit_towards, path
            # 'vessel_arcs': {vessel: vessel_arcs_v for vessel, vessel_arcs_v in self.VESSEL_ARCS.items()},
            # 'vessel_paths' : self.vessel_paths,
            # 'nodes': self.NODES,
            'time': 0,
            'done': False
        }
        
    def reset(self):
        # Reset the environment to an initial state
        # Initialize your state here
        # Ports, vessels, vessel_arcs, nodes and timeperiodrange is already initialized in the __init__ method. Only port- and vessel inventory and vessel position needs to be reset.
        for port in self.PORTS:
            port.inventory = self.INITIAL_PORT_INVENTORY[port]
        for vessel in self.VESSELS:
            vessel.inventory = self.INITIAL_VESSEL_INVENTORY[vessel]
            vessel.position = self.SOURCE_NODE.port
            vessel.in_transit_towards = None
            vessel.path = []
            vessel.action_path = []
            vessel.isFinished = False
            
        self.state = {
            'ports': self.PORTS,
            'vessels': self.VESSELS,
            # 'vessel_arcs': {vessel: vessel_arcs_v for vessel, vessel_arcs_v in self.VESSEL_ARCS.items()},
            # 'vessel_paths' : self.vessel_paths,
            # 'nodes': self.NODES,
            'time' : 0,
            'done' : False
        }
            
        return self.state
    
    def encode_state(self, state):
        port_inventories = np.array([(port.inventory)/port.capacity for port in state['ports']])
        vessel_inventories = np.array([(vessel.inventory)/(vessel.max_inventory) for vessel in state['vessels']])
        
        # Initialize arrays to store the position and in-transit status
        vessel_positions = np.zeros(len(state['vessels']))
        vessel_in_transit = np.zeros(len(state['vessels']))  # 0 if at port, 1 if in transit
        
        for i, vessel in enumerate(state['vessels']):
            if vessel.position:  # Vessel is at a port
                vessel_positions[i] = vessel.position.number / (len(self.PORTS) + 2)  # +2 to account for the sink and sourceport
            elif vessel.in_transit_towards:  # Vessel is in transit
                vessel_positions[i] = vessel.in_transit_towards[0].number / (len(self.PORTS) + 2)  # +1 for the same reason
                vessel_in_transit[i] = 1  # Mark as in transit
        
        time_period = np.array([self.state['time'] / max(self.TIME_PERIOD_RANGE)])
        # Concatenate all parts of the state into a single vector
        return np.concatenate([port_inventories, vessel_inventories, vessel_positions, vessel_in_transit, time_period])
    

    # def find_legal_actions_for_vessel(self, state, vessel):
        
    #     # If the vessel is in transit, return 
    #     vessel_id = vessel.number
    #     time = state['time']
    #     # If time is 0. The vessels are not allowed to operate. Only choice is to travel to their starting positions, or travel to sink.
    #     # If the vessel is in transit, only no-action is allowed
    #     if time == 0 or vessel.position is None:
    #         operation_type = 0
        
    #     elif vessel.position.isLoadingPort == 1:
    #         # Port is a production port. Loading onto vessels is operation_type 1
    #         operation_type = 1
    #     else:
    #         # Port is a consumption port. Offloading from vessels is operation_type 2
    #         operation_type = 2
            
    #     legal_quantities = self.get_legal_quantities(vessel, state)
    #     legal_arcs = self.get_legal_arcs(state = state, vessel = vessel)
        
    #     # Create all possible combinations of legal quantities and arcs
    #     legal_actions = set()
    #     for quantity in legal_quantities:
    #         for arc in legal_arcs:
    #             # Avoid adding actions where the vessel would like to operate and then traverse the waiting arc.
    #             if arc.origin_node.port == arc.destination_node.port and quantity != 0:
    #                 continue
    #             # If vessel is at a consumption port and is empty, it has to either travel to a loading port or to the sink
    #             if vessel.position is not None:
    #                 if vessel.position.isLoadingPort == -1 and vessel.inventory == quantity:
    #                     if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node.port == self.SINK_NODE.port:
    #                         legal_actions.add((vessel_id, operation_type, quantity, arc))
    #                 else:            
    #                     legal_actions.add((vessel_id, operation_type, quantity, arc))
                        
    #     if vessel.isFinished or vessel.position is None:
    #         legal_actions.add((vessel_id, 0, 0, vessel.action_path[-1][3]))
                
    #     if len(legal_actions) == 0:
    #         print('Issue ')
        
    #     return legal_actions
    
    
    def find_legal_actions_for_vessel(self, state, vessel):
        # Initialize the operation type and vessel ID
        # operation_type = 0  # Assume no operation by default
        vessel_id = vessel.number

        # Determine the operation type based on the vessel's position and the state's time
        if state['time'] == 0:
            operation_type = 0
        elif vessel.position.isLoadingPort == 1:
            operation_type = 1  # Loading at a production port
        else:
            operation_type = 2  # Offloading at a consumption port

        # Get legal quantities and arcs for the vessel
        legal_quantity = self.get_legal_quantities(operation_type=operation_type, vessel=vessel)
        legal_arcs = self.get_legal_arcs(state=state, vessel=vessel)

        # Generate legal actions, considering special conditions
        legal_actions = []
        
        for arc in legal_arcs:
            legal_actions.append((vessel_id, operation_type, legal_quantity, arc))

        # Warn if no legal actions are found (optional, could also raise an exception or handle differently)
        if not legal_actions:
            print('No legal actions found for the vessel.')

        return legal_actions

    
    def get_legal_arcs(self, state, vessel):
    
        vessel_arcs_v = self.VESSEL_ARCS[vessel]
        # Find the node the vessel is currently at
        current_port = vessel.position
        current_time = state['time']
        for node in self.NODES:
            if node.port == current_port and node.time == current_time:
                current_node = node
                break
            
        # Find the arcs that are legal for the vessel to traverse
        legal_arcs = set()
        for arc in vessel_arcs_v:
            if arc.origin_node == current_node and arc.destination_node.port != arc.origin_node.port:
                legal_arcs.add(arc)
                
        # If vessel is at a consumption port and is empty, it has to either travel to a loading port or to the sink
        if current_port.isLoadingPort == -1 and vessel.inventory == 0:
            # Remove all arcs from legal_arcs that do not lead to a loading port or the sink
            legal_arcs = {arc for arc in legal_arcs if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node.port == self.SINK_NODE.port}
        
        if current_port.isLoadingPort == 1 and vessel.inventory == vessel.max_inventory:
            # Remove all arcs from legal_arcs that do not lead to a consumption port or the sink
            legal_arcs = {arc for arc in legal_arcs if arc.destination_node.port.isLoadingPort == -1 or arc.destination_node.port == self.SINK_NODE.port}
                
        return legal_arcs
                
    def get_legal_quantities(self, operation_type, vessel):
        
        if operation_type == 0:
            # Return a set containing only 0
            return {0}
        
        else:
            port = vessel.position
            # Calculate available capacity or inventory based on port type.
            if port.isLoadingPort == 1:
                # For loading ports, calculate the maximum quantity that can be loaded onto the vessel.
                limit_quantity = port.inventory - port.min_amount
                available_capacity = vessel.max_inventory - vessel.inventory
            else:
                # For discharging ports, calculate the maximum quantity that can be unloaded from the vessel.
                limit_quantity = port.capacity - port.inventory
                available_capacity = vessel.inventory
            
            # Determine legal quantities based on available capacity and port limits.
            quantity = min(available_capacity, limit_quantity)
           
            return quantity
    
    def simulate_time_step(self, state):
        # Action pipeline is a list of actions not yet executed
        # Some things happen independent of actions:
        
        # 1. Time is incremented
        state['time'] += 1
        # 2. Ports consume/produce goods. Production happens before loading. Consumption happens after unloading.
        for port in state['ports']:
            if port.isLoadingPort:
                port.inventory += port.rate
                
        self.check_state(state)
        available_vessels = self.find_available_vessels(state)
                
        # The available vessels now need to do an action
        for vessel in available_vessels:
            legal_actions_v = self.find_legal_actions_for_vessel(state, vessel)
            if legal_actions_v is not None:
                action = self.select_action(legal_actions_v)
                self.execute_action(state, vessel, action)
                
    def update_vessel_status(self, state):
        time = state['time']
        updated_vessels = []            
        for vessel in state['vessels']:
            destination_port = vessel.in_transit_towards[0]
            destination_time = vessel.in_transit_towards[1]
            if time == destination_time:
                # Vessel has reached its destination
                # Update the vessel's position
                self.update_vessel_position(vessel, destination_port)
                updated_vessels.append(vessel)
            else:
                updated_vessels.append(vessel)
        return updated_vessels
                
        
    def is_infeasible(self, state):
        # Implement a check for infeasible states
        # For example, if the inventory of a port or vessel is negative, the state is infeasible
        for port in state['ports']:
            # Use 0 as a lower limit for inventory
            if port.inventory < 0 or port.inventory > port.capacity:
                return True
        for vessel in state['vessels']:
            if vessel.inventory < 0 or vessel.inventory > vessel.max_inventory:
                return True
        return False

    def is_terminal(self, state):
        if state['time'] == len(self.TIME_PERIOD_RANGE):
            return True
        
    def check_state(self, state, experience_path, replay, agent):
        '''Evaluates the state and returns status and reward.'''
        if self.is_infeasible(state):
            experience_path = self.update_rewards_in_experience_path(state, experience_path, -1000, agent)
            for exp in experience_path:
                replay.push(exp)
            state['done'] = True
            return state
        
        if self.is_terminal(state):
            self.update_rewards_in_experience_path(state, experience_path, 1000, agent)
            for exp in experience_path:
                replay.push(exp)
            state['done'] = True
            return state
        return state
    
    def update_rewards_in_experience_path(self, state, experience_path, reward, agent):
        for exp in experience_path:
            time = state['time']
            time_marked = exp[0]['time']
            time_diff = time - time_marked
            new_state = exp[3]
            exp[2] = reward + agent.gamma**time_diff*(agent.target_model(torch.FloatTensor(self.encode_state(new_state))).detach().numpy())
        return experience_path
            
            
    
    def execute_action(self, vessel, action):
        # Action is on the form (vessel_id, operation_type, quantity, arc)
        # Execute the action and update the state
        _, operation_type, quantity, arc = action
        
        port = arc.origin_node.port
        # Update the vessel's inventory
        if operation_type == 1:
            #Loading
            vessel.inventory += quantity
            port.inventory -= quantity
        elif operation_type == 2:
            #Unloading
            vessel.inventory -= quantity
            port.inventory += quantity
        
        # ROUTING
        # Update the vessel's position and in_transit_towards attributes
        vessel.position = None
        vessel.in_transit_towards = (arc.destination_node.port, arc.destination_node.time)
        vessel.action_path.append(action)
        if arc.destination_node.port == self.SINK_NODE.port:
            vessel.isFinished = True
        
    def update_vessel_position(self, vessel, new_position):
        # Update the vessel's position and in_transit_towards attributes
        vessel.position = new_position
        vessel.in_transit_towards = None
        
        
    def increment_time_and_produce(self, state):
        state['time'] += 1
        # 2. Ports consume/produce goods. Production happens before loading. Consumption happens after unloading.
        for port in state['ports']:
            if port.isLoadingPort == 1:
                port.inventory += port.rate
        return state
                
    def consumption(self, state):
        for port in state['ports']:
            if port.isLoadingPort == -1:
                port.inventory -= port.rate
        return state
                
    def step(self, state, actions):
        for vessel, action in actions.items():
                self.execute_action(vessel=vessel, actions=action)
                
        # Consumption happens
        state = self.consumption(state)
        
        # Check if state is infeasible or terminal
        state = self.check_state(state)
        return state
    

    def find_available_vessels(self, vessels):
        available_vessels = []
        for v in vessels:
            if v.position is not None:
                available_vessels.append(v)
        return available_vessels
    
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)  # First fully connected layer
        self.relu = nn.ReLU()                 # ReLU activation
        self.fc2 = nn.Linear(24, 24)          # Second fully connected layer
        self.fc3 = nn.Linear(24, action_size) # Output layer

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, event):
        self.memory.append(event)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
class DQNAgent:
    def __init__(self, ports, vessels):
        # ports plus source and sink, vessel inventory, (vessel position, vessel in transit), time period, vessel_number
        state_size = len(ports) + 3 * len(vessels) + 2
        
        # Ports plus sink port
        action_size = len(ports) + 1
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.main_model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.main_model.parameters())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy())
            target_f = self.model(torch.FloatTensor(state))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.FloatTensor(state)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def select_action(self, legal_actions, state, env):
        
        if np.random.rand() < self.epsilon:
            # Choose one of the legal actions at random
            return random.choice(legal_actions)
        
        encoded_state = env.encode_state(state)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.main_model(state_tensor).detach().numpy()
        
        # Sort the q-values, but keep track of the original indices
        q_values = [(q_value, index) for index, q_value in enumerate(q_values)]
        q_values.sort(reverse=True)
        # Choose the action with the highest q-value that is legal
        for q_value, index in q_values:
            for action in legal_actions:
                arc = action[3]
                if arc.destination_node.port.number == index + 1:
                    return action        
        
def main():
    # np.random.seed(0)
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
    
    
    simp_model, env_data = build_simplified_RL_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES)
    
    # for v in vessels:
    #     visualize_network_for_vessel(v, vessel_arcs, False, NODES, sinkNode)
    
    env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode)
    agent = DQNAgent(ports = ports, vessels=vessels)
    replay = ReplayMemory(10000)
    print(env.encode_state(env.state))

    num_episodes = 1000  # for example
    feasible_counter = 0
    for episode in range(num_episodes):
        experience_path = []
        
        state = env.reset()
        done = False
        total_reward = 0
        
        # Perform initial actions to get the environment started
        legal_actions = {}
        for vessel in vessels:
            legal_actions[vessel] = env.find_legal_actions_for_vessel(state=state, vessel=vessel)
        
        actions = {}
        for vessel in vessels:
            action = agent.select_action(legal_actions=legal_actions[vessel], state=state, env=env)
            actions[vessel] = action
        
        exp_dict ={}
        for vessel in vessels:
            old_state = state
            env.execute_action(vessel=vessel, action=actions[vessel])
            # Make a deep, not a shallow copy of the state
            state_copy = copy.deepcopy(state)
            fake_state = env.consumption(state=state_copy)
            fake_state['time'] += 1
            exp_dict[vessel] = (old_state, actions[vessel], None, fake_state)
            experience_path.append(exp_dict[vessel])
            
        env.consumption(state=state)
        
        # All vessels have made their initial action.
        # Now we can start the main loop
                
        while not done:
            # Increase time and make production ports produce.
            state = env.increment_time_and_produce(state=state)
            
            # Check if state is infeasible or terminal        
            state = env.check_state(state=state, experience_path=experience_path, replay=replay, agent=agent)
            if state['done']:
                break
            
            # Production ports have produced, and vessels have moved one time step closer to their destination.    
            # Find the vessels that are available to perform an action
            env.update_vessel_status(state=state)
            
            available_vessels = env.find_available_vessels(vessels=state['vessels'])
            
            if len(available_vessels) >= 0:
                legal_actions={}
                for vessel in available_vessels:
                    legal_actions[vessel] = env.find_legal_actions_for_vessel(state=state, vessel=vessel)
                    
                actions = {}
                for vessel in vessels:
                    action = agent.select_action(state=state, legal_actions=legal_actions[vessel], env=env)
                    actions[vessel] = action
            
                state = env.step(state=state, actions=actions)
            
            # agent.update_policy(state, action, reward, next_state)
            done = state['done']
            
            if done:
                print(f"Episode {episode}: Total Reward = {total_reward}")
                if total_reward > 0:
                    feasible_counter += 1
    
    # print(f"Episode {episode}: Total Reward = {total_reward}")
    # print(f"Feasible solutions found: {feasible_counter}/{num_episodes}")
    
import sys        
if __name__ == "__main__":
    # print(sys.prefix)
    main()
        
        
    
    