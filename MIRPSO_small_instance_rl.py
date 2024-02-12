import networkx as nx
import numpy as np
import math

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
            self.vessel_paths[vessel] = vessel.path
            
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
    

    def find_legal_actions_for_vessel(self, state, vessel):
        
        vessel_id = vessel.number
        
        # If the vessel is in transit, only no-action is allowed
        if vessel.position is None:
            operation_type = 0
        elif vessel.position.port.isLoadingPort:
            # Port is a production port. Loading onto vessels is operation_type 1
            operation_type = 1
        else:
            # Port is a consumption port. Offloading from vessels is operation_type 2
            operation_type = 2
            
        legal_quantities = self.get_legal_quantities(vessel)
        legal_arcs = self.get_legal_arcs(vessel)
        
        # Create all possible combinations of legal quantities and arcs
        legal_actions = set()
        for quantity in legal_quantities:
            for arc in legal_arcs:
                legal_actions.add((vessel_id, operation_type, quantity, arc))
        
        return legal_actions
    
    
    def get_legal_arcs(self, state, vessel):
        if vessel.position is None:
            return None
        
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
            if arc.origin_node == current_node:
                legal_arcs.add(arc)
                
        return legal_arcs
                
    def get_legal_quantities(self, vessel):
        port = vessel.position
        legal_quantities = set()
        
        # Calculate available capacity or inventory based on port type.
        if port.isLoadingPort:
            # For loading ports, calculate the maximum quantity that can be loaded onto the vessel.
            limit_quantity = max(0, port.inventory - port.min_amount)
            available_capacity = vessel.max_inventory - vessel.inventory
        else:
            # For discharging ports, calculate the maximum quantity that can be unloaded from the vessel.
            limit_quantity = max(0, port.max_amount - port.inventory)
            available_capacity = vessel.inventory
        
        # Determine legal quantities based on available capacity and port limits.
        full_quantity = min(available_capacity, limit_quantity)
        half_quantity = math.ceil(full_quantity / 2)
        
        # Add quantities to the legal actions set, ensuring they do not exceed port or vessel constraints.
        # Only add quantities greater than 0.
        if half_quantity > 0:
            legal_quantities.add(half_quantity)
        if full_quantity > half_quantity:
            legal_quantities.add(full_quantity)
        
        return legal_quantities
    
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
                
    def find_available_vessels(self, state):
        time = state['time']
        available_vessels = []            
        for vessel in state['vessels']:
            if vessel.position is None:
                continue
            else:
                destination_port = vessel.in_transit_towards[0]
                destination_time = vessel.in_transit_towards[1]
                if time == destination_time:
                    # Vessel has reached its destination
                    # Update the vessel's position
                    self.update_vessel_position(vessel, destination_port)
                    available_vessels.append(vessel)  
        return available_vessels
        
    def is_infeasible(self, state):
        # Implement a check for infeasible states
        # For example, if the inventory of a port or vessel is negative, the state is infeasible
        for port in state['ports']:
            if port.inventory < port.min_amount or port.inventory > port.max_amount:
                return True
        for vessel in state['vessels']:
            if vessel.inventory < 0 or vessel.inventory > vessel.max_inventory:
                return True
        return False

    def is_terminal(self, state):
        if state['time'] == max(self.TIME_PERIOD_RANGE):
            return True
        
    def check_state(self, state):
        '''Evaluates the state and returns status and reward.'''
        reward = 0  # Default reward for feasible states
        if self.is_infeasible(state):
            # Implement logic for penalties
            state['done'] = True
            reward = -100  # Example large negative reward for infeasibility
            return state, reward
        if self.is_terminal(state):
            state['done'] = True
            # Additional logic for terminal rewards could go here
            return state, reward
        # You could add more nuanced rewards/penalties based on operational efficiency here
        return state, reward

        
    # def select_action(self, legal_actions):
    #     # Implement your action selection logic
    #     # Choose a random action for starters
    #     action = np.random.choice(list(legal_actions))
    #     return action
    
    def execute_action(self, state, vessel, action):
        # OPERATION
        # Execute the action and update the state
        quantity, arc = action
        port = arc.origin_node.port
        # Update the vessel's inventory
        if port.isLoadingPort:
            #Loading
            vessel.inventory += quantity
            port.inventory -= quantity
        else:
            #Unloading
            vessel.inventory -= quantity
            port.inventory += quantity
            
        # ROUTING
        # Update the vessel's position and in_transit_towards attributes
        vessel.position = None
        vessel.in_transit_towards = (arc.destination_node.port, arc.destination_node.time)
        
    def update_vessel_position(self, vessel, new_position):
        # Update the vessel's position and in_transit_towards attributes
        vessel.position = new_position
        vessel.in_transit_towards = None



    
class BasicRLAgent:
    def __init__(self, action_space, learning_rate = 0.1, discount_factor = 0.95):
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
    def select_action(self, state, legal_actions):
        # Placeholder for random selection among legal actions for a given state
        return np.random.choice(legal_actions)

    def update_policy(self, state, action, reward, next_state, legal_actions):
        # Implement how your agent updates its policy based on feedback
        state_action = (state, action)

        #initialize Q value to 0 if the state-action pair has not been seen before
        if state_action not in self.q_table:
            self.q_table[state_action] = 0
        
        #Estimate the optimal future value
        next_state_action = [(next_state, a) for a in legal_actions]
        future_rewards = [self.q_table.get(s_a, 0) for s_a in next_state_action]
        max_future_reward = max(future_rewards) if future_rewards else 0

        self.q_table[state_action] = self.q_table[state_action] + self.learning_rate * (reward + self.discount_factor * max_future_reward - self.q_table[state_action])
        pass
    


def main():
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
    print(env.encode_state(env.state))
    agent = BasicRLAgent(action_space=env.action_space)

    num_episodes = 100  # for example
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            
            
            legal_actions={}
            for vessel in vessels:
                legal_actions[vessel] = env.find_legal_actions_for_vessel(state, vessel)
            
            actions = {}
            for vessel in vessels:
                action = agent.select_action(state, legal_actions[vessel])
                actions[vessel] = action
            
            
            for vessel in v
            next_state, reward, done = env.step(action)
            agent.update_policy(state, action, reward, next_state)
            state = next_state
            total_reward += reward
    
    print(f"Episode {episode}: Total Reward = {total_reward}")

        
    
        
if __name__ == "__main__":
    main()
        
        
    
    