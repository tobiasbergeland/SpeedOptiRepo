import networkx as nx
import numpy as np
import math
import random

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
        operation_type = 0  # Assume no operation by default
        vessel_id = vessel.number

        # Determine the operation type based on the vessel's position and the state's time
        if state['time'] == 0 or vessel.position is None:
            pass  # operation_type remains 0
        elif vessel.position.isLoadingPort == 1:
            operation_type = 1  # Loading at a production port
        else:
            operation_type = 2  # Offloading at a consumption port

        # Get legal quantities and arcs for the vessel
        legal_quantities = self.get_legal_quantities(vessel, state)
        legal_arcs = self.get_legal_arcs(state=state, vessel=vessel)

        # Generate legal actions, considering special conditions
        legal_actions = set()
        for quantity in legal_quantities:
            for arc in legal_arcs:
                # Skip actions that involve operating and waiting at the same port with non-zero quantity
                if arc.origin_node.port == arc.destination_node.port and quantity != 0:
                    continue

                # Handling for vessels at consumption ports or with specific inventory conditions
                if vessel.position and vessel.position.isLoadingPort == -1 and vessel.inventory == quantity:
                    if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node.port == self.SINK_NODE.port:
                        legal_actions.add((vessel_id, operation_type, quantity, arc))
                else:
                    legal_actions.add((vessel_id, operation_type, quantity, arc))

        # Include a default action for finished vessels or those without a position
        if vessel.isFinished or vessel.position is None:
            last_arc = vessel.action_path[-1][3] if vessel.action_path else None
            if last_arc:  # Ensure there is a last action to refer to
                legal_actions.add((vessel_id, 0, 0, last_arc))

        # Warn if no legal actions are found (optional, could also raise an exception or handle differently)
        if not legal_actions:
            print('No legal actions found for the vessel.')

        return legal_actions

    
    def get_legal_arcs(self, state, vessel):
        if vessel.position is None or vessel.isFinished:
            legal_arcs = set()
            legal_arcs.add(vessel.action_path[-1][3])
            return legal_arcs
    
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
                
        # # If vessel is at a consumption port and is empty, it has to either travel to a loading port or to the sink
        # if current_port.isLoadingPort == -1 and vessel.inventory == 0:
        #     # Remove all arcs from legal_arcs that do not lead to a loading port or the sink
        #     legal_arcs = {arc for arc in legal_arcs if arc.destination_node.port.isLoadingPort == 1 or arc.destination_node.port == self.SINK_NODE.port}
                
        return legal_arcs
                
    def get_legal_quantities(self, vessel, state):
        time = state['time']
        sinkPort = self.SINK_NODE.port
        
        if time == 0 or vessel.position is None or vessel.position == sinkPort:
            # Return a set containing only 0
            return {0}
        else:
        
            port = vessel.position
            legal_quantities = set()
            
            # Calculate available capacity or inventory based on port type.
            if port.isLoadingPort == 1:
                # For loading ports, calculate the maximum quantity that can be loaded onto the vessel.
                limit_quantity = max(0, port.inventory - port.min_amount)
                available_capacity = vessel.max_inventory - vessel.inventory
            else:
                # For discharging ports, calculate the maximum quantity that can be unloaded from the vessel.
                limit_quantity = max(0, port.capacity - port.inventory)
                available_capacity = vessel.inventory
            
            # Determine legal quantities based on available capacity and port limits.
            full_quantity = min(available_capacity, limit_quantity)
            half_quantity = math.ceil(full_quantity / 2)
            
            # Add quantities to the legal actions set, ensuring they do not exceed port or vessel constraints.
            # Only add quantities greater than 0.
            if half_quantity > 0:
                legal_quantities.add(half_quantity)
            else:
                print('Issue')
            if full_quantity > half_quantity:
                legal_quantities.add(full_quantity)
            else:
                print('Issue')
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
        
    def check_state(self, state):
        '''Evaluates the state and returns status and reward.'''
        reward = 1  # Default reward for feasible states
        if self.is_infeasible(state):
            # Implement logic for penalties
            state['done'] = True
            reward = -1000  # Example large negative reward for infeasibility
            return state, reward
        if self.is_terminal(state):
            state['done'] = True
            reward = 100000
            # Additional logic for terminal rewards could go here
            return state, reward
        if reward == 0 and state['time'] >= 44:
            print('Here')
        # You could add more nuanced rewards/penalties based on operational efficiency here
        return state, reward
    
    def execute_action(self, state, vessel, action):
        # Action is on the form (vessel_id, operation_type, quantity, arc)
        # Execute the action and update the state
        vessel_id, operation_type, quantity, arc = action
        
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
        for vessel in actions.keys():
                self.execute_action(state, vessel, actions[vessel])
                
        # Consumption happens
        state = self.consumption(state)
        
        # Check if state is infeasible or terminal
        next_state, reward = self.check_state(state)
        return next_state, reward
    
class BasicRLAgent:
  
    def __init__(self):
        pass
        
    def select_action(self, state, legal_actions):
        # Placeholder for random selection among legal actions for a given state
        # Transform the legal_actions set into a list
        legal_actions = list(legal_actions)
        
        return random.choice(legal_actions)

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
    print(env.encode_state(env.state))
    agent = BasicRLAgent()

    num_episodes = 1000  # for example
    feasible_counter = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Perform initial actions to get the environment started
        legal_actions = {}
        for vessel in vessels:
            legal_actions[vessel] = env.find_legal_actions_for_vessel(state, vessel)
        
        actions = {}
        for vessel in vessels:
            action = agent.select_action(state, legal_actions[vessel])
            actions[vessel] = action
        
        for vessel in vessels:
            env.execute_action(state, vessel, actions[vessel])
            
        # All vessels have made their initial action.
        # Now we can start the main loop
                
        while not done:
            # Increase time and make production ports produce.
            state = env.increment_time_and_produce(state)
            
            # Check if state is infeasible or terminal        
            state, reward = env.check_state(state)
            if reward == 1000:
                print(f"Episode {episode}: Total Reward = {total_reward}")
            
            # Production ports have produced, and vessels have moved one time step closer to their destination.    
            # Find the vessels that are available to perform an action
            updated_vessels = env.update_vessel_status(state)
            # available_vessels = env.find_available_vessels(state)
                    
            legal_actions={}
            for vessel in updated_vessels:
                legal_actions[vessel] = env.find_legal_actions_for_vessel(state, vessel)
                
            actions = {}
            for vessel in vessels:
                action = agent.select_action(state, legal_actions[vessel])
                actions[vessel] = action
            
            next_state, reward = env.step(state, actions)
            
            # agent.update_policy(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = state['done']
            if done:
                print(f"Episode {episode}: Total Reward = {total_reward}")
                if total_reward > 0:
                    feasible_counter += 1
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
    print(f"Feasible solutions found: {feasible_counter}/{num_episodes}")

    
        
if __name__ == "__main__":
    main()
        
        
    
    