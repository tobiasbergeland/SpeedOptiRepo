import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import math
import re
import json

class Port:
    def __init__(self, capacity, inventory, rate, price, berth_limit, port_fee, max_amount, min_amount, number, isLoadingPort):
        self.capacity = capacity
        self.inventory = inventory
        self.rate = rate
        self.price = price 
        self.berth_limit = berth_limit 
        self.port_fee = port_fee
        self.max_amount = max_amount
        self.min_amount = min_amount
        self.number = number
        self.isLoadingPort = isLoadingPort
        
    def __repr__(self):
        return f'Port {self.number}'
    
    def __repr2__(self):
        return f'Port {self.number}: Capacity = {self.capacity}, Inventory = {self.inventory}, Rate = {self.rate}, Price = {self.price}, Berth Limit = {self.berth_limit}, Port Fee = {self.port_fee}, Max Amount = {self.max_amount}, Min Amount = {self.min_amount}, is Loading Port = {self.isLoadingPort}'
        
class Node:
    def __init__(self, port, time):
        self.port = port
        self.time = time
        self.tuple = (port.number if port else None, time)
        self.incoming_arcs = set()
        self.outgoing_arcs = set()
        self.berths = port.berth_limit if port else None
    
    def __repr__(self):
        return str(self.tuple)

class Arc:
    def __init__(self, origin_node, destination_node, distance, cost, travel_time, speed, is_waiting_arc):
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.tuple = (origin_node, destination_node)
        self.distance = distance
        self.cost = cost
        self.travel_time = travel_time
        self.speed = speed
        self.is_waiting_arc = is_waiting_arc
            
    def __repr__(self):
        return f'{self.origin_node} -> {self.destination_node} --- Cost: {self.cost:.1f} --- Speed: {self.speed:.1f}'

class Vessel:
    def __init__(self, max_inventory, initial_inventory, max_operating_quantity, number):
        self.max_inventory = int(max_inventory)
        self.inventory = initial_inventory
        self.max_operating_quantity = max_operating_quantity
        self.number = number
        self.arcs = set()
        self.all_arcs_v = set()
        # Position is which port it is located at. If it is in transit, position is None
        self.position = None
        self.path = []
        # If a vessel is in transit towards a port, this attribute will store the destination port.
        # It will change to None when the vessel arrives at the port. The position attribute will also be updated to the value of the destination 
        self.in_transit_towards = (None, None)
        
    def __repr__(self):
        return f'Vessel {self.number}'

    def __repr2__(self):
        return f'Vessel {self.number}: Max Inventory = {self.max_inventory}, Inventory = {self.inventory}, Max Operating Quantity = {self.max_operating_quantity}'
           
def convert_to_correct_type(value, data_type):
    """Converts a string value to the specified data type."""
    try:
        if data_type == 'int':
            return int(value)
        elif data_type == 'float':
            return float(value)
        elif data_type == 'list':
            # Handles both comma-separated and space-separated lists
            items = value.strip('[]').split(',')
            return [int(item.strip()) for item in items if item]
    except ValueError:
        return None
    
def extract_metadata(content):
    """Extracts and converts metadata from a given content string."""
    metadata_keys = {
        'numPeriods': 'int', 'numCommodities': 'int', 'numLoadingRegions': 'int',
        'numDischargingRegions': 'int', 'numLoadingPortsInRegion': 'list',
        'numDischargingPortsInRegion': 'list', 'numVesselClasses': 'int',
        'numTermVesselsInClass': 'list', 'hoursPerPeriod': 'int',
        'spotMarketPricePerUnit': 'float', 'spotMarketDiscountFactor': 'float',
        'perPeriodRewardForFinishingEarly': 'float', 'attemptCost': 'float',
        'constantForSinglePeriodAlphaSlack': 'float', 'constantForCumulativeAlphaSlack': 'float'
    }
    metadata = {}
    start_index = content.index("----- MetaData -----") + len("----- MetaData -----")
    end_index = content.find("\n\n", start_index) if "\n\n" in content[start_index:] else len(content)
    metadata_section = content[start_index:end_index].strip().split("\n")
    
    for line in metadata_section:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in metadata_keys:
                metadata[key] = convert_to_correct_type(value.strip(), metadata_keys[key])
    return metadata

def adjust_vessel_classes(metadata):
    """Adjusts metadata for a single vessel class scenario."""
    metadata['numVesselClasses'] = 1
    vessels_in_classes = metadata['numTermVesselsInClass']
    metadata['numTermVesselsInClass'] = [sum(vessels_in_classes)]
    return metadata

def get_vesels_data(VESSELINFO_PATH):
    ### Read the vessel data
    with open(VESSELINFO_PATH, 'r') as file:
        file_content = file.read()

    # Regular expression to extract vessel information including first time available
    pattern = r"name\s+Vessel_(\d+)\s+.*?initialInventory\s+(\d+)\s+initialPort\s+([\w_]+)\s+firstTimeAvailable\s+(\d+)"

    # Finding all matches
    vessel_info = re.findall(pattern, file_content, re.DOTALL)

    # Creating a dictionary to store the information
    vessel_data = {}
    for vessel in vessel_info:
        vessel_index = int(vessel[0])
        initial_inventory = int(vessel[1])
        initial_port = vessel[2]
        first_time_available = int(vessel[3])

        # Storing in dictionary
        vessel_data[f"Vessel_{vessel_index}"] = {
            "Initial Inventory": initial_inventory,
            "Initial Port": initial_port,
            "First Time Available": first_time_available
        }
    return vessel_data

### Read the port data
def parse_region_table(content):
    # Extract the region table section
    start_index = content.index("----- Region Table ----") + len("----- Region Table ----")
    end_index = content.find("-----", start_index)  # Find the next section separator
    region_section = content[start_index:end_index].strip().split("\n")[1:]  # Exclude the header line
    regions = {}
    for line in region_section:
        if "Note:" not in line:  # Exclude the note lines
            attribute, *values = line.split()
            regions[attribute] = values
    return regions

def parse_port_table_for_region(content, region_index):
    # Extract the port table section for the specified region
    search_str = f"----- Port Table For Region {region_index} ----"
    start_index = content.index(search_str) + len(search_str)
    end_index = content.find("-----", start_index)  # Find the next section separator
    port_section = content[start_index:end_index].strip().split("\n")[1:]  # Exclude the header line
    ports = {}
    for line in port_section:
        attribute, *values = line.split()
        ports[attribute] = values
    return ports

def extract_region_and_port_info(content):    
    # Extract region and port information
    # Read the content of the provided file
    regions_info = parse_region_table(content)
    ports_info = {f"Region {i}": parse_port_table_for_region(content, i) for i in range(len(regions_info['NumPorts']))}
    return regions_info, ports_info

def create_ports_from_info_with_loading(ports_info):
    all_ports = {}
    loading_regions = {}
    discharging_regions = {}

    tot_num = 1
    
    for region, port_attributes in ports_info.items():
        region_ports = []
        is_loading_region = all(int(rate) > 0 for rate in port_attributes['Rate'])
        is_discharging_region = all(int(rate) < 0 for rate in port_attributes['Rate'])

        for i in range(len(port_attributes['Capacity'])):  # Assuming 'Capacity' is always present
            rate = int(port_attributes['Rate'][i]) if 'Rate' in port_attributes else 0
            isLoading = 1 if rate > 0 else -1  # Loading port if rate is positive

            port = Port(
                capacity=int(port_attributes['Capacity'][i]),
                inventory=int(port_attributes['Inventory'][i]) if 'Inventory' in port_attributes else None,
                rate=abs(rate),
                price=int(port_attributes['Price'][i]) if 'Price' in port_attributes and port_attributes['Price'] else None,
                berth_limit=int(port_attributes['BerthLimit'][i]) if 'BerthLimit' in port_attributes else None,
                port_fee=int(port_attributes['PortFee'][i]) if 'PortFee' in port_attributes else None,
                max_amount=int(port_attributes['maxAmt'][i]) if 'maxAmt' in port_attributes else None,
                min_amount=int(port_attributes['minAmt'][i]) if 'minAmt' in port_attributes else None,
                number=tot_num,  # Using 1 to numports+1 as the port number
                isLoadingPort=isLoading)  # Determine loading port based on Rate value
            region_ports.append(port)
            tot_num += 1

        all_ports[region] = region_ports

        # Assign region to correct dictionary
        if is_loading_region:
            loading_regions[region] = region_ports
        elif is_discharging_region:
            discharging_regions[region] = region_ports

    return all_ports, loading_regions, discharging_regions

def get_ports(all_ports):
    # Create a list of all ports
    ports = []
    for region, region_ports in all_ports.items():
        ports.extend(region_ports)
    return ports

def parse_vessel_table(INSTANCE_PATH):
    with open(INSTANCE_PATH, 'r') as file:
        content = file.read()
    # Extract the vessel table section
    start_index = content.index("----- Vessel Table ----") + len("----- Vessel Table ----")
    end_index = content.find("-----", start_index)  # Find the next section separator
    vessel_section = content[start_index:end_index].strip().split("\n")[1:]  # Exclude the header line

    vessels = {}
    class_0_capacity = None

    # Extract vessel attributes
    for line in vessel_section:
        if "Note:" not in line:  # Exclude the note lines
            attribute, *values = line.split()
            vessels[attribute] = values

    # Find indices of Class 0 vessels
    class_0_indices = [i for i, v in enumerate(vessels.get("Class", [])) if v == "0"]

    # Extract the capacity for Class 0 vessels
    if class_0_indices:
        class_0_capacity_index = class_0_indices[0]
        class_0_capacity = int(vessels.get("Capacity", [])[class_0_capacity_index])

    return vessels, class_0_capacity

def get_vessels(metadata, vessel_data, VESSEL_CAP):    
    vessels = {}
    tot = 1
    for vessel_class in range(metadata['numVesselClasses']):
        vessel_list = []
        vessels_in_class = metadata['numTermVesselsInClass'][vessel_class]
        for i in range(vessels_in_class):
            if int(vessel_data['Vessel_'+str(i)]['Initial Inventory'])>0:
                init_inventory = VESSEL_CAP
            else:
                init_inventory = 0
                
            vessel_list.append(Vessel(
                max_inventory= VESSEL_CAP,
                initial_inventory= init_inventory,
                max_operating_quantity=VESSEL_CAP,
                number=tot
            ))
            tot += 1
        vessels[vessel_class] = vessel_list

    # We only have one vessel class. Convert the dictionary to a list
    vessels = vessels[0]
    return vessels

def create_regular_nodes(NUM_TIME_PERIODS, ports):
    # Create the regular nodes
    regularNodes = []
    for t in range(1, NUM_TIME_PERIODS+1):
        for port in ports:
            node = Node(port=port, time=t)
            regularNodes.append(node)
    return regularNodes

def create_special_ports(vessels, ports):
    # Create fictional source and sink port
    sourcePort = Port(capacity=None, inventory=None, rate=None, price=None, berth_limit=len(vessels), port_fee=0, max_amount=None, min_amount=None, number=0, isLoadingPort=True)
    sinkPort = Port(capacity=None, inventory=None, rate=None, price=None, berth_limit=len(vessels), port_fee=0, max_amount=None, min_amount=None, number=len(ports)+1, isLoadingPort=False)
    return sourcePort, sinkPort

def create_special_nodes(sourcePort, sinkPort, NUM_TIME_PERIODS):
    # Create source and sink node
    sourceNode = Node(port=sourcePort, time=0)
    sinkNode = Node(port=sinkPort, time=NUM_TIME_PERIODS+1)
    return sourceNode, sinkNode

def create_node_list_and_dict(sourceNode, regularNodes, sinkNode):
    NODES = [sourceNode] + regularNodes + [sinkNode]
    # Create a node dictionary with key = (port, time) tuple and value = node object
    NODE_DICT = {}
    for node in NODES:
        NODE_DICT[node.tuple] = node
    return NODES, NODE_DICT

### CREATE ARCS
def parse_full_distance_matrix(INSTANCE_PATH):
    with open(INSTANCE_PATH, 'r') as file:
        content = file.read()
    # Extract the full distance matrix section
    start_str = "----- FullDistanceMatrix ----"
    start_index = content.index(start_str) + len(start_str)
    end_index = content.find("-----", start_index)  # Find the next section separator
    matrix_section = content[start_index:end_index].strip().split("\n")[2:]  # Exclude the header lines
    
    # Convert the matrix section to a 2D list of distances
    distances = []
    for line in matrix_section:
        try:
            distance_row = list(map(float, line.split()[1:]))  # Excluding the leading port number
            distances.append(distance_row)
        except ValueError:
            continue
    
    return distances

def km_to_nautical_miles(km):
    return km / 1.852

def convert_matrix_to_nautical_miles(matrix):
    return [[km_to_nautical_miles(distance) for distance in row] for row in matrix]

def fuel_consumption_speed_nm(speed, nautical_miles):
    """
    Calculate the fuel consumption based on speed and nautical miles.

    Args:
    - speed (float): Speed of the vessel in knots.
    - nautical miles (float): .

    Returns:
    - float: Fuel consumption in tons.
    """
    return  (0.15*14 * (speed / 14) ** 3) * nautical_miles/speed
    
def calc_cost(fuel_consumption, FUEL_PRICE):
    """
    Calculate the cost based on fuel consumption.

    Args:
    - fuel_consumption (float): Fuel consumption in tons.

    Returns:
    - float: Cost in USD.
    """
    return fuel_consumption * FUEL_PRICE

def calculate_minimum_timesteps_and_speed(distance_nm, MAX_SPEED, MIN_SPEED):
    """
    Determine the minimum timesteps and speed based on distance and max speed.

    Args:
    - distance_nm (float): Distance in nautical miles.
    - MAX_SPEED (float): Maximum speed in knots.
    - MIN_SPEED (float): Minimum speed in knots.

    Returns:
    - tuple: Minimum timesteps and speed.
    """
    hours = distance_nm / MAX_SPEED
    minimum_timesteps = math.ceil(hours / 24)
    speed = distance_nm / (minimum_timesteps * 24)
    return minimum_timesteps, max(speed, MIN_SPEED)
    
def create_arc_info(speed, minimum_timesteps, departure, origin_port, destination_port, lowest_speed, distance_to_port, vessel, is_waiting_arc, NUM_TIME_PERIODS):
    # Create a list of tuples with the speed and the time period
    '''arc_info: (speed in knots, timesteps for sailing, time period of departure, time period of arrival, origin port, destination port, fuel_consumption)'''
    
    arrival_time = departure + minimum_timesteps
    if arrival_time > NUM_TIME_PERIODS:
        return None
    if is_waiting_arc:
        fuel_consumption = 0
        arc_info = [(speed, 1, departure, departure + minimum_timesteps, origin_port, destination_port, fuel_consumption, distance_to_port, vessel, is_waiting_arc)]
    else:
        # Create the info for all the speed alternatives
        fuel_consumption = fuel_consumption_speed_nm(speed=speed, nautical_miles=distance_to_port)
        arc_info = [(speed, minimum_timesteps, departure, departure + minimum_timesteps, origin_port, destination_port, fuel_consumption, distance_to_port, vessel, is_waiting_arc)]
    
        timesteps = minimum_timesteps+1
        while True:
            
            # Calculate the next speed
            speed = distance_to_port / ((timesteps)*24)
            # If the speed is lower than the lowest speed, break the loop
            if speed < lowest_speed:
                break
            fuel_consumption = fuel_consumption_speed_nm(speed=speed, nautical_miles=distance_to_port)
            arrival_time = departure + timesteps
            # Otherwise, add the speed to the list
            arc_info.append((speed, timesteps, departure, arrival_time, origin_port, destination_port, fuel_consumption, distance_to_port, vessel, is_waiting_arc))
            # Increment the time period
            timesteps += 1
            arrival_time = departure + timesteps
            if arrival_time > NUM_TIME_PERIODS:
                break
       
    return arc_info
   
def reindex_regions(loading_regions, discharging_regions):
    # For the keys in discharging_regions, change the key to index of the region starting with 0
    loading_regions_reidx = {}

    # Reassign keys
    for i, key in enumerate(loading_regions.keys()):
        loading_regions_reidx[i] = loading_regions[key]
    
    # For the keys in discharging_regions, change the key to index of the region starting with 0
    discharging_regions_reidx = {}

    # Reassign keys
    for i, key in enumerate(discharging_regions.keys()):
        discharging_regions_reidx[i] = discharging_regions[key]
        
    return loading_regions_reidx, discharging_regions_reidx

def generate_source_arc_info(sourceNode, vessels, vessel_data, loading_regions_reidx, discharging_regions_reidx, start_info, NODE_DICT):
    source_arcs_info = []
        
    for vessel in vessels:
        initial_port_str = vessel_data['Vessel_'+str(vessel.number-1)]['Initial Port']
        region_index, port_index = int(initial_port_str.split('_')[1]), int(initial_port_str.split('_')[3])
        # Check whether the initial port is a loading port or a discharging port based on initial_port_str
        if initial_port_str.startswith('Loading'):
            region_ports = loading_regions_reidx[region_index]
            initial_port = region_ports[int(port_index)]
        else:
            region_ports = discharging_regions_reidx[region_index]
            initial_port = region_ports[int(port_index)]
    
        first_time_available = int(vessel_data['Vessel_'+str(vessel.number-1)]['First Time Available']+1)
        # Add the port and first time available to the start_info dictionary
        start_info[vessel] = NODE_DICT[(initial_port.number, first_time_available)]
        # Only create one arc from the source to the corresponding port and time pair
        source_arcs_info_for_vessel = []
        speed = 0
        arc_info = [(speed, first_time_available, 0, first_time_available, sourceNode.port.number, initial_port.number, 0, 0, vessel, False)]
        source_arcs_info_for_vessel.append(arc_info)
        source_arcs_info.append(source_arcs_info_for_vessel)
    return source_arcs_info

def create_arcs_for_node(node, vessels, all_port_numbers, source_times, full_distance_matrix_nm, MAX_SPEED, MIN_SPEED, ports, NUM_TIME_PERIODS):
    node_arcs = []
    
    for vessel in vessels:
        if node.port.number in [0, len(ports) + 1]:
            continue
        start_time = source_times[vessel]
        if node.time < start_time:
            continue

        for destination_port_number in all_port_numbers:
            if destination_port_number == node.port.number:
                # Create waiting_arc
                arc_info_matrix = create_arc_info(speed=0, minimum_timesteps=1, departure=node.time, origin_port=node.port.number, destination_port=destination_port_number, lowest_speed=0, distance_to_port=0, vessel=vessel, is_waiting_arc=True, NUM_TIME_PERIODS = NUM_TIME_PERIODS)
                node_arcs.append(arc_info_matrix)
            else:
                distance_nm = full_distance_matrix_nm[node.port.number - 1][destination_port_number - 1]
                minimum_timesteps, speed = calculate_minimum_timesteps_and_speed(distance_nm=distance_nm, MAX_SPEED=MAX_SPEED, MIN_SPEED=MIN_SPEED)
                arrival_time = node.time + minimum_timesteps
                
                if arrival_time < source_times[vessel] or arrival_time > NUM_TIME_PERIODS:
                    continue

                arc_info_matrix = create_arc_info(speed=speed, minimum_timesteps=minimum_timesteps, departure=node.time, origin_port=node.port.number, destination_port=destination_port_number, lowest_speed=MIN_SPEED, distance_to_port=distance_nm, vessel=vessel, is_waiting_arc=False, NUM_TIME_PERIODS = NUM_TIME_PERIODS)
                node_arcs.append(arc_info_matrix)

    return node_arcs

def add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS, WAITING_COST, FUEL_PRICE):
    arc_dict = {}
    vessel_arcs = {}
    waiting_arcs = {}

    for sublist in all_info:
        for subsublist in sublist:
            if not subsublist:
                continue
                
            for tuple_data in subsublist:
                speed, timesteps, departure, arrival, origin_port_number, destination_port_number, fuel_consumption, distance_nm, vessel, is_waiting_arc = tuple_data
                cost = calc_cost(fuel_consumption, FUEL_PRICE)
                origin_node_obj = NODE_DICT.get((origin_port_number, departure))
                destination_node_obj = NODE_DICT.get((destination_port_number, arrival))

                if origin_node_obj and destination_node_obj and arrival <= NUM_TIME_PERIODS:
                    arc = Arc(origin_node=origin_node_obj, destination_node=destination_node_obj, distance=distance_nm, cost=cost + WAITING_COST, travel_time=timesteps, speed=speed, is_waiting_arc=is_waiting_arc)
                    origin_node_obj.outgoing_arcs.add(arc)
                    destination_node_obj.incoming_arcs.add(arc)
                    arc_dict[(origin_node_obj.tuple, destination_node_obj.tuple, vessel)] = arc
                    
                    if is_waiting_arc:
                        waiting_arcs.setdefault(vessel, []).append(arc)
                    vessel_arcs.setdefault(vessel, []).append(arc)

    return arc_dict, vessel_arcs, waiting_arcs

def populate_all_info(source_arcs_info, vessels, ports, start_times, full_distance_matrix_nm, NODE_DICT, MAX_SPEED, MIN_SPEED, NUM_TIME_PERIODS):     
    # Initialize the all_info list with the source arcs info
    all_info = source_arcs_info.copy()
    
    # Append the all_info list with the arcs for each node
    for vessel in vessels:
        # For each node, create the outgoing arcs
        arcs_for_current_vessel = [create_arcs_for_node(node, [vessel], [port.number for port in ports], start_times, full_distance_matrix_nm, MAX_SPEED, MIN_SPEED, ports, NUM_TIME_PERIODS) for node in NODE_DICT.values()]
        all_info.extend(arcs_for_current_vessel)
    return all_info

def add_arc_to_dict(origin, destination, vessel, arc_dict, vessel_arcs):
    """Helper function to add arc to dictionaries and nodes."""
    arc = Arc(origin_node=origin, destination_node=destination, distance=0, cost=0, travel_time=0, speed=0, is_waiting_arc=False)
    origin.outgoing_arcs.add(arc)
    destination.incoming_arcs.add(arc)
    arc_dict[(origin.tuple, destination.tuple, vessel)] = arc
    vessel_arcs.setdefault(vessel, []).append(arc)
    return arc_dict, vessel_arcs

def create_sink_arcs(vessels, sinkNode, start_times, arc_dict, vessel_arcs, sourceNode, NODE_DICT, ports):
    for vessel in vessels:
        # Arc from source node to sink node
        arc_dict, vessel_arcs = add_arc_to_dict(sourceNode, sinkNode, vessel, arc_dict, vessel_arcs)
        
        # Arcs from other nodes to sink node
        for node in NODE_DICT.values():
            if node.port.number not in [0, len(ports) + 1] and node.time >= start_times[vessel]:
                arc_dict, vessel_arcs = add_arc_to_dict(node, sinkNode, vessel, arc_dict, vessel_arcs)
    
    return arc_dict, vessel_arcs

def visualize_network_for_vessel(vessel, vessel_arcs, drop_sink_arcs, NODES, sinkNode):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in NODES:
        G.add_node(str(node.tuple))

    # Add edges (arcs) to the graph
    for arc in vessel_arcs[vessel]:
        # If the arc goes to the sink node, skip it
        if drop_sink_arcs and arc.destination_node == sinkNode:
            continue
        G.add_edge(str(arc.origin_node.tuple), str(arc.destination_node.tuple))

    # Determine NODES with incoming and outgoing arcs
    nodes_with_incoming_arcs = [node for node, degree in G.in_degree() if degree > 0]
    nodes_with_outgoing_arcs = [node for node, degree in G.out_degree() if degree > 0]

    # Create a list to hold node colors
    node_colors = []
    for node in G.nodes():
        if node in nodes_with_incoming_arcs or node in nodes_with_outgoing_arcs:
            node_colors.append('green')  # Color for nodes with arcs
        else:
            node_colors.append('skyblue')  # Default color

    # Resetting the y_offset and y_spacing
    y_offset = 10
    y_spacing = -30  # Increase vertical spacing for better clarity

    # Manually specify the positions for each node
    pos = {}

    # Manually set the position for the source and sink nodes
    # pos["(0, 0)"] = (0, 0)  # Positioning source node at leftmost, middle height
    # pos["(5, 5)"] = (5 * 10, 0)  # Positioning sink node at rightmost, middle height

    for node in NODES:
        # Skip setting position for source and sink nodes
        # if str(node.tuple) in ["(0, 0)", "(5, 5)"]:
        #     continue
        port_index = node.port.number  # Get port number to determine y-coordinate
        # The x-coordinate is based on time, the y-coordinate is fixed for nodes with the same port
        pos[str(node.tuple)] = (node.time * 10, port_index * y_spacing)  # Multiplying time by 10 for better horizontal spacing

    # Drawing the graph using the adjusted positions
    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Nodes and Arcs Graph\nVessel " + str(vessel.number))
    plt.show()
    
def find_earliest_visits_for_port(filtered_arcs, start_node):
    # Dictionary to store the earliest visit time for each port
    earliest_visits = {}

    # Filter arcs to include only those originating from the start node
    outgoing_arcs = [arc for arc in filtered_arcs if arc.origin_node == start_node]

    # Iterate through the outgoing arcs
    for arc in outgoing_arcs:
        dest_port = arc.destination_node.port
        dest_time = arc.destination_node.time

        # Update the earliest visit time for each destination port
        if dest_port not in earliest_visits or dest_time < earliest_visits[dest_port]:
            earliest_visits[dest_port] = dest_time

    return earliest_visits

def set_earliest_visit_for_vessel(vessels, filtered_arcs, start_info):        
    earliest_visits = {}
    for vessel in vessels:
        earliest_visits[vessel] = find_earliest_visits_for_port(filtered_arcs[vessel], start_info[vessel])
    return earliest_visits

def filter_arcs_for_vessel_on_earliest_visit(filtered_arcs, earliest_visits, vessels, sourceNode, start_info):
    ev_filtered = filtered_arcs.copy()

    for v in vessels:
        # Create a new list for storing filtered arcs
        new_filtered_arcs = []

        for arc in ev_filtered[v]:
            # Skip the source and sink nodes
            # if arc.origin_node == sourceNode or arc.destination_node == sinkNode:
            if arc.origin_node == sourceNode:
                new_filtered_arcs.append(arc)
                continue
            
            if arc.origin_node == start_info[v]:
                new_filtered_arcs.append(arc)
                continue

            # Check the earliest visit time constraints
            if arc.origin_node.time >= earliest_visits[v][arc.origin_node.port] \
            and arc.destination_node.time >= earliest_visits[v][arc.destination_node.port]:
                new_filtered_arcs.append(arc)

        # Replace the original list with the new filtered list
        ev_filtered[v] = new_filtered_arcs
    return ev_filtered


def create_o_dict(vessels, vessel_arcs, sourceNode):
    o_dict = {}
    for v in vessels:
        for arc in vessel_arcs[v]:
            o_node = arc.origin_node
            #set v to key and o_node to value, if o_node in o_dict, do not add and do not include sourceNode and sinkNode
            if v not in o_dict.keys():
                if o_node != sourceNode:
                    o_dict[v] = [o_node]
            else:
                if o_node not in o_dict[v] and o_node != sourceNode:
                    o_dict[v].append(o_node)
    return o_dict

def non_operational_nodes(vessels, NODES, o_dict, sourceNode, sinkNode):
    # For each node that are in NODES, but not in o_dict, add node to a dictionary with key = vessel and value = node
    non_operational = {}
    for v in vessels:
        for node in NODES:
            if node not in o_dict[v] and node != sourceNode and node != sinkNode:
                non_operational.setdefault(v, []).append(node)
    return non_operational

def find_sink_arcs(vessels, vessel_arcs, sinkNode):
    # Find all sink arcs for each vessel
    sink_arcs = {}
    for v in vessels:
        sink_arcs[v] = [arc for arc in vessel_arcs[v] if arc.destination_node == sinkNode]
    return sink_arcs

def build_problem(INSTANCE):    
    INSTANCE_PATH = INSTANCE+'/'+INSTANCE+'.txt'
    VESSELINFO_PATH = INSTANCE+'/vessel_data.txt'
    
    # Read file content
    with open(INSTANCE_PATH, 'r') as file:
        content = file.read()
        
    # Extract and adjust metadata
    metadata = extract_metadata(content)
    metadata = adjust_vessel_classes(metadata)
    ORIGINAL_NUM_TIME_PERIODS = metadata['numPeriods']
    vessel_data = get_vesels_data(VESSELINFO_PATH)
    regions_info, ports_info = extract_region_and_port_info(content)
    all_ports, loading_regions, discharging_regions = create_ports_from_info_with_loading(ports_info)
    ports = get_ports(all_ports)
    NUM_TIME_PERIODS = 45
    metadata['numPeriods'] = NUM_TIME_PERIODS
    TIME_PERIOD_RANGE = list(range(1, NUM_TIME_PERIODS+1))
    ORIGINAL_NUM_VESSELS = metadata['numTermVesselsInClass'][0]
    MAX_SPEED = 15
    MIN_SPEED = 8
    OPERATING_SPEED = 14
    OPERATING_COST = 200
    WAITING_COST = 50
    FUEL_PRICE = 500
    ORIGINAL_NUM_PORTS = len(ports)
    EBS = 0.01
    # Extracting the vessel dictionary and the capacity for Class 0 vessels
    vessels_info, VESSEL_CAP = parse_vessel_table(INSTANCE_PATH)
    vessels = get_vessels(metadata, vessel_data, VESSEL_CAP)
    regularNodes = create_regular_nodes(NUM_TIME_PERIODS, ports)
    sourcePort, sinkPort = create_special_ports(vessels, ports)
    sourceNode, sinkNode = create_special_nodes(sourcePort, sinkPort, NUM_TIME_PERIODS)
    NODES, NODE_DICT = create_node_list_and_dict(sourceNode, regularNodes, sinkNode)
    # Extracting the full distance matrix from the file content
    full_distance_matrix = parse_full_distance_matrix(INSTANCE_PATH)
    FULL_DISTANCE_MATRIX = full_distance_matrix
    full_distance_matrix_nm = convert_matrix_to_nautical_miles(FULL_DISTANCE_MATRIX)
    loading_regions_reidx, discharging_regions_reidx = reindex_regions(loading_regions, discharging_regions)
    start_info = {}
    start_times = {vessel: vessel_data['Vessel_'+str(vessel.number-1)]['First Time Available']+1 for vessel in vessels}
    # Generate the information for the source arcs
    source_arcs_info = generate_source_arc_info(sourceNode, vessels, vessel_data, loading_regions_reidx, discharging_regions_reidx, start_info, NODE_DICT)
    all_info = populate_all_info(source_arcs_info, vessels, ports, start_times, full_distance_matrix_nm, NODE_DICT, MAX_SPEED, MIN_SPEED, NUM_TIME_PERIODS)
    arc_dict, vessel_arcs, waiting_arcs = add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS, WAITING_COST, FUEL_PRICE)
    arc_dict, vessel_arcs = create_sink_arcs(vessels, sinkNode, start_times, arc_dict, vessel_arcs, sourceNode, NODE_DICT, ports)
    filtered_arcs = vessel_arcs.copy()
    earliest_visits = set_earliest_visit_for_vessel(vessels, filtered_arcs, start_info)
    vessel_arcs = filter_arcs_for_vessel_on_earliest_visit(filtered_arcs, earliest_visits, vessels, sourceNode, start_info)
    o_dict = create_o_dict(vessels, vessel_arcs, sourceNode)
    non_operational = non_operational_nodes(vessels, NODES, o_dict, sourceNode, sinkNode)
    
    problem_data = {
        'vessels': vessels,
        'vessel_arcs': vessel_arcs,
        'regularNodes': regularNodes,
        'ports': ports,
        'TIME_PERIOD_RANGE': TIME_PERIOD_RANGE,
        'non_operational': non_operational,
        'sourceNode': sourceNode,
        'sinkNode': sinkNode,
        'waiting_arcs': waiting_arcs,
        'OPERATING_COST': OPERATING_COST,
        'OPERATING_SPEED': OPERATING_SPEED,
        'NODES' : NODES
    }
    
    return problem_data


# def get_operating_speed_and_waiting_arcs(vessel_arcs, OPERATING_SPEED, NODES, ports):
#     reduced_vessel_arcs = {}
#     # Want to only keep one arc for each origin_node and destination_port pair
#     for node in NODES:
#         outgoing_arcs = [arc for arc in node.outgoing_arcs]
#         # If the node is the source or sink node, keep all arcs
#         if node.port.number in [0, len(ports) + 1]:
#             # Keep all arcs
#             for vessel in vessel_arcs.keys():
#                 for arc in outgoing_arcs:
#                     if arc in vessel_arcs[vessel]:
#                         reduced_vessel_arcs.setdefault(vessel, []).append(arc)
                        
#         else:
#             # Keep only the arc that have the speed closest to the operating speed for each origin_node and destination_port pair
            
#             operating_speed_arc = min(outgoing_arcs, key=lambda arc: abs(arc.speed - OPERATING_SPEED))
#             # Keep also the waiting arcs
#             waiting_arc = [arc for arc in node.outgoing_arcs if arc.is_waiting_arc]
#             for vessel in vessel_arcs.keys():
#                 if operating_speed_arc in vessel_arcs[vessel]:
#                     reduced_vessel_arcs[vessel] = [operating_speed_arc]
#                     if waiting_arc in vessel_arcs[vessel]:
#                         reduced_vessel_arcs[vessel].append(waiting_arc[0])
#     return reduced_vessel_arcs

def get_operating_speed_and_waiting_arcs(vessel_arcs, OPERATING_SPEED, NODES, ports):
    reduced_vessel_arcs = {}
    for vessel, arcs in vessel_arcs.items():
        for node in NODES:
            # Filter arcs that originate from the node and belong to the current vessel
            outgoing_arcs = [arc for arc in node.outgoing_arcs if arc in arcs]
            
            if not outgoing_arcs:
                continue

            if node.port.number in [0, len(ports) + 1]:
                # Source or sink node: keep all arcs for this vessel
                reduced_vessel_arcs.setdefault(vessel, []).extend(outgoing_arcs)
            else:
                # Group arcs by their destination port
                arcs_by_destination = {}
                for arc in outgoing_arcs:
                    dest_port = arc.destination_node.port.number
                    arcs_by_destination.setdefault(dest_port, []).append(arc)
                
                for dest_port, arcs_to_dest in arcs_by_destination.items():
                    # Separate waiting arcs and operational arcs
                    waiting_arcs = [arc for arc in arcs_to_dest if arc.is_waiting_arc]
                    operational_arcs = [arc for arc in arcs_to_dest if not arc.is_waiting_arc and arc.speed <= OPERATING_SPEED]
                    
                    if operational_arcs:
                        # Find the operational arc with the speed closest to (but not exceeding) the OPERATING_SPEED
                        closest_speed_arc = min(operational_arcs, key=lambda arc: abs(arc.speed - OPERATING_SPEED))
                        reduced_vessel_arcs.setdefault(vessel, []).append(closest_speed_arc)
                    
                    reduced_vessel_arcs[vessel].extend(waiting_arcs)
    
    return reduced_vessel_arcs


    

    
### GUROBI MODEL
def build_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST):
    m = gp.Model('Maritime Inventory Routing Problem Speed Optimization')
    
    '''Creating the variables'''
    '''Binary first'''
    # x is the binary variable that indicates whether a vessel travels on arc a, where an arc is a route frome one node to another node.
    x = m.addVars(((arc.tuple, vessel) for vessel in vessels for arc in vessel_arcs[vessel]) , vtype=gp.GRB.BINARY, name="x")

    # o is the binary variable that indicates whether vessel v is operating (loading/unloading) at node n
    o = m.addVars(((node.port.number, node.time, vessel) for node in regularNodes for vessel in vessels), vtype=gp.GRB.BINARY, name="o")

    '''Continuous varibles'''
    # q is the amount of product loaded or unloaded at port i by vessel v at time t
    q_bounds = {(node.port.number, node.time, vessel): min(vessel.max_inventory, node.port.capacity) for node in regularNodes for vessel in vessels}
    q = m.addVars(q_bounds.keys(), lb=0, ub=q_bounds, vtype=gp.GRB.CONTINUOUS, name="q")

    # s is the amount of product at port i at the end of period t
    s_bounds = {(node.port.number, node.time): node.port.capacity for node in regularNodes}
    s = m.addVars(s_bounds.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")

    # Create s vars for each port in time period 0
    s_bounds_source = {(port.number, 0): port.capacity for port in ports}
    s_source = m.addVars(s_bounds_source.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")
    s.update(s_source)

    # w is the amount of product on board of vessel v at the end of time period t
    w_bounds = {(t, vessel): vessel.max_inventory for vessel in vessels for t in TIME_PERIOD_RANGE}
    w = m.addVars(w_bounds.keys(), lb=0, ub=w_bounds, vtype=gp.GRB.CONTINUOUS, name="w")

    w_bounds_source = {(0, vessel): vessel.max_inventory for vessel in vessels}
    w_source = m.addVars(w_bounds_source.keys(), lb=0, ub=w_bounds, vtype=gp.GRB.CONTINUOUS, name="w")
    w.update(w_source)
    
    # Create a dict where the arc.tuple is the key and arc.cost is the value
    costs = {(arc.tuple, vessel): arc.cost for vessel in vessels for arc in vessel_arcs[vessel] }
    m.update()

    original_obj = gp.quicksum(costs[key]*x[key] for key in costs) + gp.quicksum(o[node.port.number, node.time, vessel] * OPERATING_COST for node in regularNodes for vessel in vessels)
    #Minimize the costs
    m.setObjective(original_obj, GRB.MINIMIZE)
    m.update()

    # Can fix some variables, to reduce the complexity of the model
    for v in vessels:
        for node in non_operational[v]:   
            port_number = node.port.number
            time = node.time
            o[port_number, time, v] = 0
            q[port_number, time, v] = 0
    m.update()

    # Constraint (2)
    '''Must leave the source node'''
    for v in vessels:
        outgoing_from_source = [arc for arc in vessel_arcs[v] if arc.origin_node == sourceNode]
        m.addConstr(gp.quicksum((x[arc.tuple, v]) for arc in outgoing_from_source) == 1, name = 'SourceFlow')
    m.update()

    # Constraint (3)
    '''Must enter the sink node'''
    for v in vessels:
        incoming_to_sink = [arc for arc in vessel_arcs[v] if arc.destination_node == sinkNode]
        m.addConstr(gp.quicksum((x[arc.tuple, v]) for arc in incoming_to_sink) == 1, name = 'SinkFlow')
    m.update()

    # Constraint (4)
    '''For each node we enter, we must leave'''
    for v in vessels:
        for node in regularNodes:
            outgoing_from_node = [arc for arc in vessel_arcs[v] if arc.origin_node == node]
            incoming_to_node = [arc for arc in vessel_arcs[v] if arc.destination_node == node]
            m.addConstr(gp.quicksum((x[in_arc.tuple, v]) for in_arc in incoming_to_node) - gp.quicksum((x[out_arc.tuple, v]) for out_arc in outgoing_from_node) == 0, name = "FlowBalance")
    m.update()

    # Constraint (5)
    '''Set correct initial inventory at each port'''
    for port in ports:
        m.addConstr(s_source[port.number, 0] == port.inventory, name = 'InitialInventoryPort')
    m.update()

    # Constraint (6)
    '''Inventory balance at ports'''
    # Inventory balance for ports at the end of each time period t
    for port in ports:
        for t in TIME_PERIOD_RANGE:
            m.addConstr(s[port.number, t] == (s[port.number, t-1] + (port.isLoadingPort * port.rate) - gp.quicksum(port.isLoadingPort * q[port.number, t, v] for v in vessels)) , name = 'PortBalance')
    m.update()

    # Constraint (7)
    '''Set correct initial inventory at each vessel'''
    for v in vessels:
        m.addConstr(w_source[0, v] == v.inventory, name = 'InitialInventoryVessel')
    m.update()

    # Constraint (8)
    '''Inventory balance at vessel'''
    # for each vessel, the inventory at the end of the time period is equal to the inventory at the beginning of the time period + the amount of product loaded/unloaded at the ports
    for t in TIME_PERIOD_RANGE:
        for v in vessels:
            m.addConstr(w[t, v] == gp.quicksum(port.isLoadingPort * q[port.number, t, v] for port in ports) + w[t-1,v], name = 'VesselBalance')
    m.update()

    # Constraint (9)
    '''Berth limit'''
    for node in regularNodes:
        m.addConstr((gp.quicksum((o[node.port.number, node.time, v]) for v in vessels) <= node.port.berth_limit), name = 'Birth_limit_in_time_t')
    m.update()

    #Constraint (10)
    '''Must be physically present to operate'''
    for v in vessels:
        for node in regularNodes:
            incoming_to_node = [arc for arc in vessel_arcs[v] if arc.destination_node == node]
            m.addConstr(o[node.port.number, node.time, v] <= gp.quicksum((x[in_arc.tuple, v]) for in_arc in incoming_to_node), name = 'Cannot operate unless vessel is at node')
    m.update()

    # Constraint (11)
    '''After an operation, we must either continue operating or move to another port'''
    all_waiting_arcs = {}
    for v in vessels:
        for arc in waiting_arcs[v]:
            origin_node = arc.origin_node
            destination_node = arc.destination_node
            all_waiting_arcs[((origin_node.port.number, origin_node.time),(destination_node.port.number, destination_node.time) , v)] = arc
                    
    for node in regularNodes:
        for v in vessels:
            if (node.port.number, node.time + 1, v) in o.keys():  # check if o for the next period exists
                
                waiting_arc_key = ((node.port.number, node.time), (node.port.number, node.time + 1), v)
                if waiting_arc_key in all_waiting_arcs.keys():
                    waiting_arc = all_waiting_arcs[waiting_arc_key]
                else:
                    continue
                
                actual_key = (waiting_arc.tuple, v)
                
                if actual_key in x.keys():  # check if the waiting arc exists
                    m.addConstr(o[node.port.number, node.time, v] <= 
                                o[node.port.number, node.time + 1, v] + 
                                (1 - x[actual_key]),
                                name=f"Operate_or_Move_{node.port.number}_{node.time}_{v}")
    m.update()

    # #Constraint (12)
    '''Cannot load/unload more than the maximum operating quantity'''
    for v in vessels:
        for node in regularNodes:
            m.addConstr(q[node.port.number, node.time, v] <= o[node.port.number, node.time, v]*v.max_operating_quantity, name = 'Max_operating_quantity')
    m.update()

    # Contraint (13)
    '''Must operate in every port we visit.'''
    # O_itv <= o_i(t+1)v + sum of x_ijv for all j except waiting arc
    for v in vessels:
        for node in regularNodes:
            outgoing_from_node = [arc for arc in vessel_arcs[v] if arc.origin_node == node]
            # Remove the waiting arc from the list
            for a in outgoing_from_node:
                if a.is_waiting_arc:
                    outgoing_from_node.remove(a)
            m.addConstr(o[node.port.number, node.time, v] >= gp.quicksum((x[out_arc.tuple, v]) for out_arc in outgoing_from_node), name = 'Must_operate_in_every_port')
    m.update()

    # Constraint (14) modification
    # Ensure that q is at least 1/4 of the vessel's capacity if o is 1
    for v in vessels:
        for node in regularNodes:
            m.addConstr(q[node.port.number, node.time, v] >= 1/4 * v.max_inventory * o[node.port.number, node.time, v], 
                        name=f'q_{node.port.number}_{node.time}_{v}')
    m.update()
    
    return m, costs

### GUROBI MODEL
def build_simplified_RL_model(vessels, vessel_arcs, regularNodes, ports, TIME_PERIOD_RANGE, non_operational, sourceNode, sinkNode, waiting_arcs, OPERATING_COST, OPERATING_SPEED, NODES):
    m = gp.Model('Maritime Inventory Routing Problem Speed Optimization')
    port_dict = {port.number: port for port in ports}
    print(port_dict)
    
    # Keep only the arcs that are using the operating speed
    vessel_arcs = get_operating_speed_and_waiting_arcs(vessel_arcs, OPERATING_SPEED, NODES, ports)
    
    
    '''Creating the variables'''
    '''Binary first'''
    # x is the binary variable that indicates whether a vessel travels on arc a, where an arc is a route frome one node to another node.
    # ONLY ADD THE ARCS THAT ARE USING OPERATING SPEED AND ARE NOT WAITING ARCS
    active_x_keys = []
    for vessel in vessels:
        for arc in vessel_arcs[vessel]:
            # if arc.speed == OPERATING_SPEED and arc.is_waiting_arc==False:
            if arc.is_waiting_arc==False:
                active_x_keys.append((arc.tuple, vessel))
                
    # x = m.addVars(((arc.tuple, vessel) for vessel in vessels for arc in vessel_arcs[vessel] if arc.speed == OPERATING_SPEED and arc.waiting_arc==False) , vtype=gp.GRB.BINARY, name="x")
    x = m.addVars(((arc.tuple, vessel) for vessel in vessels for arc in vessel_arcs[vessel] if arc.is_waiting_arc==False) , vtype=gp.GRB.BINARY, name="x")
    
    active_o_keys = []
    for vessel in vessels:
        for arc in vessel_arcs[vessel]:
            # if arc.speed == OPERATING_SPEED and arc.is_waiting_arc==False:
            # if arc.is_waiting_arc==False and arc.origin_node.port.number != 0 and arc.destination_node.port.number != len(ports) + 1:
            if arc.is_waiting_arc==False and arc.origin_node.port.number != 0:
                o_key = (arc.origin_node.port.number, arc.origin_node.time, vessel)
                if o_key not in active_o_keys:
                    active_o_keys.append(o_key)

    # o is the binary variable that indicates whether vessel v is operating (loading/unloading) at node n
    o = m.addVars((o_key for o_key in active_o_keys), vtype=gp.GRB.BINARY, name="o")

    '''Continuous varibles'''
    # q is the amount of product loaded or unloaded at port i by vessel v at time t
    q_bounds = {o_key: min(o_key[2].max_inventory, port_dict[o_key[0]].capacity) for o_key in active_o_keys}
    q = m.addVars(q_bounds.keys(), lb=0, ub=q_bounds, vtype=gp.GRB.CONTINUOUS, name="q")

    # s is the amount of product at port i at the end of period t
    s_bounds = {(node.port.number, node.time): node.port.capacity for node in regularNodes}
    s = m.addVars(s_bounds.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")

    # Create s vars for each port in time period 0
    s_bounds_source = {(port.number, 0): port.capacity for port in ports}
    s_source = m.addVars(s_bounds_source.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")
    s.update(s_source)

    # w is the amount of product on board of vessel v at the end of time period t
    w_bounds = {(t, vessel): vessel.max_inventory for vessel in vessels for t in TIME_PERIOD_RANGE}
    w = m.addVars(w_bounds.keys(), lb=0, ub=w_bounds, vtype=gp.GRB.CONTINUOUS, name="w")

    w_bounds_source = {(0, vessel): vessel.max_inventory for vessel in vessels}
    w_source = m.addVars(w_bounds_source.keys(), lb=0, ub=w_bounds, vtype=gp.GRB.CONTINUOUS, name="w")
    w.update(w_source)
    
    # Create a dict where the arc.tuple is the key and arc.cost is the value
    costs = {(arc.tuple, vessel): arc.cost for vessel in vessels for arc in vessel_arcs[vessel]}
    m.update()
    

    original_obj = gp.quicksum(costs[key]*x[key] for key in active_x_keys) + gp.quicksum(o[o_key] * OPERATING_COST for o_key in active_o_keys)
    #Minimize the costs
    m.setObjective(original_obj, GRB.MINIMIZE)
    m.update()

    # Can fix some variables, to reduce the complexity of the model
    for v in vessels:
        for node in non_operational[v]:   
            port_number = node.port.number
            time = node.time
            o[port_number, time, v] = 0
            q[port_number, time, v] = 0
    m.update()

    # Constraint (2)
    '''Must leave the source node'''
    for v in vessels:
        outgoing_from_source = [arc for arc in vessel_arcs[v] if arc.origin_node == sourceNode]
        m.addConstr(gp.quicksum((x[arc.tuple, v]) for arc in outgoing_from_source if (arc.tuple, v) in active_x_keys) == 1, name = 'SourceFlow')
    m.update()

    # Constraint (3)
    '''Must enter the sink node'''
    for v in vessels:
        incoming_to_sink = [arc for arc in vessel_arcs[v] if arc.destination_node == sinkNode]
        m.addConstr(gp.quicksum((x[arc.tuple, v]) for arc in incoming_to_sink if (arc.tuple, v) in active_x_keys) == 1, name = 'SinkFlow')
    m.update()

    # Constraint (4)
    '''For each node we enter, we must leave'''
    for v in vessels:
        for node in regularNodes:
            outgoing_from_node = [arc for arc in vessel_arcs[v] if arc.origin_node == node]
            incoming_to_node = [arc for arc in vessel_arcs[v] if arc.destination_node == node]
            m.addConstr(gp.quicksum((x[in_arc.tuple, v]) for in_arc in incoming_to_node if (in_arc.tuple, v) in active_x_keys) - gp.quicksum((x[out_arc.tuple, v]) for out_arc in outgoing_from_node if (out_arc.tuple, v) in active_x_keys) == 0, name = "FlowBalance")
    m.update()

    # Constraint (5)
    '''Set correct initial inventory at each port'''
    for port in ports:
        m.addConstr(s_source[port.number, 0] == port.inventory, name = 'InitialInventoryPort')
    m.update()

    # Constraint (6)
    '''Inventory balance at ports'''
    # Inventory balance for ports at the end of each time period t
    for port in ports:
        for t in TIME_PERIOD_RANGE:
            m.addConstr(s[port.number, t] == (s[port.number, t-1] + (port.isLoadingPort * port.rate) - gp.quicksum(port.isLoadingPort * q[port.number, t, v] for v in vessels if (port.number, t, v) in active_o_keys)) , name = 'PortBalance')
    m.update()

    # Constraint (7)
    '''Set correct initial inventory at each vessel'''
    for v in vessels:
        m.addConstr(w_source[0, v] == v.inventory, name = 'InitialInventoryVessel')
    m.update()

    # Constraint (8)
    '''Inventory balance at vessel'''
    # for each vessel, the inventory at the end of the time period is equal to the inventory at the beginning of the time period + the amount of product loaded/unloaded at the ports
    for t in TIME_PERIOD_RANGE:
        for v in vessels:
            m.addConstr(w[t, v] == gp.quicksum(port.isLoadingPort * q[port.number, t, v] for port in ports if (port.number, t, v) in active_o_keys) + w[t-1,v], name = 'VesselBalance')
    m.update()

    # Constraint (9)
    '''Berth limit'''
    for node in regularNodes:
        m.addConstr((gp.quicksum((o[node.port.number, node.time, v]) for v in vessels if (port.number, t, v) in active_o_keys) <= node.port.berth_limit), name = 'Birth_limit_in_time_t')
    m.update()

    #Constraint (10)
    '''Must be physically present to operate'''
    for v in vessels:
        for node in regularNodes:
            incoming_to_node = [arc for arc in vessel_arcs[v] if arc.destination_node == node]
            m.addConstr(o[node.port.number, node.time, v] <= gp.quicksum((x[in_arc.tuple, v]) for in_arc in incoming_to_node if (in_arc.tuple, v) in active_x_keys and (node.port.number, node.time, v) in active_o_keys), name = 'Cannot operate unless vessel is at node')
    m.update()

    # Constraint (11)
    '''After an operation, we must either continue operating or move to another port'''
    all_waiting_arcs = {}
    for v in vessels:
        for arc in waiting_arcs[v]:
            origin_node = arc.origin_node
            destination_node = arc.destination_node
            all_waiting_arcs[((origin_node.port.number, origin_node.time),(destination_node.port.number, destination_node.time) , v)] = arc
                    
    for node in regularNodes:
        for v in vessels:
            if (node.port.number, node.time + 1, v) in o.keys():  # check if o for the next period exists
                
                waiting_arc_key = ((node.port.number, node.time), (node.port.number, node.time + 1), v)
                if waiting_arc_key in all_waiting_arcs.keys():
                    waiting_arc = all_waiting_arcs[waiting_arc_key]
                else:
                    continue
                
                actual_key = (waiting_arc.tuple, v)
                
                if actual_key in x.keys():  # check if the waiting arc exists
                    m.addConstr(o[node.port.number, node.time, v] <= 
                                o[node.port.number, node.time + 1, v] + 
                                (1 - x[actual_key]),
                                name=f"Operate_or_Move_{node.port.number}_{node.time}_{v}")
    m.update()

    # #Constraint (12)
    '''Cannot load/unload more than the maximum operating quantity'''
    for v in vessels:
        for node in regularNodes:
            m.addConstr(q[node.port.number, node.time, v] <= o[node.port.number, node.time, v]*v.max_operating_quantity, name = 'Max_operating_quantity')
    m.update()

    # Contraint (13)
    '''Must operate in every port we visit.'''
    # O_itv <= o_i(t+1)v + sum of x_ijv for all j except waiting arc
    for v in vessels:
        for node in regularNodes:
            outgoing_from_node = [arc for arc in vessel_arcs[v] if arc.origin_node == node]
            # Remove the waiting arc from the list
            for a in outgoing_from_node:
                if a.is_waiting_arc:
                    outgoing_from_node.remove(a)
            m.addConstr(o[node.port.number, node.time, v] >= gp.quicksum((x[out_arc.tuple, v]) for out_arc in outgoing_from_node), name = 'Must_operate_in_every_port')
    m.update()

    # Constraint (14) modification
    # Ensure that q is at least 1/4 of the vessel's capacity if o is 1
    for v in vessels:
        for node in regularNodes:
            m.addConstr(q[node.port.number, node.time, v] >= 1/4 * v.max_inventory * o[node.port.number, node.time, v], 
                        name=f'q_{node.port.number}_{node.time}_{v}')
    m.update()
    
    env_data = {
        'vessels': vessels,
        'vessel_arcs': vessel_arcs,
        'regularNodes': regularNodes,
        'ports': ports,
        'TIME_PERIOD_RANGE': TIME_PERIOD_RANGE,
        'non_operational': non_operational,
        'sourceNode': sourceNode,
        'sinkNode': sinkNode,
        'waiting_arcs': waiting_arcs,
        'OPERATING_COST': OPERATING_COST,
        'OPERATING_SPEED': OPERATING_SPEED,
        'ports_dict' : port_dict
    }
    
    return m, env_data

def find_initial_solution(model):
    # Set the time limit for finding an initial solution
    H = 3600  # 1 hour in seconds
    model.setParam(gp.GRB.Param.TimeLimit, 3*H)  # 3 hours limit

    # Set SolutionLimit to 1 to stop after finding the first feasible solution
    model.setParam(gp.GRB.Param.SolutionLimit, 1)
    
    # model.setParam(gp.GRB.Param.Emphasis, gp.GRB.Emphasis.Feasibility)
    
    model.setParam(gp.GRB.Param.Heuristics, 0.5)  # Range is 0 to 1; higher values increase heuristic efforts
    
    model.setParam(gp.GRB.Param.MIPFocus, 1)

    
    # Run the optimization
    model.optimize()
    
    # Check if a feasible solution was found
    if model.SolCount > 0:
        # Retrieve the solution
        x_solution = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
        # Optionally, you can also capture the objective value of the initial solution
        # initial_solution_obj = model.ObjVal
        
        return x_solution, model
    else:
        print("No feasible solution found within the time limit.")
        return None, None
    

def solve_model(model):
    H = 3600  # 1 hour in seconds
    model.setParam(gp.GRB.Param.TimeLimit, 0.25*H)  # 3 hours limit

    # Run the optimization
    model.optimize()
    
    # Check the status of the optimization
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found!")
        
    if model.status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        # Write the IIS to a .ilp file
        model.write("MIRPSO.ilp")
        print("Model is infeasible")