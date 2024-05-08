import re
import time
import copy
import torch
import math
import gurobipy as gp
from gurobipy import GRB
from common_definitions import MIRPSOEnv, DQNAgent, ReplayMemory, DQNAgent
from optimization_utils import *
from rolling_horizon import rolling_horizon_optimization

class Port:
    def __init__(self, capacity, inventory, rate, price, berth_limit, port_fee, max_amount, min_amount, number, isLoadingPort, name, spotMarketPricePerUnit, spotMarketDiscountFactor):
        self.capacity = capacity
        self.initial_inventory = inventory
        self.inventory = inventory
        self.rate = rate
        self.price = price 
        self.berth_limit = berth_limit 
        self.port_fee = port_fee
        self.max_amount = max_amount
        self.min_amount = min_amount
        self.number = number
        self.isLoadingPort = isLoadingPort
        self.name = name
        self.spotMarketPricePerUnit = spotMarketPricePerUnit
        self.spotMarketDiscountFactor = spotMarketDiscountFactor
        
    def __repr__(self):
        return f'Port {self.number}'
    
    def __repr2__(self):
        return f'Port {self.number}: Capacity = {self.capacity}, Inventory = {self.inventory}, Rate = {self.rate}, Price = {self.price}, Berth Limit = {self.berth_limit}, Port Fee = {self.port_fee}, Max Amount = {self.max_amount}, Min Amount = {self.min_amount}, is Loading Port = {self.isLoadingPort}'
        
class Node:
    def __init__(self, port, time):
        self.port = port
        self.time = time
        self.tuple = (port.number if port else None, time)
        self.incoming_arcs = {}
        self.outgoing_arcs = {}
        self.berths = port.berth_limit if port else None
    
    def __repr__(self):
        return str(self.tuple)

class Arc:
    def __init__(self, origin_node, destination_node, distance, cost, travel_time, speed, is_waiting_arc, is_source_or_sink_arc, vessel_class, speed_ranking, num_speed_options):
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.tuple = (origin_node, destination_node)
        self.distance = distance
        self.cost = cost
        self.travel_time = travel_time
        self.speed = speed
        self.is_waiting_arc = is_waiting_arc
        self.is_source_or_sink_arc = is_source_or_sink_arc
        self.vessel_class = vessel_class
        self.speed_ranking = speed_ranking
        self.num_speed_options = num_speed_options
        
        
    def to_dict(self):
        return {
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "distance": self.distance,
            "cost": self.cost,
            "travel_time": self.travel_time,
            "speed": self.speed,
            "is_waiting_arc": self.is_waiting_arc,
            "is_source_or_sink_arc": self.is_source_or_sink_arc,
            "vessel_class": self.vessel_class,
            "speed_ranking": self.speed_ranking,
            "num_speed_options": self.num_speed_options
        }
            
    def __repr__(self):
        return f'{self.origin_node} -> {self.destination_node} --- Cost: {self.cost:.1f} --- Speed: {self.speed:.1f}'

class Vessel:
    def __init__(self, capacity, initial_inventory, number, vessel_class, avg_speed, travel_cost_as_term_per_km, discount_traveling_empty, first_time_available, initial_port):
        self.capacity = int(capacity)
        self.inventory = initial_inventory
        self.number = number
        self.vessel_class = vessel_class
        self.avg_speed = avg_speed
        self.travel_cost_as_term_per_km = travel_cost_as_term_per_km
        self.discount_traveling_empty = discount_traveling_empty
        self.first_time_available = first_time_available
        self.initial_port = initial_port
        self.isFinished = False
        self.arcs = set()
        self.all_arcs_v = {}
        # If a vessel is in transit towards a port, this attribute will store the destination port.
        # It will change to None when the vessel arrives at the port. The position attribute will also be updated to the value of the destination 
        self.in_transit_towards = (None, None)
        self.action_path = []
        # Position is which port it is located at. If it is in transit, position is None
        self.position = None
        self.min_speed = avg_speed - 5 
        self.max_speed = avg_speed + 2 
        
    def __repr__(self):
        return f'Vessel {self.number}'

    def __repr2__(self):
        return f'Vessel {self.number}: Max Inventory = {self.max_inventory}, Inventory = {self.inventory}, Max Operating Quantity = {self.max_operating_quantity}'

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
    
def get_vessels_data(VESSELINFO_PATH):
    ### Read the vessel data
    with open(VESSELINFO_PATH, 'r') as file:
        file_content = file.read()

    # Extended regular expression to extract all vessel information
    pattern = r"""
        name\s+Vessel_(\d+)\s+                  # Vessel index
        index\s+(\d+)\s+                        # Duplicate of Vessel index, can ignore or verify
        vesselClass\s+(\d+)\s+                  # Vessel class
        type\s+(\w+)\s+                         # Type
        initialInventory\s+(\d+)\s+             # Initial inventory
        initialPort\s+([\w_]+)\s+               # Initial port
        firstTimeAvailable\s+(\d+)              # First time available
    """

    # Finding all matches
    vessel_info = re.findall(pattern, file_content, re.DOTALL | re.VERBOSE)

    # Creating a dictionary to store the information
    vessel_data = {}
    for vessel in vessel_info:
        vessel_index, _, vessel_class, vessel_type, initial_inventory, initial_port, first_time_available = vessel

        vessel_data[f"Vessel_{vessel_index}"] = {
            "Vessel Class": int(vessel_class),
            "Type": vessel_type,
            "Initial Inventory": int(initial_inventory),
            "Initial Port": initial_port,
            "First Time Available": int(first_time_available)
            # You can add other attributes here as needed, following the same pattern.
        }
    return vessel_data

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

def create_ports_from_info_with_loading(ports_info, spotMarketPricePerUnit, spotMarketDiscountFactor):
    all_ports = {}
    loading_regions = {}
    discharging_regions = {}
    total_ports = 1
    port_dict = {}

    # Counters for loading and discharging regions
    loading_region_counter = 0
    discharging_region_counter = 0
    port_counter = {}  # Keeps track of port numbers within each region

    for region, port_attributes in ports_info.items():
        region_ports = []
        is_loading_region = all(int(rate) > 0 for rate in port_attributes['Rate'])
        is_discharging_region = all(int(rate) < 0 for rate in port_attributes['Rate'])

        if is_loading_region and region not in port_counter:
            port_counter[region] = (loading_region_counter, 0)  # Initialize counter for this region
            loading_region_counter += 1
        elif is_discharging_region and region not in port_counter:
            port_counter[region] = (discharging_region_counter, 0)
            discharging_region_counter += 1

        for i in range(len(port_attributes['Capacity'])):
            rate = int(port_attributes['Rate'][i]) if 'Rate' in port_attributes else 0
            isLoading = rate > 0
            region_counter, port_num = port_counter[region]
            port_counter[region] = (region_counter, port_num + 1)  # Increment port number

            # Set port name based on its type (loading or discharging)
            if isLoading:
                name = f'LoadingRegion_{region_counter}_Port_{port_num}'
            else:
                name = f'DischargeRegion_{region_counter}_Port_{port_num}'

            port = Port(
                capacity=int(port_attributes['Capacity'][i]),
                inventory=int(port_attributes['Inventory'][i]) if 'Inventory' in port_attributes else None,
                rate=abs(rate),
                price=int(port_attributes['Price'][i]) if 'Price' in port_attributes and port_attributes['Price'] else None,
                berth_limit=int(port_attributes['BerthLimit'][i]) if 'BerthLimit' in port_attributes else None,
                port_fee=int(port_attributes['PortFee'][i]) if 'PortFee' in port_attributes else None,
                max_amount=int(port_attributes['maxAmt'][i]) if 'maxAmt' in port_attributes else None,
                min_amount=int(port_attributes['minAmt'][i]) if 'minAmt' in port_attributes else None,
                number=total_ports,  # Using port_num + 1 as the port number
                isLoadingPort=1 if isLoading else -1,  # Determine loading port based on Rate value
                name=name,  # Set the name based on region and port type
                spotMarketPricePerUnit = spotMarketPricePerUnit,
                spotMarketDiscountFactor = spotMarketDiscountFactor
            )
            region_ports.append(port)
            port_dict[port.number] = port
            total_ports += 1

        all_ports[region] = region_ports

        # Assign region to the correct dictionary
        if is_loading_region:
            loading_regions[region] = region_ports
        elif is_discharging_region:
            discharging_regions[region] = region_ports

    return all_ports, loading_regions, discharging_regions, port_dict

def get_ports(all_ports):
    # Create a list of all ports
    ports = []
    for region, region_ports in all_ports.items():
        ports.extend(region_ports)
    return ports

def parse_vessel_table(INSTANCE_PATH):
    with open(INSTANCE_PATH, 'r') as file:
        lines = file.readlines()

    vessel_class_data = {}
    current_class = None
    for line in lines:
        line = line.strip()
        if line.startswith("name"):
            # Extract vessel class index from the line
            _, class_name = line.split()
            current_class = int(class_name.split('_')[-1])  # Assumes name format "Vessel_Class_X"
            vessel_class_data[current_class] = {}
        else:
            key, value = line.split(maxsplit=1)
            # Convert numerical values appropriately
            if value.replace('.', '', 1).isdigit():
                value = float(value) if '.' in value else int(value)
            vessel_class_data[current_class][key] = value

    return vessel_class_data

def get_vessels(metadata, vessel_data, vessel_class_data, ports):    
    vessels = []
    name_num = 0
    tot = 1
    for vessel_class in range(metadata['numVesselClasses']):
        vessels_in_class = metadata['numTermVesselsInClass'][vessel_class]
        VESSEL_CAP = vessel_class_data[vessel_class]['capacity']  # Use capacity for each vessel class
        avg_speed_in_knots = vessel_class_data[vessel_class]['avgSpeedInKnots']
        travel_costs_as_term_per_km = vessel_class_data[vessel_class]['travelCostAsTermPerKm']
        discount_traveling_empty = vessel_class_data[vessel_class]['discountTravelingEmpty']
        for i in range(vessels_in_class):
            first_time_available = vessel_data['Vessel_'+str(name_num)]['First Time Available']
            # Find initial port for the vessel
            initial_port = None
            for port in ports:
                if port.name == vessel_data['Vessel_'+str(name_num)]['Initial Port']:
                    initial_port = port
                    break
            if int(vessel_data['Vessel_'+str(name_num)]['Initial Inventory']) > 0:
                init_inventory = VESSEL_CAP
            else:
                init_inventory = 0

            vessels.append(Vessel(
                capacity=VESSEL_CAP,
                initial_inventory=init_inventory,
                number=tot,
                vessel_class=vessel_class,
                avg_speed = avg_speed_in_knots, 
                travel_cost_as_term_per_km = travel_costs_as_term_per_km,
                discount_traveling_empty = discount_traveling_empty,
                first_time_available=first_time_available,
                initial_port=initial_port
            ))
            tot += 1
            name_num += 1
    return vessels

def create_regular_nodes(NUM_TIME_PERIODS, ports):
    # Create the regular nodes
    regularNodes = []
    for t in range(NUM_TIME_PERIODS):
        for port in ports:
            node = Node(port=port, time=t)
            regularNodes.append(node)
    return regularNodes

def create_special_ports(vessels, ports):
    # Create fictional source and sink port
    sourcePort = Port(capacity=None, inventory=None, rate=None, price=None, berth_limit=len(vessels), port_fee=0, max_amount=None, min_amount=None, number=0, isLoadingPort=True, name='SourcePort', spotMarketPricePerUnit=0, spotMarketDiscountFactor=0)
    sinkPort = Port(capacity=None, inventory=None, rate=None, price=None, berth_limit=len(vessels), port_fee=0, max_amount=None, min_amount=None, number=len(ports)+1, isLoadingPort=False, name='SinkPort', spotMarketPricePerUnit=0, spotMarketDiscountFactor=0)
    return sourcePort, sinkPort

def create_special_nodes(sourcePort, sinkPort, NUM_TIME_PERIODS):
    # Create source and sink node
    sourceNode = Node(port=sourcePort, time=-1)
    sinkNode = Node(port=sinkPort, time=NUM_TIME_PERIODS)
    return sourceNode, sinkNode

def create_node_list_and_dict(sourceNode, regularNodes, sinkNode):
    NODES = [sourceNode] + regularNodes + [sinkNode]
    # Create a node dictionary with key = (port, time) tuple and value = node object
    NODE_DICT = {}
    for node in NODES:
        NODE_DICT[node.tuple] = node
    return NODES, NODE_DICT

def parse_full_distance_matrix(INSTANCE_PATH):
    '''KILOMETERS'''
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

def parse_full_travel_cost_matrices(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_marker = "----- FullTravelCostMatrixForClass -----"
    end_marker = "FullTravelCostMatrixForClass(vc,i,j)"
    cost_matrix_content = content.split(start_marker)[1].split(end_marker)[0]

    travel_costs = {}

    pattern = r"Vessel_Class_(\d+)\s+([\s\S]+?)(?=Vessel_Class_|$)"
    matches = re.findall(pattern, cost_matrix_content)

    for match in matches:
        vessel_class = int(match[0])
        matrix_data = match[1].strip()

        # Split the matrix data into lines and ignore the first line which contains column indices
        matrix_lines = matrix_data.split('\n')[1:]  # Start from the second line to skip column headers

        matrix = []
        for line in matrix_lines:
            # Split each line into values, convert to float, and remove the first value which is a row index
            row = [float(x) for x in line.split()][1:]  # Skip the first element (row label)
            matrix.append(row)

        travel_costs[vessel_class] = matrix

    return travel_costs

def parse_full_travel_time_matrices(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_marker = "----- FullTravelTimeMatrixForClass -----"
    end_marker = "FullTravelTimeMatrixForClass(vc,i,j)"
    travel_time_matrix_content = content.split(start_marker)[1].split(end_marker)[0]

    travel_times = {}

    # Pattern to extract sections for each vessel class's travel time matrix
    pattern = r"Vessel_Class_(\d+)\s+([\s\S]+?)(?=Vessel_Class_|$)"
    matches = re.findall(pattern, travel_time_matrix_content)

    for match in matches:
        vessel_class = int(match[0])
        matrix_data = match[1].strip()

        # Split the matrix data into lines and ignore the first line which contains column indices
        matrix_lines = matrix_data.split('\n')[1:]  # Start from the second line to skip column headers

        matrix = []
        for line in matrix_lines:
            # Split each line into values, convert to float, and remove the first value which is a row index
            row = [int(x) for x in line.split()][1:]  # Skip the first element (row label), convert to int
            matrix.append(row)

        travel_times[vessel_class] = matrix

    return travel_times

def generate_source_arc_info(sourceNode, vessels):
    source_arcs_info = []
    for vessel in vessels:
        initial_port = vessel.initial_port
        first_time_available = vessel.first_time_available
        speed = 0
        arc_info = [speed, first_time_available +1, -1, first_time_available, sourceNode.port.number, initial_port.number, initial_port.port_fee, 0, vessel, False, True, None, None]
        source_arcs_info.append(arc_info)
    return [source_arcs_info]

def populate_all_info(source_arcs_info, vessels, ports, source_times, full_distance_matrix_km, full_distance_matrix_nm, NODE_DICT, NUM_TIME_PERIODS, full_travel_cost_matrices, full_travel_time_matrices, port_dict, FUEL_PRICE, MIRPSO):     
    # Initialize the all_info list with the source arcs info
    all_info = source_arcs_info.copy()
    # Append the all_info list with the arcs for each node
    for vessel in vessels:
        travel_cost_matrix = full_travel_cost_matrices[vessel.vessel_class]
        travel_time_matrix = full_travel_time_matrices[vessel.vessel_class]
        
        if MIRPSO:
            arcs_for_current_vessel = [create_arcs_for_node_MIRPSO(node=node, vessel=vessel, all_port_numbers = [port.number for port in ports], source_times=source_times, full_distance_matrix_nm = full_distance_matrix_nm, MAX_SPEED=vessel.max_speed, MIN_SPEED=vessel.min_speed, ports=ports, NUM_TIME_PERIODS=NUM_TIME_PERIODS, port_dict=port_dict, FUEL_PRICE=FUEL_PRICE) for node in NODE_DICT.values()]
        else:
            # For each node, create the outgoing arcs
            arcs_for_current_vessel = [create_arcs_for_node(node, vessel, [port.number for port in ports], full_distance_matrix_km, ports, NUM_TIME_PERIODS, travel_cost_matrix, travel_time_matrix, port_dict) for node in NODE_DICT.values()]
            
        # Remove all empty lists from the arcs_for_current_vessel list
        arcs_for_current_vessel = [arc for arc in arcs_for_current_vessel if arc]
        if arcs_for_current_vessel:
            all_info.extend(arcs_for_current_vessel)
    return all_info

def create_arcs_for_node(node, vessel, all_port_numbers, full_distance_matrix_km, ports, NUM_TIME_PERIODS, travel_cost_matrix, travel_time_matrix, port_dict):
    node_arcs = []
    if node.port.number in [0, len(ports) + 1]:
        return node_arcs

    for destination_port_number in all_port_numbers:
        destination_port = port_dict[destination_port_number]
        if destination_port_number == node.port.number:
            # Create waiting_arc
            arc_info = [0, 1, node.time, node.time+1, node.port.number, destination_port_number, 0, 0, vessel, True, False, None, None]
            node_arcs.append(arc_info)
        else:
            # Only create the arcs from a loading to a discharging port and vice versa
            if node.port.isLoadingPort != destination_port.isLoadingPort:
                # The the arc is from a discharging port to a loading port, the cost is reduced by the discount factor
                distance_km = full_distance_matrix_km[node.port.number - 1][destination_port_number - 1]
                if node.port.isLoadingPort != 1:
                    # Cannot extract the costs directly.
                    cost_per_km_for_vessel = vessel.travel_cost_as_term_per_km  
                    travel_cost = distance_km * cost_per_km_for_vessel
                    # Add discount
                    travel_cost = travel_cost * (1 - vessel.discount_traveling_empty)
                    # Add port fee at destination
                    travel_cost += destination_port.port_fee  
                else:
                    # Note that there is a difference between [origin, destination] and [destination, origin] in the travel cost matrix. Check that it is correct
                    travel_cost = travel_cost_matrix[node.port.number - 1][destination_port_number - 1]
                    
                travel_time = travel_time_matrix[node.port.number - 1][destination_port_number - 1]
                arrival_time = node.time + travel_time
                
                if arrival_time < vessel.first_time_available or arrival_time > NUM_TIME_PERIODS:
                    continue
                arc_info = [vessel.avg_speed, travel_time, node.time, arrival_time, node.port.number, destination_port_number, travel_cost, distance_km, vessel, False, False, None, None]
                node_arcs.append(arc_info)

    return node_arcs

def create_arcs_for_node_MIRPSO(node, vessel, all_port_numbers, source_times, full_distance_matrix_nm,  MAX_SPEED, MIN_SPEED, ports, NUM_TIME_PERIODS, port_dict, FUEL_PRICE):
    node_arcs = []
    
    # for vessel in vessels:
    if node.port.number in [0, len(ports) + 1]:
        return node_arcs
    start_time = source_times[vessel]
    if node.time < start_time:
        return node_arcs

    for destination_port_number in all_port_numbers:
        if destination_port_number == node.port.number:
            # Create waiting_arc
            arc_info_matrix = create_arc_info(speed=0, minimum_timesteps=1, departure=node.time, origin_port=node.port, destination_port=node.port, lowest_speed=MIN_SPEED, distance_to_port=0, vessel=vessel, is_waiting_arc=True, NUM_TIME_PERIODS=NUM_TIME_PERIODS, FUEL_PRICE=FUEL_PRICE)
            node_arcs.append(arc_info_matrix)
        else:
            destination_port = port_dict[destination_port_number]
            origin_is_loading = node.port.isLoadingPort == 1
            destination_is_loading = destination_port.isLoadingPort == 1
            # If both are loading or both are discharging, skip
            if origin_is_loading == destination_is_loading:
                continue
            # Only create the arcs from a loading to a discharging port and vice versa
            distance_nm = full_distance_matrix_nm[node.port.number - 1][destination_port_number - 1]
            minimum_timesteps, speed = calculate_minimum_timesteps_and_speed(distance_nm=distance_nm, MAX_SPEED=MAX_SPEED, MIN_SPEED=MIN_SPEED)
            arrival_time = node.time + minimum_timesteps
            if arrival_time < source_times[vessel] or arrival_time > NUM_TIME_PERIODS:
                continue
            
            arc_info_matrix = create_arc_info(speed=speed, minimum_timesteps=minimum_timesteps, departure=node.time, origin_port=node.port, destination_port=destination_port, lowest_speed=MIN_SPEED, distance_to_port=distance_nm, vessel=vessel, is_waiting_arc=False, NUM_TIME_PERIODS = NUM_TIME_PERIODS, FUEL_PRICE=FUEL_PRICE)
            for arc in arc_info_matrix:
                if arc:
                    node_arcs.append(arc)
    return node_arcs

def calc_cost(fuel_consumption, FUEL_PRICE):
    """
    Calculate the cost based on fuel consumption.

    Args:
    - fuel_consumption (float): Fuel consumption in tons.

    Returns:
    - float: Cost in USD.
    """
    return fuel_consumption * FUEL_PRICE

def create_arc_info(speed, minimum_timesteps, departure, origin_port, destination_port, lowest_speed, distance_to_port, vessel, is_waiting_arc, NUM_TIME_PERIODS, FUEL_PRICE):
    num_speed_options = 0
    
    # Create a list of tuples with the speed and the time period
    arc_info = []
    
    arrival_time = departure + minimum_timesteps
    
    if arrival_time > NUM_TIME_PERIODS:
        return None
    if is_waiting_arc:
        arc_info = [speed, 1, departure, departure + minimum_timesteps, origin_port.number, destination_port.number, 0, distance_to_port, vessel, is_waiting_arc, False, None, None]
    else:
        # Create the info for all the speed alternatives
        speed_ranking = 1
        num_speed_options = 1
        if origin_port.isLoadingPort == 1:
            empty = False
        else:
            empty = True
        
        travel_cost = calculate_cost(vessel=vessel, speed=speed, nautical_miles=distance_to_port, empty=empty)
        # Add port fee at destination
        travel_cost += destination_port.port_fee
        
        arc_tuple = [speed, minimum_timesteps, departure, arrival_time, origin_port.number, destination_port.number, travel_cost, distance_to_port, vessel, False, False, speed_ranking, num_speed_options]
        arc_info.append(arc_tuple)
        
        timesteps = minimum_timesteps+1
        while True:
            speed_ranking += 1
            num_speed_options += 1
            # Calculate the next speed
            speed = distance_to_port / ((timesteps)*24)
            # If the speed is lower than the lowest speed, break the loop
            if speed < lowest_speed:
                break
            travel_cost = calculate_cost(vessel=vessel, speed=speed, nautical_miles=distance_to_port, empty=empty)
            # Add port fee at destination
            travel_cost += destination_port.port_fee
            
            arrival_time = departure + timesteps
            arc_tuple = [speed, minimum_timesteps, departure, arrival_time, origin_port.number, destination_port.number, travel_cost, distance_to_port, vessel, False, False, speed_ranking, num_speed_options]
            arc_info.append(arc_tuple)
            
            # Increment the time period
            timesteps += 1
            arrival_time = departure + timesteps
            
            if arrival_time > NUM_TIME_PERIODS:
                break
    return arc_info

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

def calculate_cost(vessel, speed, nautical_miles, empty):
    """
    Calculate the fuel cost for a vessel class traveling at a given speed over a certain distance.

    Args:
    - vessel
    - speed (float): Vessel speed in knots.
    - nautical_miles (float): Distance to be traveled in nautical miles.
    - empty (bool): Whether the vessel is traveling empty.

    Returns:
    - float: Total fuel cost for the trip.
    """
    base_speed = vessel.avg_speed
    cost_per_km_at_base_speed = vessel.travel_cost_as_term_per_km
    discount = vessel.discount_traveling_empty if empty else 0
    
    # Adjust cost per km based on actual speed vs base speed, using cubic relationship
    cost_per_km = cost_per_km_at_base_speed * (speed / base_speed) ** 3

    # Apply discount if vessel is traveling empty
    cost_per_km *= (1 - discount) 
    
    # Convert nautical miles to km (1 nautical mile = 1.852 km)
    distance_km = nautical_miles * 1.852
    
    # Calculate total cost based on distance
    total_cost = cost_per_km * distance_km
    
    return total_cost

def convert_matrix_to_nautical_miles(matrix):
    return [[km_to_nautical_miles(distance) for distance in row] for row in matrix]

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

def add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS):
    arc_dict = {}
    vessel_arcs = {}
    waiting_arcs = {}
    sink_and_source_arcs = {}

    for node in all_info:
        for arc in node:
            if not arc:
                continue

            # Unpack the tuple data
            speed, travel_time, departure, arrival, origin_port_number, destination_port_number, travel_cost, distance_nm, vessel, is_waiting_arc, is_source_or_sink_arc, speed_ranking, num_speed_options = arc
            vessel_class = vessel.vessel_class

            # Retrieve the origin and destination nodes
            origin_node_obj = NODE_DICT.get((origin_port_number, departure))
            destination_node_obj = NODE_DICT.get((destination_port_number, arrival))
            
            if origin_node_obj and destination_node_obj and arrival <= NUM_TIME_PERIODS:
                # Initialize outgoing and incoming arcs for vessel classes if not already done
                origin_node_obj.outgoing_arcs.setdefault(vessel_class, [])
                destination_node_obj.incoming_arcs.setdefault(vessel_class, [])

                # Create the arc
                arc = Arc(origin_node=origin_node_obj, destination_node=destination_node_obj, distance=distance_nm, cost=travel_cost, travel_time=travel_time, speed=speed, is_waiting_arc=is_waiting_arc, is_source_or_sink_arc=is_source_or_sink_arc, vessel_class = vessel_class, speed_ranking=speed_ranking, num_speed_options=num_speed_options)
                
                # Add the arc to the respective lists
                origin_node_obj.outgoing_arcs[vessel_class].append(arc)
                destination_node_obj.incoming_arcs[vessel_class].append(arc)

                # Maintain a separate dictionary for easy access to arcs by vessel class
                arc_dict.setdefault(vessel_class, {})
                arc_dict[vessel_class][(origin_node_obj.tuple, destination_node_obj.tuple)] = arc
                
                # If it's a waiting arc, add it to the waiting_arcs dictionary
                if is_waiting_arc:
                    waiting_arcs.setdefault(vessel, []).append(arc)
                    
                if is_source_or_sink_arc:
                    sink_and_source_arcs.setdefault(vessel, []).append(arc)

                # Add the arc to the vessel-specific list of arcs
                vessel_arcs.setdefault(vessel, []).append(arc)

    return arc_dict, vessel_arcs, waiting_arcs, sink_and_source_arcs

def add_arc_to_dict(origin, sinkNode, vessel, arc_dict, vessel_arcs, perPeriodRewardForFinishingEarly):
    """Helper function to add arc to dictionaries and nodes."""
    # Calculate the reward for finishing early
    origin_time = origin.time
    time_diff = (sinkNode.time-1) - origin_time
    reward = perPeriodRewardForFinishingEarly * time_diff
    arc = Arc(origin_node=origin, destination_node=sinkNode, distance=0, cost=-reward, travel_time=0, speed=0, is_waiting_arc=False, is_source_or_sink_arc=True, vessel_class=vessel.vessel_class, speed_ranking=None, num_speed_options=None)
    origin.outgoing_arcs.setdefault(vessel.vessel_class, [])
    sinkNode.incoming_arcs.setdefault(vessel.vessel_class, [])
    origin.outgoing_arcs[vessel.vessel_class].append(arc)
    sinkNode.incoming_arcs[vessel.vessel_class].append(arc)
    
    arc_dict.setdefault(vessel.vessel_class, {})
    arc_dict[vessel.vessel_class][(origin.tuple, sinkNode.tuple)] = arc
    vessel_arcs.setdefault(vessel, []).append(arc)
    return arc_dict, vessel_arcs

def create_sink_arcs(vessels, sinkNode, arc_dict, vessel_arcs, sourceNode, NODE_DICT, perPeriodRewardForFinishingEarly):

    for vessel in vessels:
        sinkNode.incoming_arcs.setdefault(vessel.vessel_class, [])
        
        # Arcs from other nodes to sink node
        for node in NODE_DICT.values():
            if node.port.number != sinkNode.port.number and node.time >= vessel.first_time_available:
                arc_dict, vessel_arcs = add_arc_to_dict(node, sinkNode, vessel, arc_dict, vessel_arcs, perPeriodRewardForFinishingEarly)
        
    return arc_dict, vessel_arcs

def populate_vessel_arcs(vessels, vessel_arcs):
    for v in vessels:
        v.arcs = vessel_arcs[v]
        for arc in vessel_arcs[v]:
            v.all_arcs_v[arc.tuple] = arc
    return vessels

def add_special_sink_arcs(vessels, travel_times, NODE_DICT, ports, TIME_PERIOD_RANGE):
    # Create new nodes for the special sink arcs
    special_nodes_dict = {}
    for port in ports:
        special_sink = Node(port=port, time=TIME_PERIOD_RANGE[-1] + 1)
        special_nodes_dict[port.number] = special_sink
        
    special_sink_arcs = {}
    for vessel in vessels:
        for node in NODE_DICT.values():
            # Continue if node is the source or sink node
            if node.port.number in [0, len(ports) + 1]:
                continue
            # Find the travel times from this node
            travel_times_from_node = travel_times[vessel.vessel_class][node.port.number - 1]
            for port, travel_time in enumerate(travel_times_from_node):
                if node.time + travel_time > TIME_PERIOD_RANGE[-1]:
                    # Create a special sink arc
                    special_sink_arc = Arc(origin_node=node, destination_node=special_nodes_dict[port + 1], distance=0, cost=0, travel_time=travel_time, speed=0, is_waiting_arc=False, is_source_or_sink_arc=True, vessel_class=vessel.vessel_class, speed_ranking = None, num_speed_options = None) 
                    special_sink_arcs.setdefault(vessel, []).append(special_sink_arc)
                    
    return special_sink_arcs, special_nodes_dict
                
def rearrange_arcs(arc_dict):
    # New dictionary structures
    origin_node_arcs = {}
    destination_node_arcs = {}
    vessel_class_arcs = {}  # All arcs for each vessel class

    # Iterate through the original dictionary
    for vessel_class, arcs in arc_dict.items():
        # Initialize vessel class dictionaries if not present
        if vessel_class not in origin_node_arcs:
            origin_node_arcs[vessel_class] = {}
        if vessel_class not in destination_node_arcs:
            destination_node_arcs[vessel_class] = {}
        if vessel_class not in vessel_class_arcs:  # Initialize list for all arcs in vessel class
            vessel_class_arcs[vessel_class] = []

        for (origin_tuple, destination_tuple), arc in arcs.items():
            # Rearrange by origin node
            if origin_tuple not in origin_node_arcs[vessel_class]:
                origin_node_arcs[vessel_class][origin_tuple] = []
            origin_node_arcs[vessel_class][origin_tuple].append(arc)

            # Rearrange by destination node
            if destination_tuple not in destination_node_arcs[vessel_class]:
                destination_node_arcs[vessel_class][destination_tuple] = []
            destination_node_arcs[vessel_class][destination_tuple].append(arc)
            
            # Aggregate all arcs under their vessel class
            vessel_class_arcs[vessel_class].append(arc)

    return origin_node_arcs, destination_node_arcs, vessel_class_arcs
    
def build_model(vessels, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, vessel_classes, origin_node_arcs, destination_node_arcs, vessel_class_arcs, NODE_DICT, vessel_class_capacities):
    
    m = gp.Model('Maritime Inventory Routing Problem Speed Optimization')
    
    vessels_in_class = {vc: sum(1 for vessel in vessels if vessel.vessel_class == vc) for vc in vessel_classes}
    
    interregional_arcs = []
    for vc in vessel_classes:
        for arc in vessel_class_arcs[vc]:
            if not arc.is_waiting_arc and not arc.origin_node==sourceNode:
                interregional_arcs.append(arc)
    
    non_interregional_arcs = []
    for vc in vessel_classes:
        for arc in vessel_class_arcs[vc]:
            if arc not in interregional_arcs:
                non_interregional_arcs.append(arc)
                
    all_arcs = [arc for vc, arcs in vessel_class_arcs.items() for arc in arcs]
    non_interregional_arcs = [arc for arc in all_arcs if arc not in interregional_arcs]
    
    x_interregional = {}
    for arc in interregional_arcs:
        variable_name = f"x_interregional_{arc.tuple}_{arc.vessel_class}"
        x_interregional[(arc.tuple, arc.vessel_class)] = m.addVar(vtype=gp.GRB.BINARY, name=variable_name)
    m.update()  
        
    
    x_non_interregional = {}
    for arc in non_interregional_arcs:
        variable_name = f"x_non_interregional_{arc.tuple}_{arc.vessel_class}"
        x_non_interregional[(arc.tuple, arc.vessel_class)] = m.addVar(vtype=gp.GRB.INTEGER, ub=vessels_in_class[arc.vessel_class], name=variable_name)
    m.update()  # Update the model to add the variables

    x = {**x_interregional, **x_non_interregional}
                    
    s_bounds = {(node.port.number, node.time): node.port.capacity for node in regularNodes if node.time > 0}
    s = m.addVars(s_bounds.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")

    s_bounds_source = {(port.number, 0): port.initial_inventory for port in ports}
    s_source = m.addVars(s_bounds_source.keys(), lb=0, ub=s_bounds_source, vtype=gp.GRB.CONTINUOUS, name="s")
    s.update(s_source)
    
    s_bound_sink = {(port.number, TIME_PERIOD_RANGE[-1] + 1): port.capacity for port in ports}
    s_sink = m.addVars(s_bound_sink.keys(), lb=0, ub=s_bound_sink, vtype=gp.GRB.CONTINUOUS, name="s")
    s.update(s_sink)
    
    # alpha_bounds = {(node.port.number, node.time): ((node.port.capacity))for node in regularNodes}
    alpha_bounds = {(node.port.number, node.time): (node.port.rate) for node in regularNodes}
    # alpha = m.addVars(alpha_bounds.keys(), lb=0, ub=alpha_bounds, vtype=gp.GRB.CONTINUOUS, name="alpha")
    alpha = m.addVars(alpha_bounds.keys(), lb=0, ub=[alpha_bounds[key] for key in alpha_bounds], vtype=gp.GRB.CONTINUOUS, name=[f'alpha[{key[0]},{key[1]}]' for key in alpha_bounds]) 
    
    m.update()
    
    
    costs = {}
    costs_namekey = {}

    # Loop through each vessel class
    for vc in vessel_classes:
        for arc in vessel_class_arcs[vc]:
            # Generate the variable name based on whether the arc is interregional or not
            if not arc.is_waiting_arc and not arc.origin_node == sourceNode:
                # This arc is interregional
                key = (arc.tuple, vc)
                variable_name = f"x_interregional_{arc.tuple}_{vc}"
            else:
                # This arc is non-interregional
                key = (arc.tuple, vc)
                variable_name = f"x_non_interregional_{arc.tuple}_{vc}"
            
            # Assuming arc.cost is accessible and applicable for both arc types
            costs[key] = arc.cost
            costs_namekey[variable_name] = arc.cost

    # Now, costs dictionary keys should match the keys used in x_interregional and x_non_interregional
    m.update()  # Update the model after creating the costs dictionary
        
    # # Create a dict where the arc.tuple is the key and arc.cost is the value
    # costs = {(arc.tuple, vc): arc.cost for vc in vessel_classes for arc in vessel_class_arcs[vc]}
    # m.update()

    #Parameters:
    P = {(node.port.number, node.time): node.port.spotMarketPricePerUnit * (node.port.spotMarketDiscountFactor ** (node.time)) for node in regularNodes}
    original_obj = gp.quicksum(costs[key]*x[key] for key in costs) + gp.quicksum(alpha[node.port.number, node.time] * P[(node.port.number, node.time)] for node in regularNodes)

    m.setObjective(original_obj, GRB.MINIMIZE)
    m.update()


    # Create a dict with the vessel object in each vessel class
    vessels_in_vclass = {vc: [vessel for vessel in vessels if vessel.vessel_class == vc] for vc in vessel_classes}
    # Create a dict with the start_node for each vessel in each vessel class
    start_nodes_for_vessels_in_vc = {vc: [NODE_DICT[(vessel.initial_port.number, vessel.first_time_available)] for vessel in vessels_in_vclass[vc]] for vc in vessel_classes}
    # Use start_nodes_for_vessels_in_vc to count the amount of vessels that start at the same node in each vessel class
    start_node_counter = {vc: {node: 0 for node in start_nodes_for_vessels_in_vc[vc]} for vc in vessel_classes}
    for vc in vessel_classes:
        for v in vessels_in_vclass[vc]:
            start_node = NODE_DICT[(v.initial_port.number, v.first_time_available)]
            start_node_counter[vc][start_node] += 1
            
    # Constraint (2)
    '''Must leave the source node'''
    for vc in vessel_classes:
        for vessel in vessels_in_vclass[vc]:
            # Find the destination node for the arc
            destination_node = NODE_DICT[(vessel.initial_port.number, vessel.first_time_available)]
            key = (sourceNode, destination_node)
            source_arc = vessel.all_arcs_v[key]
            # All arcs from the source node must be used. Every vessel has only one arc from the source node
            m.addConstr(x[(source_arc.tuple), vessel.vessel_class] == start_node_counter[vessel.vessel_class][destination_node], name = 'SourceFlow')
    m.update()

    # Constraint (3)
    '''Must enter the sink node'''
    for vc in vessel_classes:
        incoming_to_sink = [arc for arc in vessel_class_arcs[vc] if arc.destination_node == sinkNode]
        m.addConstr(gp.quicksum((x[arc.tuple, vc]) for arc in incoming_to_sink) == vessels_in_class[vc], name = 'SinkFlow')
    m.update()

    # Constraint (4)
    '''For each node we enter, we must leave'''
    for vc in vessel_classes:
        for node in regularNodes:
            # Check if node is one of the keys in destination_node_arcs
            incoming_to_node = destination_node_arcs[vc].get(node.tuple, [])
            outgoing_from_node = origin_node_arcs[vc].get(node.tuple, [])
            if not incoming_to_node and outgoing_from_node:
                m.addConstr(gp.quicksum((x[out_arc.tuple, out_arc.vessel_class]) for out_arc in outgoing_from_node) == 0, name = f"FlowBalance_No_Incoming_to_{node.tuple}")
                
            elif incoming_to_node and not outgoing_from_node:
                m.addConstr(gp.quicksum((x[in_arc.tuple, in_arc.vessel_class]) for in_arc in incoming_to_node) == 0, name = f"FlowBalance_No_Outgoing_from_{node.tuple}")
                
            elif not incoming_to_node and not outgoing_from_node:
                continue
                
            else:
                m.addConstr(gp.quicksum((x[in_arc.tuple, in_arc.vessel_class]) for in_arc in incoming_to_node) - gp.quicksum((x[out_arc.tuple, out_arc.vessel_class]) for out_arc in outgoing_from_node) == 0, name = f'FlowBalance_{node.port.number}_{node.time}')
    m.update()

    # Constraint (5)
    '''Set correct initial inventory at each port'''
    for port in ports:
        m.addConstr(s[port.number, 0] == port.initial_inventory, name = f'InitialInventoryPort{port.number}_Time{0}')
    m.update()

   
    # Constraint (6)
    '''Inventory balance at ports'''
    # Inventory balance for ports at the end of each time period t
    # Create a new list with one additional time period at the end
    TIME_PERIOD_RANGE_INCL_SINK = list(range(len(TIME_PERIOD_RANGE)+1))
    for port in ports:
        for t in TIME_PERIOD_RANGE_INCL_SINK:
            if t == 0:
                continue
            else:
                arcs_with_origin_t_minus_1 = [arc for arc in interregional_arcs if arc.origin_node.time == t-1 and arc.origin_node.port == port]
                if arcs_with_origin_t_minus_1:
                    m.addConstr(s[port.number, t] == (s[port.number, t-1] + (port.isLoadingPort * port.rate) - port.isLoadingPort*alpha[port.number, t-1] - (port.isLoadingPort * gp.quicksum((x[arc.tuple, arc.vessel_class] * vessel_class_capacities[arc.vessel_class] for arc in arcs_with_origin_t_minus_1)))) , name = f'PortBalance_1_Inter_regional_Port{port.number}_Time{t}')
                else:
                    m.addConstr(s[port.number, t] == (s[port.number, t-1] + (port.isLoadingPort * port.rate) - port.isLoadingPort*alpha[port.number, t-1]), name = f'PortBalance_2_Inter_regional_Port{port.number}_Time{t}')
    m.update()
    
    # Constraint (7)
    '''Berth limit'''
    for node in regularNodes:
        interregional_arcs_for_node = []
        for arc in interregional_arcs:
            if arc.origin_node == node:
                interregional_arcs_for_node.append(arc)
        m.addConstr((gp.quicksum((x[arc.tuple, arc.vessel_class]) for arc in interregional_arcs_for_node) <= node.port.berth_limit), name = f'BerthLimit_Port{node.port.number}_Time{node.time}')
    m.update()
    
    return m, costs, P, costs_namekey
    
def build_problem(INSTANCE, RUNNING_MIRPSO):    
    INSTANCE_PATH = 'GROUP_2/'+INSTANCE+'/'+INSTANCE+'.txt'
    VESSELINFO_PATH = 'GROUP_2/'+INSTANCE+'/vessel_data.txt'
    VESSEL_CLASS_DATA_PATH = 'GROUP_2/'+INSTANCE+'/vessel_class_data.txt'
    
    # Read file content
    with open(INSTANCE_PATH, 'r') as file:
        content = file.read()
        
    # Extract and adjust metadata
    metadata = extract_metadata(content)
    spotMarketPricePerUnit = metadata['spotMarketPricePerUnit']
    spotMarketDiscountFactor = metadata['spotMarketDiscountFactor']
    perPeriodRewardForFinishingEarly = metadata['perPeriodRewardForFinishingEarly']
    vessel_data = get_vessels_data(VESSELINFO_PATH)
    _, ports_info = extract_region_and_port_info(content)
    all_ports, loading_regions, discharging_regions, port_dict = create_ports_from_info_with_loading(ports_info, spotMarketPricePerUnit, spotMarketDiscountFactor)
    ports = get_ports(all_ports)
    NUM_TIME_PERIODS = metadata['numPeriods']
    NUM_TIME_PERIODS = 40
    TIME_PERIOD_RANGE = list(range(0, NUM_TIME_PERIODS))
    FUEL_PRICE = 500
    ORIGINAL_NUM_PORTS = len(ports)
    
    # Extracting the vessel dictionary and the capacity for Class 0 vessels
    vessel_class_data = parse_vessel_table(VESSEL_CLASS_DATA_PATH)
    vessel_class_capacities = {vc: vessel_class_data[vc]['capacity'] for vc in range(metadata['numVesselClasses'])}
    VESSEL_CLASSES = list(range(metadata['numVesselClasses']))
    vessels = get_vessels(metadata, vessel_data, vessel_class_data, ports)
    regularNodes = create_regular_nodes(NUM_TIME_PERIODS, ports)
    sourcePort, sinkPort = create_special_ports(vessels, ports)
    sourceNode, sinkNode = create_special_nodes(sourcePort, sinkPort, NUM_TIME_PERIODS)
    NODES, NODE_DICT = create_node_list_and_dict(sourceNode, regularNodes, sinkNode)
    # Extracting the full distance matrix from the file content
    full_distance_matrix = parse_full_distance_matrix(INSTANCE_PATH)
    FULL_DISTANCE_MATRIX_KM = full_distance_matrix
    FULL_DISTANCE_MATRIX_NM = convert_matrix_to_nautical_miles(FULL_DISTANCE_MATRIX_KM)
    # full_distance_matrix_nm = convert_matrix_to_nautical_miles(FULL_DISTANCE_MATRIX)
    full_travel_cost_matrices = parse_full_travel_cost_matrices(INSTANCE_PATH)
    full_travel_time_matrices = parse_full_travel_time_matrices(INSTANCE_PATH)
    source_times = {vessel: vessel_data['Vessel_'+str(vessel.number-1)]['First Time Available'] for vessel in vessels}
    
    source_arcs_info = generate_source_arc_info(sourceNode, vessels)
    all_info = populate_all_info(source_arcs_info = source_arcs_info, vessels = vessels, ports = ports, source_times = source_times, full_distance_matrix_km = FULL_DISTANCE_MATRIX_KM, full_distance_matrix_nm= FULL_DISTANCE_MATRIX_NM, NODE_DICT=NODE_DICT, NUM_TIME_PERIODS=NUM_TIME_PERIODS, full_travel_cost_matrices=full_travel_cost_matrices, full_travel_time_matrices=full_travel_time_matrices, port_dict = port_dict, FUEL_PRICE=FUEL_PRICE, MIRPSO=RUNNING_MIRPSO)
    arc_dict, vessel_arcs, waiting_arcs, sink_and_source_arcs = add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS)
    arc_dict, vessel_arcs = create_sink_arcs(vessels=vessels, sinkNode=sinkNode, arc_dict=arc_dict, vessel_arcs=vessel_arcs, sourceNode=sourceNode, NODE_DICT=NODE_DICT, perPeriodRewardForFinishingEarly=perPeriodRewardForFinishingEarly)
    special_sink_arcs, special_nodes_dict = add_special_sink_arcs(vessels, full_travel_time_matrices, NODE_DICT, ports, TIME_PERIOD_RANGE)
    populate_vessel_arcs(vessels, vessel_arcs)
 
    problem_data = {
        'vessels': vessels,
        'vessel_arcs': vessel_arcs,
        'arc_dict': arc_dict,
        'regularNodes': regularNodes,
        'ports': ports,
        'TIME_PERIOD_RANGE': TIME_PERIOD_RANGE,
        'sourceNode': sourceNode,
        'sinkNode': sinkNode,
        'waiting_arcs': waiting_arcs,
        'NODES' : NODES,
        'NODE_DICT' : NODE_DICT,
        'VESSEL_CLASSES': VESSEL_CLASSES,
        'vessel_class_capacities': vessel_class_capacities,
        'special_sink_arcs': special_sink_arcs,
        'special_nodes_dict': special_nodes_dict
    }
    
    return problem_data

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
    
    # If the model is infeasible, write to the ilp
    if model.status == gp.GRB.Status.INFEASIBLE:
        model.setParam(gp.GRB.Param.TimeLimit, 300)
        model.computeIIS()
        model.write('infeasible_model_MIRP2.ilp')
    
    # Check if a feasible solution was found
    if model.SolCount > 0:
        # Retrieve the solution
        x_solution = {v.VarName: v.X for v in model.getVars() if v.VarName.startswith('x')}
        # Optionally, you can also capture the objective value of the initial solution
        initial_solution_obj = model.ObjVal
        print("Initial solution objective value:", initial_solution_obj)
        
        return x_solution, model
    else:
        print("No feasible solution found within the time limit.")
        return None, None







def update_weights_based_on_action(weights, active_arcs_chosen_by_agent):
    
    for arc in active_arcs_chosen_by_agent:
            # origin_node_agent = arc.origin_node
            # destination_port_number_agent = arc.destination_node.port.number
            varname_agent = convert_key_to_varname((arc.tuple, arc.vessel_class))
            weights[varname_agent] = 1
    
    
    # if RUNNING_MIRPSO:
    #     for arc in active_arcs_chosen_by_agent:
    #         origin_node_agent = arc.origin_node
    #         destination_port_number_agent = arc.destination_node.port.number
    #         varname_agent = convert_key_to_varname((arc.tuple, arc.vessel_class))
    #         weights[varname_agent] = 1
            
            
    #         for varname, w in weights.items():
    #             key = varname_to_key(varname)
    #             origin_port, origin_time, destination_port, destination_time, vc = key_to_node_info(key)
    #             origin_node = NODE_DICT[(origin_port, origin_time)]
    #             if origin_node_agent == origin_node and destination_port_number_agent == destination_port and arc.vessel_class == vc:
    #                 weights[varname] = 1
    # else:
    #     for vessel, action in actions.items():
    #         arc = action[3]
    #         varname = convert_key_to_varname((arc.tuple, arc.vessel_class))
    #         if arc in active_arcs_chosen_by_agent:
    #             weights[varname] = 1
    #         else:
    #             weights[varname] = 0
   
    return weights
            
        
        
# def perform_proximity_search(ps_data, RUNNING_NPS_AND_RH, window_end):  # Add other parameters as needed
    
#     model = ps_data['model']
#     vessel_class_arcs = ps_data['vessel_class_arcs']
#     model.setParam(gp.GRB.Param.SolutionLimit, 1)
#     model.setParam(gp.GRB.Param.OutputFlag, 0)
#     model.optimize()
#     original_objective_function = model.getObjective()
#     current_solution_vals_x = get_current_x_solution_vals(model)
#     current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
#     active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
#     current_solution_alpha = get_current_alpha_solution_vals(model)
#     current_solution_s = get_current_s_solution_vals(model)
#     current_best_obj = model.getObjective().getValue()
    
#     PERCENTAGE_DECREASE = 0.005
#     INFEASIBILITY_MULTIPLIER = 0.1
#     cutoff_value = (PERCENTAGE_DECREASE * current_best_obj)
    
#     y = model.addVars(current_solution_vars_x.keys(), vtype=GRB.BINARY, name="y")
#     i = 0
#     has_been_infeasible = False
    
#     # Start loop here:
#     while True:
#         if has_been_infeasible:
#             model.setParam()
#         i += 1
#         print_info = (i % 10 == 0)
            
#         if print_info or has_been_infeasible:
#             print(f'Proximity Search Iteration {i}')
        
#         # Set the objective (back) to the original objective function
#         model.setObjective(original_objective_function, GRB.MINIMIZE)
#         model.update()

#         cutoff_value = (PERCENTAGE_DECREASE * current_best_obj)
#         add_cutoff_constraint(model, current_best_obj, cutoff_value)

#         update_objective_to_minimize_hamming_distance(model, y, current_solution_vars_x, current_solution_vals_x, None)
#         model.setParam(gp.GRB.Param.SolutionLimit, 1)
#         model.setParam(gp.GRB.Param.TimeLimit, 300)
#         model.optimize()

#         # Check if a new solution is found with the lowest amount of changes to the structure as possible
#         if model.Status ==  GRB.SOLUTION_LIMIT:
#             if print_info or has_been_infeasible:
#                 print("Hamming Distance from previous solution:", model.objVal)
            
#             current_solution_vals_x = get_current_x_solution_vals(model)

#             if RUNNING_NPS_AND_RH:
#                 new_obj_value = evaluate_solution_with_original_objective_for_RH(model, ps_data, window_end)
#             else:
#                 new_obj_value = evaluate_solution_with_original_objective(model, ps_data, print_info, has_been_infeasible)
#             if print_info or has_been_infeasible:
#                 print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')
                
#             current_best_obj = new_obj_value
#             current_solution_vars_x = get_current_x_solution_vars(model)
#             current_solution_alpha = get_current_alpha_solution_vals(model)
#             current_solution_s = get_current_s_solution_vals(model)

            
#         elif model.Status == GRB.TIME_LIMIT or model.Status == GRB.INFEASIBLE:
#             has_been_infeasible = True
#             PERCENTAGE_DECREASE *= INFEASIBILITY_MULTIPLIER
#             # Return the best solution found. Time limit reached.
#             print(f'Best objective value is {current_best_obj}.')
#             # return current_solution, current_best_obj, model
#             return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
        
def find_corresponding_action_MIRPSO(possible_actions, dport_chosen_by_agent, active_arcs, vessel, legal_actions, traveling_arcs_taken, waiting_arcs_taken):
    vc = vessel.vessel_class
    # If waiting is correct, traveling arcs taken will be empty for the next vessel, meaning there is a possibility of choosing the wrong arc for the next vessel.
    # Issue if two or more vessels are at the same port at the same time and one or more of them are waiting but not all. 
    
    for action in possible_actions:
        possible_arc = action[3]
        for arc in active_arcs[vc]:
            if is_similar_arc(arc, possible_arc) and arc.is_waiting_arc == False:
                if any(is_similar_arc(arc, taken_arc) for taken_arc in traveling_arcs_taken[vc]):
                    print('Active, but already taken by another vessel')
                    continue
                return action, True
            # if arc.origin_node == possible_arc.origin_node and arc.destination_node == possible_arc.destination_node and arc.vessel_class == vc and arc.is_waiting_arc == False:
            #     if arc in traveling_arcs_taken[vc]:
                    # continue
                    
    a = 2     
    # Sligthly wrong to do it like this, but now trvaleing arcs are forced to be chosen before waiting, ensuring all traveling arcs are taken. Fewer weights will be set to one, but it is not a big issue.
    for action in legal_actions:
        possible_arc = action[3]
        for arc in active_arcs[vc]:
            if is_similar_arc(arc, possible_arc) and arc.is_waiting_arc == False:
                if any(is_similar_arc(arc, taken_arc) for taken_arc in traveling_arcs_taken[vc]):
                    print('Active, but already taken by another vessel')
                    continue
                return action, False
    
    a = 1
    for action in possible_actions:
        possible_arc = action[3]
        for arc in active_arcs[vc]:
            if arc.is_waiting_arc:
                if arc.origin_node == possible_arc.origin_node and arc.destination_node == possible_arc.destination_node and arc.vessel_class == vc:
                    return action, True
        
    
            
            # if arc.origin_node == possible_arc.origin_node and arc.vessel_class == vc and arc.is_waiting_arc == False and arc.destination_node == possible_arc.destination_node:
            #     if arc in traveling_arcs_taken[vc]:
            #         continue
     
    a = 3       
    for action in legal_actions:
        possible_arc = action[3]
        if possible_arc.is_waiting_arc:
            for arc in active_arcs[vc]:
                if arc.is_waiting_arc:
                    if arc.origin_node == possible_arc.origin_node and arc.vessel_class == vc:
                        return action, False
    
    # If no arcs were 
                    
    return None, False
        
    


def find_corresponding_action(action, active_arcs, vessel, active_arcs_chosen_by_agent):
    vc = vessel.vessel_class
    agent_arc = action[3]
    
    potential_active_arcs = [arc for arc in active_arcs[vc] if arc.origin_node == agent_arc.origin_node]
    if not potential_active_arcs:
        return None
    
    # Find the potential arcs that the vessel can take
    for arc in potential_active_arcs:
        if arc.destination_node.port == agent_arc.destination_node.port:
            # A potential arc with same vessel class, origin and destination port was found
            active_arcs_chosen_by_agent.append(agent_arc)
            return action
    
    # If no corresponding arc was found, we have to make the action based on the arcs that are active
    for arc in potential_active_arcs:
        # We now now that the vessel can take the arc, but the arc is not the one chosen by the agent
        # Find the operation type
        if arc.is_waiting_arc:
            # Generate legal actions for waiting arcs
            operation_type = 0
            quantity = 0
            action = (vessel.number, operation_type, quantity, arc)
        elif arc.destination_node.port.isLoadingPort != 1:
            # We are at a loading port and loading the vessel
            operation_type = 1
            quantity = vessel.capacity
            action = (vessel.number, operation_type, quantity, arc)
        else:
            # We are at a discharging port and unloading the vessel
            operation_type = 2
            quantity = vessel.capacity
            action = (vessel.number, operation_type, quantity, arc)
        return action
    
    return None

def is_similar_arc(arc1, arc2):
    return (arc1.origin_node == arc2.origin_node and
            arc1.destination_node == arc2.destination_node and
            arc1.vessel_class == arc2.vessel_class)


def init_weights(active_arcs, vessel_classes, current_solution_x, vessel_class_arcs):
    weights = {}
    for vc in vessel_classes:
        for arc in vessel_class_arcs[vc]:
            varname = convert_key_to_varname((arc.tuple, vc))
            weights[varname] = 0
    return weights
                
def calculate_weights(current_solution_x, current_solution_alpha, current_solution_s, active_arcs, agent, env, vessel_classes, RUNNING_MIRPSO, NODE_DICT, vessel_class_arcs):
    weights = init_weights(active_arcs, vessel_classes, current_solution_x, vessel_class_arcs)
        
    active_arcs_chosen_by_agent = []
    state = env.reset()
    experience_path = []
    decision_basis_states = {vessel.number: env.custom_deep_copy_of_state(state) for vessel in state['vessels']}
    actions = {vessel: env.find_legal_actions_for_vessel(state=state, vessel=vessel, random_start = False)[0] for vessel in state['vessels']}
    for vessel, action in actions.items():
        arc = action[3]
        active_arcs_chosen_by_agent.append(arc)
    weights = update_weights_based_on_action(weights, active_arcs_chosen_by_agent)
    
    state = env.step(state=state, actions=actions, experience_path=experience_path, decision_basis_states=decision_basis_states)
    done = False
    
    while not done:
        state['time'] += 1
        time = state['time']
        # Use the s-variables to update the port inventory levels in the state
        # Find the keys for the s-variables that correspond to the current time period
        for port in env.PORTS:
            var_name = f's[{port.number},{time}]'
            port.inventory_level = current_solution_s[var_name]
        state = env.produce(state)
        
        if state['time'] == env.TIME_PERIOD_RANGE[-1]:
            done = True
        
         # With the increased time, the vessels have moved and some of them have maybe reached their destination. Updating the vessel status based on this.
        env.update_vessel_status(state=state)
        # Find the vessels that are available to perform an action
        available_vessels = env.find_available_vessels(state=state)
        
        if available_vessels:
            traveling_arcs_taken = {vc : [] for vc in vessel_classes}
            waiting_arcs_taken = {vc : [] for vc in vessel_classes}
            actions = {}
            actions_to_make = {}
            active_arcs_chosen_by_agent = []
            decision_basis_states = {}
            decision_basis_state = env.custom_deep_copy_of_state(state)
            
            for vessel in available_vessels:
                corresponding_vessel = decision_basis_state['vessel_dict'][vessel.number]
                decision_basis_states[corresponding_vessel['number']] = decision_basis_state
                legal_actions = env.sim_find_legal_actions_for_vessel(state=decision_basis_state, vessel=corresponding_vessel, queued_actions=actions_to_make, RUNNING_WPS = True)
                
                if RUNNING_MIRPSO:
                    possible_actions, dport_chosen_by_agent = agent.select_action_for_ps(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, RUNNING_MIRPSO=RUNNING_MIRPSO)
                else:
                    action = agent.select_action_for_ps(state=copy.deepcopy(decision_basis_state), legal_actions=legal_actions, env=env, vessel_simp=corresponding_vessel, RUNNING_MIRPSO=RUNNING_MIRPSO)
                    
                if action is None:
                    print('Action is None')
                # Now I must find the corresponding arc that should be taken, and not the action that the agent selected
                if RUNNING_MIRPSO:
                    # If one of the possible actions is in the active arcs, we should choose this action
                    correct_action, correct_choice_by_agent = find_corresponding_action_MIRPSO(possible_actions, dport_chosen_by_agent, active_arcs, vessel, legal_actions, traveling_arcs_taken, waiting_arcs_taken)
                    correct_arc = correct_action[3]
                    if not correct_arc.is_waiting_arc:
                        traveling_arcs_taken[vessel.vessel_class].append(correct_arc)
                    else:
                        waiting_arcs_taken[vessel.vessel_class].append(correct_arc)
                    if correct_choice_by_agent:
                        active_arcs_chosen_by_agent.append(correct_arc)
                        
                    actions_to_make[vessel] = correct_action
                    
                    decision_basis_state = env.update_vessel_in_transition_and_inv_for_state(state = decision_basis_state, vessel = corresponding_vessel, destination_port = correct_arc.destination_node.port, destination_time = correct_arc.destination_node.time, origin_port = correct_arc.origin_node.port, quantity = correct_action[2], operation_type = correct_action[1])
                    
                else:
                    actions[vessel] = action
                    action_chosen_by_model = find_corresponding_action(action, active_arcs, vessel, active_arcs_chosen_by_agent)
                    model_arc = action_chosen_by_model[3]
                    actions_to_make[vessel] = action_chosen_by_model
                    
                    decision_basis_state = env.update_vessel_in_transition_and_inv_for_state(state = decision_basis_state, vessel = corresponding_vessel, destination_port = model_arc.destination_node.port, destination_time = model_arc.destination_node.time, origin_port = model_arc.origin_node.port, quantity = action_chosen_by_model[2], operation_type = action_chosen_by_model[1])
                            
                    for vessel, action in actions.items():
                        arc = action[3]
                        active_arcs_chosen_by_agent.append(arc)
                         
            # Now we have to update the weights based on the actions taken
            weights = update_weights_based_on_action(weights, active_arcs_chosen_by_agent)
            
            
            # Perform the operations based on the active arcs, and not the actions taken by the agent
            # for action in legal_actions:
            #     arc = action[3]
            #     if arc in active_arcs:
            #         actions_to_make[vessel] = action
            
            state = env.step(state=state, actions=actions_to_make, experience_path=experience_path, decision_basis_states=decision_basis_states)
            # Perform the operation and routing actions and update the state based on this
        else:
            # Should check the feasibility of the state, even though no actions were performed. 
            state = env.simple_step(state, experience_path)
        
        state = env.consumption(state)
        
    # print the weights that are not 0
    weight_counter = 0
    for varname, weight in weights.items():
        if weight != 0:
            weight_counter += 1
    print(f'Number of weights set to 1: {weight_counter}')
    return weights
        
def solve_model(model, costs, P, RUNNING_MIRPSO, INSTANCE, HORIZON, ps_data, time_limit):
    vessel_class_arcs = ps_data['vessel_class_arcs']

    x_vars_counter = 0
    for var in model.getVars():
        if var.VarName.startswith("x"):
            x_vars_counter += 1
    print(f"Number of x variables: {x_vars_counter}")
    
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
    
    model.printStats()
    
    flow_counter = 0
    port_balance_counter = 0
    berth_limit_counter = 0
    alpha_slack_bound_counter = 0
    source_flow_counter = 0
    sink_flow_counter = 0
    InitialInventoryPort_counter = 0
    
    for con in model.getConstrs():
        if con.ConstrName.startswith("Flow"):
            flow_counter += 1
        if con.ConstrName.startswith("PortBalance"):
            port_balance_counter += 1
        if con.ConstrName.startswith("Berth_limit"):
            berth_limit_counter += 1
        if con.ConstrName.startswith("AlphaSlackBound"):
            alpha_slack_bound_counter += 1
        if con.ConstrName.startswith("SourceFlow"):
            source_flow_counter += 1
        if con.ConstrName.startswith("SinkFlow"):
            sink_flow_counter += 1
        if con.ConstrName.startswith("InitialInventoryPort"):
            InitialInventoryPort_counter += 1

    print(f"Number of flow constraints: {flow_counter}")
    print(f"Number of port balance constraints: {port_balance_counter}")
    print(f"Number of berth limit constraints: {berth_limit_counter}")
    print(f"Number of alpha slack bound constraints: {alpha_slack_bound_counter}")
    print(f"Number of source flow constraints: {source_flow_counter}")
    print(f"Number of sink flow constraints: {sink_flow_counter}")
    print(f"Number of Initial Inventory Port constraints: {InitialInventoryPort_counter}")
    
    sum = flow_counter + port_balance_counter + berth_limit_counter + alpha_slack_bound_counter + source_flow_counter + sink_flow_counter + InitialInventoryPort_counter
    print("Number of constraints with names:", sum)
    print("Number of constraints in model:", len(model.getConstrs()))
    
    model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    try:
        print(f'Running {INSTANCE} with {HORIZON} time periods')
        model.optimize()
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        
    # Return objective function value, the current solution, the current solution values, the current solution s-values, the current solution alpha-values and the active arcs
    current_solution_vals_x = get_current_x_solution_vals(model)
    current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
    current_solution_s = get_current_s_solution_vals(model)
    current_solution_alpha = get_current_alpha_solution_vals(model)
    active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
    return model.objVal, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
    
        
def active_arcs_for_vc(vc, x_solution):
    active_arcs = {}
    
    # Iterate through the dictionary entries
    for varname, value in x_solution.items():
        if value != 0.0:  # Only consider transitions where a journey occurred
            key = varname_to_key(varname)
            arc_tuple, vessel_class = key
            if vessel_class == vc:
                origin_node, destination_node = arc_tuple
                active_arcs[arc_tuple] = round(value)
    return active_arcs                

import random
def find_vessel_routes(x_solution, vessels_in_class, sourceNode, sinkNode, node_dict, vc):
    active_arcs_for_vessel_class = active_arcs_for_vc(vc, x_solution)
    # Dictionary to store the routes for each vessel
    routes = {vessel.number : [] for vessel in vessels_in_class}

    for vessel in vessels_in_class:
        vessel_at_sink = False
        route = []
        # Every vessel starts at sourceNode
        start_node = node_dict[(vessel.initial_port.number, vessel.first_time_available)]
        arc_tuple = (sourceNode.tuple, start_node.tuple)
        # Find the arc from source to start node in active_arcs_for_vessel_class
        if arc_tuple in active_arcs_for_vessel_class:
            value = active_arcs_for_vessel_class[arc_tuple]
            if value == 0:
                raise ValueError(f'Vessel {vessel.vessel_number} has a non-positive value for the arc')
            else:
                current_node = start_node.tuple
                active_arcs_for_vessel_class[arc_tuple] -= 1
                # if active_arcs_for_vessel_class[arc_tuple] == 0:
                    # Remove the key and value from active_arcs_for_vessel_class
                    # del active_arcs_for_vessel_class[arc_tuple]
                route.append(arc_tuple)
        else:
            raise ValueError(f'{arc_tuple} not in active arcs for vessel class {vc} for vessel {vessel.vessel_number}')
        
        while not vessel_at_sink:
            # Find the next node in the route
            # next_node = None
            for arc_tuple, value in active_arcs_for_vessel_class.items():
                origin_node, _ = arc_tuple
                if origin_node == current_node:
                     # # Find all arcs from the origin node with value > 0
                    arcs_from_origin = [arc for arc in active_arcs_for_vessel_class.keys() if arc[0] == origin_node]
                    # Only keep the ones wiht value > 0
                    arcs_from_origin = [arc for arc in arcs_from_origin if active_arcs_for_vessel_class[arc] > 0]
                    if not arcs_from_origin:
                        raise ValueError(f'Vessel {vessel.number} has no legal actions form the current node')
                    else:
                        # Random choice from the arcs
                        arc_tuple = random.choice(arcs_from_origin)
                        origin_node, destination_node = arc_tuple
                        # Update the current node to the destination node
                        current_node = destination_node
                        
                        if current_node == sinkNode.tuple:
                            vessel_at_sink = True
                            
                        active_arcs_for_vessel_class[arc_tuple] -= 1
                        # if active_arcs_for_vessel_class[arc_tuple] == 0:
                            # Remove the key and value from active_arcs_for_vessel_class
                            # del active_arcs_for_vessel_class[arc_tuple]
                        route.append(arc_tuple)
                        break
                        
        routes[vessel.number] = route
            
    # Check if all variables in active_arcs_for_vessel_class are 0
    for arc_tuple, value in active_arcs_for_vessel_class.items():
        if value != 0:
            raise ValueError(f'Value for arc {arc_tuple} is not 0')
        
    return routes

def reformat_routes(routes_for_vessel_classes, vessels):
    reformat_routes = {}
    
    for vessel in vessels:
        vc = vessel.vessel_class
        route = []
        for arc_tuple in routes_for_vessel_classes[vc][vessel.number]:
            # origin_port, origin_time = arc_tuple[0]
            # destination_port, destination_time = arc_tuple[1]
            route.append([arc_tuple[0], arc_tuple[1]])
        reformat_routes[vessel.number] = route
    return reformat_routes


def flatten_vessel_routes(vessel_routes):
    flattened_routes = {}
    for vessel_id, routes in vessel_routes.items():
        # Flatten each sequence of movements
        flattened_list = [coord for movement in routes for coord in movement]
        # Ensure unique consecutive coordinates are not duplicated
        unique_flattened = [flattened_list[i] for i in range(len(flattened_list)) if i == 0 or flattened_list[i] != flattened_list[i-1]]
        flattened_routes[vessel_id] = unique_flattened
    return flattened_routes

    
    


import json
import gurobipy as gp
import logging

def save_solution(INSTANCE, HORIZON, model, time_limit, elapsed_time, costs, P,
                  RUNNING_MIRPSO, best_obj_val, solution_vals_x,
                  solution_vals_s, solution_vals_alpha, active_arcs, vessel_classes,
                  NPS, WPS, RH, WPS_AND_RH, NPS_AND_RH, VESSEL_CLASSES, vessels, sourceNode, sinkNode, NODE_DICT):
    # If the model is not optimal store the gap also.
    

    method_map = {"NPS": NPS, "WPS": WPS, "RH": RH, "WPS_AND_RH": WPS_AND_RH, "NPS_AND_RH": NPS_AND_RH}
    method = next((method_name for method_name, condition in method_map.items() if condition), "SOLVER")
    
    status_map = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF"
    }
    model_status = status_map.get(model.status, "UNKNOWN")

    if model.status == gp.GRB.INFEASIBLE:
        model.computeIIS()
        model.write("MIRPSO.ilp")
        logging.error("Model is infeasible")
        return

    # Convert tuple keys to strings
    costs_str_keys = {str(k): v for k, v in costs.items()}
    P_str_keys = {str(k): v for k, v in P.items()}
    
    

    solution_data = {
        "instance": INSTANCE,
        "horizon": HORIZON,
        'method': method,
        "status": model_status,
        "objective_value": best_obj_val,
        "elapsed_time": elapsed_time,
        "time_limit": time_limit if model.status == gp.GRB.TIME_LIMIT else None,
        "costs": costs_str_keys,
        "P": P_str_keys,
        "solution_x": {str(var_name): var_val for var_name, var_val in solution_vals_x.items()},
        "solution_s": {str(k): v for k, v in solution_vals_s.items()},
        "solution_alpha": {str(k): v for k, v in solution_vals_alpha.items()},
        "active_arcs": [{**arc.to_dict(), "origin_node": str(arc.origin_node), "destination_node": str(arc.destination_node)} for vc in vessel_classes for arc in active_arcs[vc]]
    }
    
    routes_for_vessel_classes = {}
    for vc in VESSEL_CLASSES:
        vessels_in_class = [v for v in vessels if v.vessel_class == vc]
        routes_for_vessel_classes[vc] = find_vessel_routes(solution_data['solution_x'], vessels_in_class, sourceNode, sinkNode, NODE_DICT, vc)
        
    solution_data['vessel_routes'] = routes_for_vessel_classes
    
    reformatted_routes = reformat_routes(routes_for_vessel_classes, vessels)
        
    reformatted_routes = flatten_vessel_routes(reformatted_routes)
        
    flattened_routes = {}
    flattened_routes["vessel_routes_flattened"] = reformatted_routes
    

  
    
    
    # Save to a JSON file for easy parsing and readability
    filename = f"group2_solutions/{'MIRPSO' if RUNNING_MIRPSO else 'MIRP'}_GROUP_2_{INSTANCE}_{HORIZON}_{method}.json"
    filename_readable_routes = f"group2_routes/{'MIRPSO' if RUNNING_MIRPSO else 'MIRP'}_GROUP_2_{INSTANCE}_{HORIZON}_{method}_routes.txt"
    with open(filename, 'w') as file:
        json.dump(solution_data, file, indent=4)
    with open(filename_readable_routes, 'w') as file:
        for key, value in flattened_routes['vessel_routes_flattened'].items():
            file.write('%s: %s\n' % (key, value))
            
        file.close()
            
        # json.dump(flattened_routes, file, indent=None)


def proximity_search_using_agent(ps_data, agent, env, RUNNING_WPS_AND_RH, window_end, RUNNING_MIRPSO, time_limit):

    WEIGHT_UPDATE_FREQUENCY = 10
    PERCENTAGE_DECREASE = 0.005
    INFEASIBILITY_MULTIPLIER = 0.1
    
    model = ps_data['model']
    NODE_DICT = ps_data['node_dict']
    vessel_class_arcs = ps_data['vessel_class_arcs']
    original_objective_function = model.getObjective()
    model.setParam(gp.GRB.Param.SolutionLimit, 1)
    
    if not RUNNING_WPS_AND_RH:
        "Do not use construction heuristic in RH"
        experience_path, port_inventory_dict, vessel_inventory_dict = evaluate_agent(env, agent)
        active_X_keys, S_values, alpha_values = convert_path_to_MIRPSO_solution(env, experience_path, port_inventory_dict)
        current_solution_vals_x, model, warm_start_sol = warm_start_model(model, active_X_keys, S_values, alpha_values)
        current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
        # active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        y = model.addVars(current_solution_vars_x.keys(), vtype=GRB.BINARY, name="y")
        update_objective_to_minimize_hamming_distance(model, y, current_solution_vars_x, current_solution_vals_x, None)
        
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        model.setParam(gp.GRB.Param.TimeLimit, 300)
        start_time = time.time()
        model.optimize()
        model.setParam(gp.GRB.Param.OutputFlag, 0)
        
        current_best_obj = evaluate_solution_with_original_objective(model, ps_data, True, False)
        current_solution_vals_x = get_current_x_solution_vals(model)
        current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
        active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        current_solution_alpha = get_current_alpha_solution_vals(model)
        current_solution_s = get_current_s_solution_vals(model)
        vessel_classes = vessel_class_arcs.keys()
        
    else:
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        start_time = time.time()
        time_left = time_limit - time_passed(start_time)
        model.setParam(gp.GRB.Param.TimeLimit, min(300, time_left) if time_left >= 0 else 10)
        model.setParam(gp.GRB.Param.OutputFlag, 1)
        # Remove y variables from the model if any
        y_vars = [v for v in model.getVars() if v.VarName.startswith('y')]
        # Remove
        for v in y_vars:
            model.remove(v)
        model.update()
        # Remove all constraints starting with y
        y_constraints = [c for c in model.getConstrs() if c.ConstrName.startswith('y')]
        for c in y_constraints:
            model.remove(c)
        model.update()
            
        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            #Write ilp file
            model.computeIIS()
            model.write("MIRPSO.ilp")
            
        model.setParam(gp.GRB.Param.OutputFlag, 0)
        current_best_obj = model.getObjective().getValue()
        current_solution_vals_x = get_current_x_solution_vals(model)
        current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
        active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
        current_solution_alpha = get_current_alpha_solution_vals(model)
        current_solution_s = get_current_s_solution_vals(model)
        vessel_classes = vessel_class_arcs.keys()
        y = model.addVars(current_solution_vars_x.keys(), vtype=GRB.BINARY, name="y")
    
    
    'Calculate the weights based on the initial solution'
    weights = calculate_weights(current_solution_vals_x, current_solution_alpha, current_solution_s, active_arcs, agent, env, vessel_classes, RUNNING_MIRPSO, NODE_DICT, vessel_class_arcs)
    
    has_been_infeasible = False
    i = 0
    
    while True:
        if has_been_infeasible:
            model.setParam(gp.GRB.Param.OutputFlag, 1)
        i += 1
        
        PRINT_INFO = (i % 10 == 0)
        CALCULATE_WEIGHTS = (i % WEIGHT_UPDATE_FREQUENCY == 0)
        if PRINT_INFO or has_been_infeasible:
            print(f'Proximity Search Iteration {i}')
        
        # Set the objective (back) to the original objective function
        model.setObjective(original_objective_function, GRB.MINIMIZE)
        model.update()

        cutoff_value = (PERCENTAGE_DECREASE * current_best_obj)
        add_cutoff_constraint(model, current_best_obj, cutoff_value)
        
        update_objective_to_minimize_hamming_distance(model, y, current_solution_vars_x, current_solution_vals_x, weights)
        model.setParam(gp.GRB.Param.SolutionLimit, 1)
        # model.setParam(gp.GRB.Param.TimeLimit, 30)
        time_left =  time_limit - time_passed(start_time)
        model.setParam(gp.GRB.Param.TimeLimit, min(300, time_left) if time_left >= 0 else 10)
            
        
        model.optimize()

        # Check if a new solution is found with the lowest amount of changes to the structure as possible
        if model.Status == GRB.SOLUTION_LIMIT or model.Status == GRB.OPTIMAL:
            if PRINT_INFO or has_been_infeasible:
                print("Hamming Distance from previous solution:", model.objVal)
            
            if RUNNING_WPS_AND_RH:
                new_obj_value = evaluate_solution_with_original_objective_for_RH(model, ps_data, window_end)
            else:
                new_obj_value = evaluate_solution_with_original_objective(model, ps_data, PRINT_INFO, has_been_infeasible)
                
            if PRINT_INFO or has_been_infeasible:            
                print(f'Previous objective value was {current_best_obj}. New objective value: {new_obj_value}')

            current_best_obj = new_obj_value
            current_solution_vals_x = get_current_x_solution_vals(model)
            current_solution_vars_x = {v.VarName: v for v in model.getVars() if v.VarName.startswith('x')}
            active_arcs = find_corresponding_arcs(current_solution_vals_x, vessel_class_arcs)
            current_solution_alpha = get_current_alpha_solution_vals(model)
            current_solution_s = get_current_s_solution_vals(model)
            vessel_classes = vessel_class_arcs.keys()
            if time.time() - start_time > time_limit:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            if CALCULATE_WEIGHTS:
                weights = calculate_weights(current_solution_vals_x, current_solution_alpha, current_solution_s, active_arcs, agent, env, vessel_classes, RUNNING_MIRPSO, NODE_DICT, vessel_class_arcs)
        
        elif model.Status == GRB.TIME_LIMIT or model.Status == GRB.INFEASIBLE:
            has_been_infeasible = True
            # Reduce the cutoff value and try again
            PERCENTAGE_DECREASE *= INFEASIBILITY_MULTIPLIER
            if cutoff_value < 1:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            elif time.time() - start_time > time_limit:
                remove_cutoff_constraints(model)
                return current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs
            
def calc_time_limit_per_window(TIME_LIMIT, HORIZON, window_size, step_size):
    iterations = (HORIZON - window_size)//(step_size) + 1
    return TIME_LIMIT / iterations
            

def main():
    RUNNING_MIRPSO = False
    RUNNING_WEIGHTED_PS = False
    RUNNING_NORMAL_PS = False
    RUNNING_RH = False
    RUNNING_NPS_AND_RH = False
    RUNNING_WPS_AND_RH = False
    
    # RUNNING_MIRPSO = True
    
    RUNNING_WEIGHTED_PS = True
    # RUNNING_NORMAL_PS = True
    # RUNNING_RH = True
    # RUNNING_NPS_AND_RH = True
    # RUNNING_WPS_AND_RH = True
    
    # GROUP 2
    INSTANCE = 'LR1_DR02_VC01_V6a'
    # INSTANCE = 'LR1_DR02_VC02_V6a'
    # INSTANCE = 'LR1_DR02_VC03_V7a'
    # INSTANCE = 'LR1_DR02_VC03_V8a'
    # INSTANCE = 'LR1_DR02_VC04_V8a'
    # INSTANCE = 'LR1_DR02_VC05_V8a'
    # INSTANCE = 'LR1_DR03_VC03_V10b'
    
    TIME_LIMIT = 30
    
    problem_data = build_problem(INSTANCE, RUNNING_MIRPSO)
    
    # Unpack the problem data
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    arc_dict = problem_data['arc_dict']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    HORIZON = len(TIME_PERIOD_RANGE)
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    VESSEL_CLASSES = problem_data['VESSEL_CLASSES']
    NODE_DICT = problem_data['NODE_DICT']
    vessel_class_capacities = problem_data['vessel_class_capacities']
    NODES = problem_data['NODES']
    special_sink_arcs = problem_data['special_sink_arcs']
    special_nodes_dict = problem_data['special_nodes_dict']
    
    origin_node_arcs, destination_node_arcs, vessel_class_arcs = rearrange_arcs(arc_dict)
    
    model, costs, P, costs_namekey = build_model(vessels = vessels, regularNodes=regularNodes, ports=ports, TIME_PERIOD_RANGE = TIME_PERIOD_RANGE, sourceNode = sourceNode, sinkNode = sinkNode, vessel_classes = VESSEL_CLASSES, origin_node_arcs = origin_node_arcs, destination_node_arcs=destination_node_arcs, vessel_class_arcs=vessel_class_arcs, NODE_DICT = NODE_DICT, vessel_class_capacities=vessel_class_capacities)
    
    model.write("Group2.lp")  # Export model to an LP file
    
    ps_data = {'model': model, 'costs': costs, 'P':P, 'regularNodes': regularNodes, 'vessels': vessels, 'vessel_arcs': vessel_arcs, 'costs_namekey':costs_namekey, 'vessel_class_arcs': vessel_class_arcs, 'node_dict': NODE_DICT}
    
    # Take the time of solving the model
    start_time = time.time()
    if RUNNING_WEIGHTED_PS:
        env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode, NODE_DICT, special_sink_arcs, special_nodes_dict)
        replay = ReplayMemory(3000)
        
        agent = DQNAgent(ports = ports, vessels=vessels, TRAINING_FREQUENCY = None, TARGET_UPDATE_FREQUENCY = None, NON_RANDOM_ACTION_EPISODE_FREQUENCY = None, BATCH_SIZE = None, replay = replay)
        replay = agent.load_replay_buffer(file_name= 'replay_buffer_8apr_nt3_50_1664.pkl')
        agent.memory = replay
        
        agent.main_model.load_state_dict(torch.load('main_model_8apr_nt3_50_1664.pth'))
        agent.target_model.load_state_dict(torch.load('target_model_8_apr_nt3_50_1664.pth'))
        current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs = proximity_search_using_agent(ps_data=ps_data, agent=agent, env=env, RUNNING_WPS_AND_RH=RUNNING_WPS_AND_RH, window_end=None, RUNNING_MIRPSO=RUNNING_MIRPSO, time_limit=TIME_LIMIT)

        
        
    elif RUNNING_NORMAL_PS:
        # Perform the proximity search using the initial solution
        current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs = perform_proximity_search(ps_data=ps_data, RUNNING_NPS_AND_RH=RUNNING_NPS_AND_RH, window_end=None, time_limit=TIME_LIMIT)
        
    elif RUNNING_RH or RUNNING_NPS_AND_RH:
        # Define your window size and execute rolling horizon optimization
        window_size = 60  # example window size
        step_size = 30
        TIME_LIMIT_PER_WINDOW = calc_time_limit_per_window(TIME_LIMIT, HORIZON, window_size, step_size)
        current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs = rolling_horizon_optimization(model=model, horizon_length=HORIZON, window_size=window_size, step_size=step_size, TIME_LIMIT_PER_WINDOW=TIME_LIMIT_PER_WINDOW, ps_data=ps_data, agent=None, env=None, RUNNING_WPS_AND_RH=RUNNING_WPS_AND_RH, RUNNING_NPS_AND_RH=RUNNING_NPS_AND_RH, proximity_search_using_agent=proximity_search_using_agent, RUNNING_MIRPSO=RUNNING_MIRPSO)
        # Print objective function value
        print(f'Objective function value: {current_best_obj}')
        
    elif RUNNING_WPS_AND_RH:
        env = MIRPSOEnv(ports, vessels, vessel_arcs, NODES, TIME_PERIOD_RANGE, sourceNode, sinkNode, NODE_DICT, special_sink_arcs, special_nodes_dict)
        replay = ReplayMemory(3000)
        
        agent = DQNAgent(ports = ports, vessels=vessels, TRAINING_FREQUENCY = None, TARGET_UPDATE_FREQUENCY = None, NON_RANDOM_ACTION_EPISODE_FREQUENCY = None, BATCH_SIZE = None, replay = replay)
        replay = agent.load_replay_buffer(file_name= 'replay_buffer_8apr_nt3_50_1664.pkl')
        agent.memory = replay
        
        agent.main_model.load_state_dict(torch.load('main_model_8apr_nt3_50_1664.pth'))
        agent.target_model.load_state_dict(torch.load('target_model_8_apr_nt3_50_1664.pth'))
        
        window_size = 60  # example window size
        step_size = 30
        TIME_LIMIT_PER_WINDOW = calc_time_limit_per_window(TIME_LIMIT, HORIZON, window_size, step_size)
        current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs = rolling_horizon_optimization(model =model, horizon_length=HORIZON, window_size=window_size, step_size=step_size, TIME_LIMIT_PER_WINDOW=TIME_LIMIT_PER_WINDOW, ps_data=ps_data, agent=agent, env=env, RUNNING_WPS_AND_RH=RUNNING_WPS_AND_RH, RUNNING_NPS_AND_RH=RUNNING_NPS_AND_RH, proximity_search_using_agent=proximity_search_using_agent, RUNNING_MIRPSO=RUNNING_MIRPSO)
        
    else:
        current_best_obj, current_solution_vars_x, current_solution_vals_x, current_solution_s, current_solution_alpha, active_arcs = solve_model(model=model, costs=costs, P=P, RUNNING_MIRPSO=RUNNING_MIRPSO, INSTANCE= INSTANCE, HORIZON=HORIZON, ps_data=ps_data, time_limit=TIME_LIMIT)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    save_solution(INSTANCE=INSTANCE, HORIZON=HORIZON, model = model, time_limit = TIME_LIMIT, elapsed_time=elapsed_time, costs=costs, P=P,
                  RUNNING_MIRPSO=RUNNING_MIRPSO, best_obj_val=current_best_obj, solution_vals_x = current_solution_vals_x,
                  solution_vals_s = current_solution_s, solution_vals_alpha = current_solution_alpha, active_arcs = active_arcs, vessel_classes=VESSEL_CLASSES,
                  NPS = RUNNING_NORMAL_PS, WPS = RUNNING_WEIGHTED_PS, RH = RUNNING_RH, WPS_AND_RH = RUNNING_WPS_AND_RH, NPS_AND_RH=RUNNING_NPS_AND_RH, VESSEL_CLASSES = VESSEL_CLASSES, vessels = vessels, sourceNode = sourceNode, sinkNode = sinkNode, NODE_DICT = NODE_DICT)
    
    print('Finished')
    
if __name__ == "__main__":
    main()