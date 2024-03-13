import re
import gurobipy as gp
from gurobipy import GRB


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
    def __init__(self, origin_node, destination_node, distance, cost, travel_time, speed, is_waiting_arc, is_source_or_sink_arc, vessel_class):
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
            first_time_available = vessel_data['Vessel_'+str(name_num)]['First Time Available'] +1
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
    for t in range(1, NUM_TIME_PERIODS+1):
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
        # source_arcs_info_for_vessel = []
        speed = 0
        # speed, timesteps, departure, arrival, origin_port_number, destination_port_number, fuel_consumption, distance_nm, vessel, is_waiting_arc, is_source_or_sink_arc = tuple_data
        arc_info = [speed, first_time_available, 0, first_time_available, sourceNode.port.number, initial_port.number, 0, 0, vessel, False, True]
        # source_arcs_info_for_vessel.append(arc_info)
        source_arcs_info.append(arc_info)
    return [source_arcs_info]

def populate_all_info(source_arcs_info, vessels, ports, full_distance_matrix_nm, NODE_DICT, NUM_TIME_PERIODS, full_travel_cost_matrices, full_travel_time_matrices, port_dict):     
    # Initialize the all_info list with the source arcs info
    all_info = source_arcs_info.copy()
    
    # Append the all_info list with the arcs for each node
    for vessel in vessels:
        travel_cost_matrix = full_travel_cost_matrices[vessel.vessel_class]
        travel_time_matrix = full_travel_time_matrices[vessel.vessel_class]
        # For each node, create the outgoing arcs
        arcs_for_current_vessel = [create_arcs_for_node(node, vessel, [port.number for port in ports], full_distance_matrix_nm, ports, NUM_TIME_PERIODS, travel_cost_matrix, travel_time_matrix, port_dict) for node in NODE_DICT.values()]
        # Remove all empty lists from the arcs_for_current_vessel list
        arcs_for_current_vessel = [arc for arc in arcs_for_current_vessel if arc]
        # Flatten the two-level list and count all arcs
        num_arcs = len([arc for sublist in arcs_for_current_vessel for arc in sublist])
        print(f"Vessel {vessel.number} has {num_arcs} arcs")
        

        if arcs_for_current_vessel:
            all_info.extend(arcs_for_current_vessel)
    return all_info

def create_arcs_for_node(node, vessel, all_port_numbers, full_distance_matrix_nm, ports, NUM_TIME_PERIODS, travel_cost_matrix, travel_time_matrix, port_dict):
    node_arcs = []
    if node.port.number in [0, len(ports) + 1]:
        return node_arcs
    # if node.time < vessel.first_time_available:
        # return node_arcs

    for destination_port_number in all_port_numbers:
        destination_port = port_dict[destination_port_number]
        if destination_port_number == node.port.number:
            # Create waiting_arc
            # arc_info_matrix = create_arc_info(speed=0, minimum_timesteps=1, departure=node.time, origin_port=node.port.number, destination_port=destination_port_number, lowest_speed=0, distance_to_port=0, vessel=vessel, is_waiting_arc=True, NUM_TIME_PERIODS = NUM_TIME_PERIODS)
            arc_info = [0, 1, node.time, node.time+1, node.port.number, destination_port_number, 0, 0, vessel, True, False]
            node_arcs.append(arc_info)
        else:
            # Only create the arcs from a loading to a discharging port and vice versa
            if node.port.isLoadingPort != destination_port.isLoadingPort:
                # Create the arcs to the discharging ports
                distance_nm = full_distance_matrix_nm[node.port.number - 1][destination_port_number - 1]
                # Note that there is a difference between [origin, destination] and [destination, origin] in the travel cost matrix. Check that it is correct
                travel_cost = travel_cost_matrix[node.port.number - 1][destination_port_number - 1]
                travel_time = travel_time_matrix[node.port.number - 1][destination_port_number - 1]
                if travel_time>17:
                    print('here')
                # minimum_timesteps, speed = calculate_minimum_timesteps_and_speed(distance_nm=distance_nm, MAX_SPEED=MAX_SPEED, MIN_SPEED=MIN_SPEED)
                arrival_time = node.time + travel_time
                
                if arrival_time < vessel.first_time_available or arrival_time > NUM_TIME_PERIODS:
                    continue
                # arc_info = [(speed, travel_time, departure, arrival_time, origin_port, destination_port, travel_cost, distance_to_port, vessel, is_waiting_arc)]
                arc_info = [vessel.avg_speed, travel_time, node.time, arrival_time, node.port.number, destination_port_number, travel_cost, distance_nm, vessel, False, False]
                # arc_info_matrix = create_arc_info_fixed_speed(speed=vessel.avg_speed, minimum_timesteps=travel_time, departure=node.time, origin_port=node.port.number, destination_port=destination_port_number, distance_to_port=distance_nm, vessel=vessel, is_waiting_arc=False, NUM_TIME_PERIODS = NUM_TIME_PERIODS)
                node_arcs.append(arc_info)

    return node_arcs

def add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS):
    arc_dict = {}
    vessel_arcs = {}
    waiting_arcs = {}
    sink_and_source_arcs = {}

    for node in all_info:
        for arc in node:
            if not arc:
                continue

            # for tuple_data in arc:
            # Unpack the tuple data
            speed, travel_time, departure, arrival, origin_port_number, destination_port_number, travel_cost, distance_nm, vessel, is_waiting_arc, is_source_or_sink_arc= arc
            vessel_class = vessel.vessel_class

            # Retrieve the origin and destination nodes
            origin_node_obj = NODE_DICT.get((origin_port_number, departure))
            destination_node_obj = NODE_DICT.get((destination_port_number, arrival))
           

            if origin_node_obj and destination_node_obj and arrival <= NUM_TIME_PERIODS:
                # Initialize outgoing and incoming arcs for vessel classes if not already done
                origin_node_obj.outgoing_arcs.setdefault(vessel_class, [])
                destination_node_obj.incoming_arcs.setdefault(vessel_class, [])

                # Create the arc
                arc = Arc(origin_node=origin_node_obj, destination_node=destination_node_obj, distance=distance_nm, cost=travel_cost, travel_time=travel_time, speed=speed, is_waiting_arc=is_waiting_arc, is_source_or_sink_arc=is_source_or_sink_arc, vessel_class = vessel_class)
                
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

def add_arc_to_dict(origin, destination, vessel, arc_dict, vessel_arcs):
    """Helper function to add arc to dictionaries and nodes."""
    arc = Arc(origin_node=origin, destination_node=destination, distance=0, cost=0, travel_time=0, speed=0, is_waiting_arc=False, is_source_or_sink_arc=True, vessel_class=vessel.vessel_class)
    origin.outgoing_arcs.setdefault(vessel.vessel_class, [])
    destination.incoming_arcs.setdefault(vessel.vessel_class, [])
    origin.outgoing_arcs[vessel.vessel_class].append(arc)
    destination.incoming_arcs[vessel.vessel_class].append(arc)
    
    arc_dict.setdefault(vessel.vessel_class, {})
    arc_dict[vessel.vessel_class][(origin.tuple, destination.tuple)] = arc
    # arc_dict[(origin.tuple, destination.tuple, vessel)] = arc
    vessel_arcs.setdefault(vessel, []).append(arc)
    return arc_dict, vessel_arcs

def create_sink_arcs(vessels, sinkNode, arc_dict, vessel_arcs, sourceNode, NODE_DICT):

        # Arc from source node to sink node
        # arc_dict, vessel_arcs = add_arc_to_dict(sourceNode, sinkNode, vessel, arc_dict, vessel_arcs)
    for vessel in vessels:
        sinkNode.incoming_arcs.setdefault(vessel.vessel_class, [])
        
        # Arcs from other nodes to sink node
        for node in NODE_DICT.values():
            if node.port.number != sinkNode.port.number and node.time >= vessel.first_time_available:
                arc_dict, vessel_arcs = add_arc_to_dict(node, sinkNode, vessel, arc_dict, vessel_arcs)
        
    return arc_dict, vessel_arcs

def populate_vessel_arcs(vessels, vessel_arcs):
    for v in vessels:
        v.arcs = vessel_arcs[v]
        for arc in vessel_arcs[v]:
            v.all_arcs_v[arc.tuple] = arc
            
    return vessels
        
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
    
def build_model(vessels, arc_dict, regularNodes, ports, TIME_PERIOD_RANGE, sourceNode, sinkNode, waiting_arcs, vessel_classes, origin_node_arcs, destination_node_arcs, vessel_class_arcs, NODE_DICT, vessel_class_capacities):
    m = gp.Model('Maritime Inventory Routing Problem Speed Optimization')
    
    # for vc, arcs in vessel_class_arcs.items():
    #     for arc in arcs:
    #         print(arc)
 
    x = m.addVars(((arc.tuple, vc) for vc, arcs in vessel_class_arcs.items() for arc in arcs), vtype=gp.GRB.BINARY, name="x")

    s_bounds = {(node.port.number, node.time): node.port.capacity for node in regularNodes}
    s = m.addVars(s_bounds.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")

    s_bounds_source = {(port.number, 0): port.capacity for port in ports}
    s_source = m.addVars(s_bounds_source.keys(), lb=0, ub=s_bounds, vtype=gp.GRB.CONTINUOUS, name="s")
    s.update(s_source)
    
    alpha_bounds = {(node.port.number, node.time): ((len(TIME_PERIOD_RANGE)*node.port.rate) - (node.port.capacity-node.port.inventory)) if node.port.isLoadingPort else (len(TIME_PERIOD_RANGE)*node.port.rate - node.port.initial_inventory) for node in regularNodes}
    alpha = m.addVars(alpha_bounds.keys(), lb=0, ub=alpha_bounds, vtype=gp.GRB.CONTINUOUS, name="alpha")
    
    # Create a dict where the arc.tuple is the key and arc.cost is the value
    costs = {(arc.tuple, vc): arc.cost for vc in vessel_classes for arc in vessel_class_arcs[vc]}
    m.update()

    #Parameters:
    P = {(node.port.number, node.time): node.port.spotMarketPricePerUnit * (node.port.spotMarketDiscountFactor ** (node.time)) for node in regularNodes}
    original_obj = gp.quicksum(costs[key]*x[key] for key in costs) + gp.quicksum(alpha[node.port.number, node.time] * P[(node.port.number, node.time)] for node in regularNodes)

    m.setObjective(original_obj, GRB.MINIMIZE)
    m.update()

    # Constraint (2)
    '''Must leave the source node'''
    for vessel in vessels:
        # Find the destination node for the arc
        destination_node = NODE_DICT[(vessel.initial_port.number, vessel.first_time_available)]
        key = (sourceNode, destination_node)
        source_arc = vessel.all_arcs_v[key]
        # All arcs from the source node must be used. Every vessel has only one arc from the source node
        m.addConstr(x[(source_arc.tuple), vessel.vessel_class] == 1, name = 'SourceFlow')
        print(x[(source_arc.tuple), vessel.vessel_class] == 1)
    m.update()

    # Constraint (3)
    '''Must enter the sink node'''
    for vc in vessel_classes:
        v_in_class = sum(1 for vessel in vessels if vessel.vessel_class == vc)
        incoming_to_sink = [arc for arc in vessel_class_arcs[vc] if arc.destination_node == sinkNode]
        m.addConstr(gp.quicksum((x[arc.tuple, vc]) for arc in incoming_to_sink) == v_in_class, name = 'SinkFlow')
    m.update()

    # Constraint (4)
    '''For each node we enter, we must leave'''
    for vc in vessel_classes:
        for node in regularNodes:
            # Check if node is one of the keys in destination_node_arcs
            incoming_to_node = destination_node_arcs[vc].get(node.tuple, [])
            outgoing_from_node = origin_node_arcs[vc].get(node.tuple, [])
            if not incoming_to_node or not outgoing_from_node:
                # continue
                if incoming_to_node:
                    # Set all the x vars to 0
                    m.addConstr(gp.quicksum((x[in_arc.tuple, vc]) for in_arc in incoming_to_node) == 0, name = "ImpossibleFlow")
                elif outgoing_from_node:
                    m.addConstr(gp.quicksum((x[out_arc.tuple, vc]) for out_arc in outgoing_from_node) == 0, name = "ImpossibleFlow")
            else:
                m.addConstr(gp.quicksum((x[in_arc.tuple, vc]) for in_arc in incoming_to_node) - gp.quicksum((x[out_arc.tuple, vc]) for out_arc in outgoing_from_node) == 0, name = "FlowBalance")
    m.update()

    # Constraint (5)
    '''Set correct initial inventory at each port'''
    for port in ports:
        m.addConstr(s_source[port.number, 0] == port.inventory, name = 'InitialInventoryPort')
    m.update()


    # interegional_arcs = []
    # for vc in vessel_classes:
    #     for arc in vessel_class_arcs[vc]:
    #             if not arc.is_waiting_arc and not arc.is_source_or_sink_arc:
    #                 interegional_arcs.append(arc)
    # for port in ports:
    #     for t in TIME_PERIOD_RANGE:
    #         # Check if 't' is the first period to handle the 't-1' index.
    #         if t == 0:
    #             continue  # Skip the constraint for the first time period, or handle it separately.
                
    #         # Find interregional arcs with origin at port 'port' and time 't'.
    #         arcs_with_origin_t = [arc for arc in interegional_arcs if arc.origin_node.time == t and arc.origin_node.port == port]
            
    #         # Calculate the total capacity of vessels leaving port 'port' at time 't'.
    #         total_vessel_capacity = gp.quicksum(x[arc.tuple, vc] * vessel_class_capacities[vc] for arc in arcs_with_origin_t for vc in vessel_classes)
            
    #         # Inventory balance constraint.
    #         m.addConstr(s[port.number, t] == (s[port.number, t-1] + port.rate - total_vessel_capacity), name = f'PortBalance_Inter_regional_port:{port.number}_time:{t}_vc:{vc}')
    # m.update()


    # Constraint (6)
    '''Inventory balance at ports'''
    # Inventory balance for ports at the end of each time period t
    interegional_arcs = []
    for vc in vessel_classes:
        for arc in vessel_class_arcs[vc]:
                if not arc.is_waiting_arc and not arc.is_source_or_sink_arc:
                    interegional_arcs.append(arc)
                    
    # Check if we have an arc where the origin node is (1,315)
    check_arcs = [arc for arc in interegional_arcs if arc.origin_node.port.number == 3 and arc.origin_node.time == 347]
                    
    for port in ports:
        for t in TIME_PERIOD_RANGE:
            # interegional_arcs with origin_node.time = t and origin_node.port = port
            arcs_with_origin_t = [arc for arc in interegional_arcs if arc.origin_node.time == t and arc.origin_node.port == port]
            if arcs_with_origin_t:
                m.addConstr(s[port.number, t] == (s[port.number, t-1] + (port.isLoadingPort * port.rate) - (port.isLoadingPort * gp.quicksum((x[arc.tuple, arc.vessel_class] * vessel_class_capacities[arc.vessel_class] for arc in arcs_with_origin_t)) - port.isLoadingPort*alpha[port.number, t])) , name = f'PortBalance_Inter_regional_port:{port.number}_time:{t}_vc:{vc}')
            else:
                m.addConstr(s[port.number, t] == s[port.number, t-1] + (port.isLoadingPort * port.rate) - port.isLoadingPort*alpha[port.number, t], name = f'PortBalance_Inter_regional_port:{port.number}_time:{t}_vc:{vc}')
    m.update()

    # Constraint (7)
    '''Berth limit'''
    for node in regularNodes:
            # interegional_arcs with origin_node.time = t and origin_node.port = port
            interegional_arcs = []
            for vc in vessel_classes:
                for arc in vessel_class_arcs[vc]:
                    if not arc.is_waiting_arc and not arc.is_source_or_sink_arc:
                        if arc.origin_node == node:
                            interegional_arcs.append(arc)
            # interegional_arcs = [arc for arc in vessel_class_arcs[vc] for vc in vessel_classes if arc.origin_node == node]
            m.addConstr((gp.quicksum((x[arc.tuple, vc]) for arc in interegional_arcs) <= node.port.berth_limit), name = 'Birth_limit_in_time_t')
    m.update()
    
    return m, costs
    
def build_problem(INSTANCE):    
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
    vessel_data = get_vessels_data(VESSELINFO_PATH)
    regions_info, ports_info = extract_region_and_port_info(content)
    all_ports, loading_regions, discharging_regions, port_dict = create_ports_from_info_with_loading(ports_info, spotMarketPricePerUnit, spotMarketDiscountFactor)
    ports = get_ports(all_ports)
    NUM_TIME_PERIODS = metadata['numPeriods']
    TIME_PERIOD_RANGE = list(range(1, NUM_TIME_PERIODS+1))
    # MAX_SPEED = 15
    # MIN_SPEED = 8
    # OPERATING_SPEED = 14
    # OPERATING_COST = 200
    # WAITING_COST = 50
    # FUEL_PRICE = 500
    ORIGINAL_NUM_PORTS = len(ports)
    # EBS = 0.01
    
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
    FULL_DISTANCE_MATRIX = full_distance_matrix
    full_distance_matrix_nm = convert_matrix_to_nautical_miles(FULL_DISTANCE_MATRIX)
    full_travel_cost_matrices = parse_full_travel_cost_matrices(INSTANCE_PATH)
    full_travel_time_matrices = parse_full_travel_time_matrices(INSTANCE_PATH)
    
    source_arcs_info = generate_source_arc_info(sourceNode, vessels)
    all_info = populate_all_info(source_arcs_info, vessels, ports, full_distance_matrix_nm, NODE_DICT, NUM_TIME_PERIODS, full_travel_cost_matrices, full_travel_time_matrices, port_dict)
    arc_dict, vessel_arcs, waiting_arcs, sink_and_source_arcs = add_arcs_to_nodes(all_info, NODE_DICT, NUM_TIME_PERIODS)
    arc_dict, vessel_arcs = create_sink_arcs(vessels=vessels, sinkNode=sinkNode, arc_dict=arc_dict, vessel_arcs=vessel_arcs, sourceNode=sourceNode, NODE_DICT=NODE_DICT)
    populate_vessel_arcs(vessels, vessel_arcs)
 
    problem_data = {
        'vessels': vessels,
        'vessel_arcs': vessel_arcs,
        'arc_dict': arc_dict,
        'regularNodes': regularNodes,
        'ports': ports,
        'TIME_PERIOD_RANGE': TIME_PERIOD_RANGE,
        # 'non_operational': non_operational,
        'sourceNode': sourceNode,
        'sinkNode': sinkNode,
        'waiting_arcs': waiting_arcs,
        # 'OPERATING_COST': OPERATING_COST,
        # 'OPERATING_SPEED': OPERATING_SPEED,
        'NODES' : NODES,
        'NODE_DICT' : NODE_DICT,
        'VESSEL_CLASSES': VESSEL_CLASSES,
        'vessel_class_capacities': vessel_class_capacities
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

def main():
    # INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V9¨a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V11a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12a'
    # INSTANCE = 'LR1_2_DR1_3_VC2_V6a'
    # INSTANCE = 'LR1_2_DR1_3_VC3_V8a'
    # INSTANCE = 'LR1_1_DR1_4_VC3_V12b'
    # INSTANCE = 'LR2_11_DR2_22_VC3_V6a'
    # INSTANCE = 'LR2_11_DR2_33_VC4_V11a'
    # GROUP 2
    # INSTANCE = 'LR1_DR02_VC01_V6a'
    INSTANCE = 'LR1_DR02_VC02_V6a'
    

    problem_data = build_problem(INSTANCE)
    
    # Unpack the problem data
    vessels = problem_data['vessels']
    vessel_arcs = problem_data['vessel_arcs']
    arc_dict = problem_data['arc_dict']
    regularNodes = problem_data['regularNodes']
    ports = problem_data['ports']
    TIME_PERIOD_RANGE = problem_data['TIME_PERIOD_RANGE']
    sourceNode = problem_data['sourceNode']
    sinkNode = problem_data['sinkNode']
    waiting_arcs = problem_data['waiting_arcs']
    VESSEL_CLASSES = problem_data['VESSEL_CLASSES']
    NODE_DICT = problem_data['NODE_DICT']
    vessel_class_capacities = problem_data['vessel_class_capacities']
    
    origin_node_arcs, destination_node_arcs, vessel_class_arcs = rearrange_arcs(arc_dict)
    
    model, costs = build_model(vessels = vessels, arc_dict = arc_dict, regularNodes=regularNodes, ports=ports, TIME_PERIOD_RANGE = TIME_PERIOD_RANGE, sourceNode = sourceNode, sinkNode = sinkNode, waiting_arcs = waiting_arcs, vessel_classes = VESSEL_CLASSES, origin_node_arcs = origin_node_arcs, destination_node_arcs=destination_node_arcs, vessel_class_arcs=vessel_class_arcs, NODE_DICT = NODE_DICT, vessel_class_capacities=vessel_class_capacities)

    x_initial_solution, model = find_initial_solution(model)
    # Print the size of the initial solution
    print("Initial solution size:", len(x_initial_solution))
    
    # Remove the solution limit
    model.setParam(gp.GRB.Param.SolutionLimit, 2000000)
    
    ps_data = {'model': model, 'initial_solution':x_initial_solution, 'costs': costs, 'regularNodes': regularNodes, 'vessels': vessels, 'operating_cost':OPERATING_COST, 'vessel_arcs': vessel_arcs}

    # Perform the proximity search using the initial solution
    improved_solution, obj_value = perform_proximity_search(ps_data)
    #Perform proximity search with simulated annealing
    #improved_solution, obj_value = perform_proximity_search_with_simulated_annealing(ps_data)
    # print("Final solution:", improved_solution)
    print("Objective value:", obj_value)
    
    
    
if __name__ == "__main__":
    main()