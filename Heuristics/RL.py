### Environment

INSTANCE = 'LR1_1_DR1_3_VC1_V7a'
INSTANCE_PATH = INSTANCE+'/'+INSTANCE+'.txt'
VESSELINFO_PATH = INSTANCE+'/vessel_data.txt'

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
        # Only have 1 decimal for cost and speed
        return f'{self.origin_node} -> {self.destination_node} --- Cost: {self.cost:.1f} --- Speed: {self.speed:.1f}'

class Vessel:
    def __init__(self, max_inventory, initial_inventory, max_operating_quantity, number):
        self.max_inventory = int(max_inventory)
        self.inventory = initial_inventory
        self.max_operating_quantity = max_operating_quantity
        self.number = number
        self.arcs = set()
        self.all_arcs_v = set()
        
    def __repr__(self):
        return f'Vessel {self.number}'

    def __repr2__(self):
        return f'Vessel {self.number}: Max Inventory = {self.max_inventory}, Inventory = {self.inventory}, Max Operating Quantity = {self.max_operating_quantity}'
    
    
def create_environment():
    
    #Helper function to convert string to int or float
    def safe_convert(value, data_type):
        try:
            if data_type == 'int':
                return int(value)
            elif data_type == 'float':
                return float(value)
            elif data_type == 'list':
                # Handle different list formats
                if value.startswith('[') and value.endswith(']'):
                    # Remove brackets, split by comma and strip spaces
                    return [int(x.strip()) for x in value[1:-1].split(',')]
                else:
                    # Split by space or other delimiters if necessary
                    return [int(x.strip()) for x in value.split()]
        except (ValueError, TypeError):
            return None
    
    try:
        # Read the content of the provided file
        with open(INSTANCE_PATH, 'r') as file:
            content = file.read()
            metadata = {}
            start_index = content.index("----- MetaData -----") + len("----- MetaData -----")
            end_index = content.find("\n\n", start_index) if "\n\n" in content[start_index:] else len(content)
            metadata_section = content[start_index:end_index].strip().split("\n")
            
            for line in metadata_section:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            
            numPeriods = safe_convert(metadata.get('numPeriods', '').split()[-1], 'int')
            numCommodities = safe_convert(metadata.get('numCommodities'), 'int')
            numLoadingRegions = safe_convert(metadata.get('numLoadingRegions'), 'int')
            numDischargingRegions = safe_convert(metadata.get('numDischargingRegions'), 'int')
            numLoadingPortsInRegion = safe_convert(metadata.get('numLoadingPortsInRegion', '[]'), 'list')
            numDischargingPortsInRegion = safe_convert(metadata.get('numDischargingPortsInRegion', '[]'), 'list')
            numVesselClasses = safe_convert(metadata.get('numVesselClasses'), 'int')
            numTermVesselsInClass = safe_convert(metadata.get('numTermVesselsInClass', '[]'), 'list')
            hoursPerPeriod = safe_convert(metadata.get('hoursPerPeriod'), 'int')
            spotMarketPricePerUnit = safe_convert(metadata.get('spotMarketPricePerUnit'), 'float')
            spotMarketDiscountFactor = safe_convert(metadata.get('spotMarketDiscountFactor'), 'float')
            perPeriodRewardForFinishingEarly = safe_convert(metadata.get('perPeriodRewardForFinishingEarly', '0'), 'float')
            attemptCost = safe_convert(metadata.get('attemptCost', '0'), 'float')
            constantForSinglePeriodAlphaSlack = safe_convert(metadata.get('constantForSinglePeriodAlphaSlack', '0'), 'float')
            constantForCumulativeAlphaSlack = safe_convert(metadata.get('constantForCumulativeAlphaSlack', '0'), 'float')
            
            print("Here")
            
        return {
            'numPeriods': numPeriods,
            'numCommodities': numCommodities,
            'numLoadingRegions': numLoadingRegions,
            'numDischargingRegions': numDischargingRegions,
            'numLoadingPortsInRegion': numLoadingPortsInRegion,
            'numDischargingPortsInRegion': numDischargingPortsInRegion,
            'numVesselClasses': numVesselClasses,
            'numTermVesselsInClass': numTermVesselsInClass,
            'hoursPerPeriod': hoursPerPeriod,
            'spotMarketPricePerUnit': spotMarketPricePerUnit,
            'spotMarketDiscountFactor': spotMarketDiscountFactor,
            'perPeriodRewardForFinishingEarly': perPeriodRewardForFinishingEarly,
            'attemptCost': attemptCost,
            'constantForSinglePeriodAlphaSlack': constantForSinglePeriodAlphaSlack,
            'constantForCumulativeAlphaSlack': constantForCumulativeAlphaSlack
        }
    except Exception as e:
        print(f'Error reading file: {e}')
        return None

environment = create_environment()
print(environment)
