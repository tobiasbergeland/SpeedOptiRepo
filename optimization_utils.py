import re

def get_current_x_solution_vars(model):
    return {v.varName : v for v in model.getVars() if v.VarName.startswith('x')}

def get_current_x_solution_vals(model):
    return {v.varName : v.x for v in model.getVars() if v.VarName.startswith('x')}

def get_current_alpha_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('a')}

def get_current_alpha_solution_vals(model):
    return {v.varName: v.x for v in model.getVars() if v.VarName.startswith('a')}

def get_current_s_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('s')}

def get_current_s_solution_vals(model):
    return {v.varName: v.x for v in model.getVars() if v.VarName.startswith('s')}


def find_corresponding_arcs(current_solution_x, vessel_class_arcs):
    active_arcs = {vc: [] for vc in vessel_class_arcs.keys()}
    
    for vessel_class, arcs in vessel_class_arcs.items():
        for arc in arcs:
            # Use the name of the variable to find the corresponding arc
            # Create the same variable name as in the model
            varname = convert_key_to_varname((arc.tuple, arc.vessel_class))
            if 1 - 1e-3 <= current_solution_x[varname]:
                # append to the active arcs with the vessel class as key if it is not present already
                if arc not in active_arcs[vessel_class]:
                    active_arcs.setdefault(arc.vessel_class, []).append(arc)
                # active_arcs.setdefault(arc.vessel_class, []).append(arc)
    return active_arcs

def convert_key_to_varname(key):
    arc_tuple = key[0]
    origin_node = arc_tuple[0]
    destination_node = arc_tuple[1]
    vc = key[1]
    
    # Check if the arc is an interregional arc or not
    # If origin port is the same as the destination port, it's a waiting arc, thus not a interregional arc
    if origin_node.port.number == destination_node.port.number or origin_node.time == -1:
        varname = f"x_non_interregional_{arc_tuple}_{vc}"
    else:
        varname = f"x_interregional_{arc_tuple}_{vc}"
    return varname

def varname_to_key(varname):
    # Regex updated to extract nested tuples and a final integer vc
    match = re.search(r"x_(non_)?interregional_\(\((\d+), (\d+)\), \((\d+), (\d+)\)\)_(\d+)", varname)
    if not match:
        raise ValueError("Variable name does not match expected format")

    # Extract values from the regex match groups
    origin_node_id = (int(match.group(2)), int(match.group(3)))
    destination_node_id = (int(match.group(4)), int(match.group(5)))
    vc = int(match.group(6))
    
    # Reconstruct the arc tuple
    arc_tuple = (origin_node_id, destination_node_id)
    
    # Return the reconstructed key
    return (arc_tuple, vc)

def key_to_node_info(key):
    arc_tuple, vc = key
    origin_node_id, destination_node_id = arc_tuple
    origin_port, origin_time  = origin_node_id
    destination_port, destination_time = destination_node_id
    return (origin_port, origin_time, destination_port, destination_time, vc)


def extract_time_period_from_s_or_alpha_var_name(var_name):
    simple_match = re.search(r'\\[(\\d+),(\\d+)\\]', var_name)
    if simple_match:
        return int(simple_match.group(2))  # Returns the second number as an integer, representing the time period
    else:
        return None

def extract_time_period_from_x_var_name(var_name):
    complex_matches = re.findall(r'\\((-?\\d+),\\s*(-?\\d+)\\)', var_name)
    if complex_matches and len(complex_matches) == 2:
        dep_time = int(complex_matches[0][1])  # Extract departure time from the first tuple
        arr_time = int(complex_matches[1][1])  # Extract arrival time from the second tuple
        return (dep_time, arr_time)
    else:
        return None
