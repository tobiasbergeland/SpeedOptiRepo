import re

def get_current_x_solution_vars(model):
    return {v.varName : v for v in model.getVars() if v.VarName.startswith('x')}

def get_current_alpha_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('a')}

def get_current_s_solution_vars(model):
    return {v.varName: v for v in model.getVars() if v.VarName.startswith('s')}

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
