"""
import numpy as np
import matplotlib.pyplot as plt

# Extended speed range
v = np.linspace(0, 25, 400)  # Speed in knots

# Fuel Cost per Kilometer calculation
fuel_cost_per_km = 0.09 * (v / 15)**3

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(v, fuel_cost_per_km, label='Fuel Cost per Kilometer', color='grey')
plt.fill_between(v, fuel_cost_per_km, where=(v >= 13) & (v <= 19), color='cornflowerblue', alpha=0.5, label='Feasible Speed Range')
plt.title('Fuel Cost')
plt.xlabel('Speed (v) in knots')
plt.ylabel('Fuel Cost per Kilometer')
plt.grid(True)
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Extended speed range
v_knots = np.linspace(0, 25, 400)  # Speed in knots

# Convert speed from knots to km/h
v_kmh = v_knots * 1.852

# Fuel Cost per Kilometer calculation
fuel_cost_per_km = 0.09 * (v_kmh / (15*1.852))**3

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(v_kmh, fuel_cost_per_km, label='Fuel Cost per Kilometer', color='grey')
plt.fill_between(v_kmh, fuel_cost_per_km, where=(v_kmh >= 13 * 1.852) & (v_kmh <= 19 * 1.852), color='cornflowerblue', alpha=0.5, label='Feasible Speed Range')
plt.title('Fuel Cost')
plt.xlabel('Speed (v) in km/h')
plt.ylabel('Fuel Cost per Kilometer')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Extended speed range for plotting
v_plot = np.linspace(0, 25, 400)  # Speed in knots

# Fuel Cost per Kilometer calculation for plotting
fuel_cost_per_km_plot = 0.09 * (v_plot / 15)**3

# Convert cost to per nautical mile for the plotting range
fuel_cost_per_nautical_mile_plot = fuel_cost_per_km_plot * 1.852  # 1 nautical mile is approx 1.852 km

# Distance in nautical miles for the plotting range
distance = 4560

# Calculate the total fuel cost for the plotting range
total_fuel_cost_plot = fuel_cost_per_nautical_mile_plot * distance

# Given specific speeds and their calculated total fuel costs
specific_speeds = np.array([19.0, 15.83, 13.57])  # Speeds in knots
total_costs = np.array([1544.67, 893.34, 562.75])  # Corresponding total costs

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(v_plot, total_fuel_cost_plot, label='Total Fuel Cost', color='grey')
plt.fill_between(v_plot, total_fuel_cost_plot, where=(v_plot >= 13) & (v_plot <= 19), color='blue', alpha=0.3, label='Feasible Range')

# Adding dashed lines to the plot for specific speeds
for speed, total_cost in zip(specific_speeds, total_costs):
    # Draw the horizontal and vertical lines
    plt.vlines(x=speed, ymin=0, ymax=total_cost, color='red', linestyle='--', linewidth=1.5)
    plt.hlines(y=total_cost, xmin=0, xmax=speed, color='red', linestyle='--', linewidth=1.5)
    # Annotate the total cost at the end of the horizontal line
    plt.annotate(f'{total_cost:,.2f}', (speed, total_cost), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Total Fuel Cost')
plt.xlabel('Speed (v) in knots')
plt.ylabel('Total Fuel Cost')
plt.grid(True)
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Extended speed range for plotting
v_plot = np.linspace(0, 25, 400)  # Speed in knots

# Fuel Cost per Kilometer calculation for plotting
fuel_cost_per_km_plot = 0.09 * (v_plot / (15*1.852))**3

# Convert cost to per nautical mile for the plotting range
fuel_cost_per_nautical_mile_plot = fuel_cost_per_km_plot * 1.852  # 1 nautical mile is approx 1.852 km

# Distance in nautical miles
distance = 4560

# Calculate the total fuel cost for the plotting range
total_fuel_cost_plot = fuel_cost_per_nautical_mile_plot * distance

# Specific speeds
specific_speeds = np.array([19.0, 15.83, 13.57])  # Speeds in knots

# Calculate the total fuel costs for the specific speeds
specific_fuel_cost_per_km = 0.09 * (specific_speeds / 15)**3
specific_fuel_cost_per_nautical_mile = specific_fuel_cost_per_km * 1.852
total_costs = specific_fuel_cost_per_nautical_mile * distance

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(v_plot, total_fuel_cost_plot, label='Total Fuel Cost', color='grey')
plt.fill_between(v_plot, total_fuel_cost_plot, where=(v_plot >= 13) & (v_plot <= 19), color='cornflowerblue', alpha=0.5, label='Feasible Speed Range')

# Adding dashed lines to the plot for specific speeds
for speed, total_cost in zip(specific_speeds, total_costs):
    # Draw the horizontal and vertical lines with a less distinct red color
    plt.vlines(x=speed, ymin=0, ymax=total_cost, color='salmon', linestyle='--', linewidth=1.5)
    plt.hlines(y=total_cost, xmin=0, xmax=speed, color='salmon', linestyle='--', linewidth=1.5)
    # Annotate the total cost at the end of the horizontal line
    plt.annotate(f'{total_cost:,.2f}', (speed, total_cost), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Total Fuel Cost')
plt.xlabel('Speed (v) in knots')
plt.ylabel('Total Fuel Cost for a Given Distance d')
plt.grid(True)
plt.legend()
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

# Extended speed range for plotting in knots and converting to km/h
v_plot_knots = np.linspace(0, 25, 400)  # Speed in knots
v_plot_kmh = v_plot_knots * 1.852  # Convert speed from knots to km/h

# Fuel Cost per Kilometer calculation for plotting, using km/h directly
fuel_cost_per_km_plot = 0.09 * (v_plot_kmh / (15*1.852))**3

# Distance in kilometers (conversion factor from nautical miles removed)
distance = 4560 * 1.852  # Assuming original distance was in nautical miles, now converted to kilometers

# Calculate the total fuel cost for the plotting range
total_fuel_cost_plot = fuel_cost_per_km_plot * distance

# Specific speeds in knots converted to km/h for plotting
specific_speeds_knots = np.array([19.0, 15.83, 13.57])  # Speeds in knots
specific_speeds_kmh = specific_speeds_knots * 1.852  # Convert speeds to km/h

# Calculate the total fuel costs for the specific speeds using km/h
specific_fuel_cost_per_km = 0.09 * (specific_speeds_kmh / (15*1.852))**3
total_costs = specific_fuel_cost_per_km * distance

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(v_plot_kmh, total_fuel_cost_plot, label='Total Fuel Cost', color='grey')
plt.fill_between(v_plot_kmh, total_fuel_cost_plot, where=(v_plot_kmh >= 13 * 1.852) & (v_plot_kmh <= 19 * 1.852), color='cornflowerblue', alpha=0.5, label='Feasible Speed Range')

# Adding dashed lines to the plot for specific speeds
for speed_kmh, total_cost in zip(specific_speeds_kmh, total_costs):
    plt.vlines(x=speed_kmh, ymin=0, ymax=total_cost, color='salmon', linestyle='--', linewidth=1.5)
    plt.hlines(y=total_cost, xmin=0, xmax=speed_kmh, color='salmon', linestyle='--', linewidth=1.5)
    plt.annotate(f'{total_cost:,.2f}', (speed_kmh, total_cost), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Total Fuel Cost')
plt.xlabel('Speed (v) in km/h')
plt.ylabel('Total Fuel Cost for a Given Distance d')
plt.grid(True)
plt.legend()
plt.show()
