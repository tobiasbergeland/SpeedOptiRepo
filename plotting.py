import numpy as np
import matplotlib.pyplot as plt

# Constants
Theta = 0.090  # Example travel cost per kilometer
V0 = 15      # Base speed in knots
Delta = 0.0  # Discount factor (NO DISCOUNT)
v_min = 10   # Minimum speed in knots
v_max = 19   # Maximum speed in knots

# Speed range (in knots)
speeds = np.linspace(0, 22, 300)  # Generates 200 points between 5 and 18 knots for a smoother curve

# Fuel Cost per Kilometer Calculation
fuel_costs_per_km = Theta * ((speeds / V0)**3) * (1 - Delta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(speeds, fuel_costs_per_km, label='Fuel Cost per km for vessel class', color='black')
plt.fill_between(speeds, fuel_costs_per_km, where=(speeds >= v_min) & (speeds <= v_max), color='gray', alpha=0.3, label='Speed Interval for vessel class')
plt.title('Speed Impact on Fuel Cost per Kilometer')
plt.xlabel('Speed (knots)')
plt.ylabel('Fuel Cost per Kilometer ($)')
plt.grid(True)
plt.legend()
plt.show()

# Save the plot
plt.savefig('speed_impact.png')
