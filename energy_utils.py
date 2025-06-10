import math
from config import (
    VEHICLE_MASS, FRONTAL_AREA, DRAG_COEFFICIENT, ROLLING_COEFFICIENT,
    AIR_DENSITY, GRAVITY, TIME_STEP, ETA, B1, C, LHV, K0
)

def calculate_power(speed, acceleration, mass=VEHICLE_MASS, frontal_area=FRONTAL_AREA):
    """Calculate power required to move the vehicle"""
    speed_ms = speed / 3.6  # Convert km/h to m/s
    F_mass = acceleration * mass
    F_rolling = ROLLING_COEFFICIENT * mass * GRAVITY
    F_air = 0.5 * AIR_DENSITY * frontal_area * DRAG_COEFFICIENT * speed_ms**2
    F_total = F_mass + F_rolling + F_air
    power = (F_total * speed_ms) / 1000  # Convert to kW
    return max(0, power)  # Power can't be negative for fuel consumption

def calculate_fuel_rate(P, N=0.035, V=3.6):
    """Calculate fuel use rate in mg per timestep using the CMEM model"""
    N0 = 30 * math.sqrt(3.0 / V)
    K = K0 * (1 + C * (N - N0))
    FR = (K * N * V + (P / ETA)) * (1 / LHV) * (1 + B1 * (N - N0) ** 2)  # (g/s)
    fuel_rate_per_timestep = FR * 1000 * TIME_STEP  # mg/timestep
    return fuel_rate_per_timestep