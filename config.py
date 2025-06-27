SECONDS_PER_EPISODE = 15
REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
EPISODES = 50
MODEL_NAME = f"FuelOptimizer__{SECONDS_PER_EPISODE}s__{EPISODES}"
MEMORY_FRACTION = 0.4

DISCOUNT = 0.99
TAU = 0.005
ALPHA = 0.2
EPSILON = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

SPEED_LIMIT = 30
VEHICLE_MASS = 1500  # kg for typical car
FRONTAL_AREA = 2.2   # m^2
DRAG_COEFFICIENT = 0.3
ROLLING_COEFFICIENT = 0.01
AIR_DENSITY = 1.2    # kg/m^3
GRAVITY = 9.81       # m/s^2
TIME_STEP = 0.05     # seconds (estimate of simulation step)

# Constants for CMEM model  
ETA = 0.45      # Indicated efficiency  
B1 = 1e-4       # Coefficient  
C = 0.00125     # Coefficient  
LHV = 43.2      # Lower heating value of diesel fuel in kJ/g  
K0 = 1          # Default value