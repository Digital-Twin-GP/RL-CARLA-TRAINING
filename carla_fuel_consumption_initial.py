import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf # 2.11.0
from tensorflow.keras import backend as K
from threading import Thread
from tqdm import tqdm
import carla

# Constants for vehicle and fuel consumption model
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

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        super().set_model(model)  # Call the parent class's set_model method

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        
        # Initialize velocity and fuel tracking
        self.last_velocity = 0
        self.current_velocity = 0
        self.acceleration = 0
        self.cumulative_fuel = 0
        self.last_update_time = 0

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        
        # Reset fuel tracking
        self.last_velocity = 0
        self.current_velocity = 0
        self.acceleration = 0
        self.cumulative_fuel = 0
        self.last_update_time = time.time()

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        # Update action space to include different throttle levels
        # Actions 0-2: Full throttle (1.0) with different steering
        # Actions 3-5: Medium throttle (0.5) with different steering
        # Actions 6-8: Low throttle (0.2) with different steering
        throttle = 1.0
        if action >= 6:
            throttle = 0.2
            action -= 6
        elif action >= 3:
            throttle = 0.5
            action -= 3
            
        # Now action is 0 (left), 1 (straight), or 2 (right)
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=1*self.STEER_AMT))

        # Calculate velocity and acceleration
        v = self.vehicle.get_velocity()
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.last_velocity = self.current_velocity
        self.current_velocity = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
        
        # Calculate acceleration (m/s²)
        if delta_time > 0:
            self.acceleration = (self.current_velocity - self.last_velocity) / 3.6 / delta_time
        else:
            self.acceleration = 0
            
        # Calculate power and fuel consumption
        power = calculate_power(self.current_velocity, self.acceleration)
        fuel_consumed = calculate_fuel_rate(power)
        self.cumulative_fuel += fuel_consumed
        
        # Calculate reward based on speed, safety, and fuel efficiency
        kmh = int(self.current_velocity)
        
        # Base reward components
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            done = False
            
            # Speed reward component
            if kmh < 20:
                speed_reward = -2  # Too slow
            elif kmh < 40:
                speed_reward = 0   # Acceptable speed
            elif kmh < 60: 
                speed_reward = 1   # Good speed
            else:
                speed_reward = -1  # Too fast, less efficient
                
            # Fuel efficiency reward component (penalize high fuel consumption)
            # Normalize fuel consumption - lower is better
            fuel_reward = -0.01 * fuel_consumed
            
            # Smooth driving reward (penalize high acceleration)
            smooth_reward = -0.5 * abs(self.acceleration) if abs(self.acceleration) > 1 else 0
            
            # Combine rewards
            reward = speed_reward + fuel_reward + smooth_reward
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, {"fuel": fuel_consumed, "speed": kmh}


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Change from 3 to 9 actions
        predictions = Dense(9, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 9)).astype(np.float32)  # Changed from (1, 3) to (1, 9)
        self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_FRACTION * 1024)]
            )
        except RuntimeError as e:
            print(e)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Add tracking for fuel consumption
    ep_fuel_consumptions = []
    
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        total_fuel_consumed = 0  # Track fuel per episode

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action from expanded action space (0-8)
                action = np.random.randint(0, 9)
                # This takes no time, so we add a delay matching 60 FPS
                time.sleep(1/FPS)

            new_state, reward, done, info = env.step(action)
            total_fuel_consumed += info["fuel"]

            # Rest of the training loop remains unchanged
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            step += 1

            if done:
                break

        # Track fuel consumption for this episode
        ep_fuel_consumptions.append(total_fuel_consumed)

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Update stats to include fuel consumption
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_fuel = sum(ep_fuel_consumptions[-AGGREGATE_STATS_EVERY:])/len(ep_fuel_consumptions[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, 
                                          reward_max=max_reward, epsilon=epsilon,
                                          fuel_consumption_avg=average_fuel)
            
            # Rest of your code remains the same

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')