import os
import time
import numpy as np
import pygame
import carla
from carla_utils.hud import HUD, KeyboardControl
from carla_utils.world import World
from agents.behavior_agent import BehaviorAgent
from carla_utils.utils import spawn_npc_vehicles
from energy_utils import calculate_power, calculate_fuel_rate
from rl.dqn_sac_agents import DQNAgent, SACAgent
from tensorflow.keras.models import load_model
import math

def run_scenario(args, use_rl_throttle=False, model_path=None, client=None):
    try:
        # Setup CARLA and pygame
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        sim_world = client.load_world('Town04_Opt')
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        agent = BehaviorAgent(world.player, behavior=args.behavior)
        agent.follow_speed_limits(True)

        # Fixed start and end points
        start_point = carla.Transform(carla.Location(x=-13.6, y=5.8, z=11), carla.Rotation(yaw=180))
        end_point = carla.Location(x=-350.015564, y=5.184233, z=2)
        world.player.set_transform(start_point)
        agent.set_destination(end_point)
        if args.num_cars > 0:
            spawn_npc_vehicles(client, world.world, args.num_cars, world.player)

        # Load RL model if needed
        if use_rl_throttle:
            if args.dqn:
                dqn_agent = DQNAgent()
                dqn_agent.model = load_model(model_path)
            else:
                sac_agent = SACAgent()
                sac_agent.actor = load_model(model_path, compile=False)


        clock = pygame.time.Clock()
        total_fuel = 0
        start_time = time.time()
        done = False

        while not done:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                break

            current_speed_limit = world.speed_limit_detector.update()
            world.tick(clock, current_speed_limit)
            world.render(display)
            pygame.display.flip()

            # Get state for RL model
            v = world.player.get_velocity()
            kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            last_velocity = getattr(world, "last_velocity", 0)
            last_update_time = getattr(world, "last_update_time", time.time())
            delta_time = time.time() - last_update_time
            acceleration = (kmh - last_velocity) / 3.6 / delta_time if delta_time > 0 else 0
            world.last_update_time = time.time()
            world.last_velocity = kmh

            # Get navigation control
            control = agent.run_step(debug=True)
            control.manual_gear_shift = False

            if use_rl_throttle:
                state = np.array([kmh, acceleration], dtype=np.float32)
                if args.dqn:
                    action = np.argmax(dqn_agent.get_qs(state))
                    throttle = [0.2, 0.5, 1.0][action]
                else: 
                    action = sac_agent.get_action(state, deterministic=True)
                    throttle = float(action[0])
                if control.brake > 0.0: # Assume if user is braking, his leg is only on the brake pedal
                    throttle = 0.0
                control.throttle = throttle

            world.player.apply_control(control)

            # Fuel calculation
            power = calculate_power(kmh, acceleration)
            fuel = calculate_fuel_rate(power)
            total_fuel += fuel

            # Check if reached destination or crashed
            if agent.done():
                done = True
            collision = world.collision_sensor.get_collision_history()
            if any([x > 0 for x in collision.values()]):
                done = True

        total_time = time.time() - start_time
        total_distance = world.distance_traveled  # in meters
        world.destroy()
        
        return total_fuel, total_time, total_distance
    
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        try:
            if world is not None:
                world.destroy()
            pygame.quit()
        except Exception:
            pass

def test_model(args, model_path):
    try:
        print("Running RL model scenario...")
        pygame.init()
        pygame.font.init()
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        result = run_scenario(args, use_rl_throttle=True, model_path=model_path, client=client)
        if result is None:
            print("RL scenario cancelled by user.")
            os._exit(0)
        fuel_rl, time_rl, dist_rl = result

        print("Running Behavirol Agent scenario...")
        result = run_scenario(args, use_rl_throttle=False, client=client)
        if result is None:
            print("Behavioral Agent scenario cancelled by user.")
            os._exit(0)
        fuel_basic, time_basic, dist_basic = result
        pygame.quit()
        print("\n=== Comparison ===")
        print(f"RL Model:     Fuel consumed = {fuel_rl:.2f} mg, Time = {time_rl:.2f} s, Distance = {dist_rl:.2f} m, Fuel/Distance = {fuel_rl/dist_rl if dist_rl > 0 else float('inf'):.2f} mg/m")
        print(f"Behavioral Agent:  Fuel consumed = {fuel_basic:.2f} mg, Time = {time_basic:.2f} s, Distance = {dist_basic:.2f} m, Fuel/Distance = {fuel_basic/dist_basic if dist_basic > 0 else float('inf'):.2f} mg/m")
        os._exit(0)
    except KeyboardInterrupt:
        print('\nTest cancelled by user. Bye!')