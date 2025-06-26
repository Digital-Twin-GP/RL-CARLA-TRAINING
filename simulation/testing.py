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
import json
import socket

collision_intensity = -1
collision_flag = False
collision_flag_counter = 0

   
def collect_step_data(world, args):
    player = world.player
    control = player.get_control()
    location = player.get_location()
    velocity = player.get_velocity()
    acceleration = player.get_acceleration()
    speed_limit = getattr(world, 'speed_limit_detector', None)
    img_b64 = getattr(world.camera_manager, 'last_image_b64', None)  # Should be updated by camera callback

    vehicles = world.world.get_actors().filter('vehicle.*')
    nearby = [
        (actor.type_id, location.distance(actor.get_location()))
        for actor in vehicles
        if location.distance(actor.get_location()) < 200 and actor.id != player.id
    ]
    nearby_sorted = sorted(nearby, key=lambda x: x[1])

    weather = world.world.get_weather()

    data = {
        "vehicle": {
            "car_type": player.type_id,
            "car_display_name": player.attributes.get('role_name', ''),
            "speed": (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5,
            "acceleration": (acceleration.x**2 + acceleration.y**2 + acceleration.z**2) ** 0.5,
            "location": {"x": location.x, "y": location.y, "z": location.z},
            "heading": player.get_transform().rotation.yaw,
            "throttle": control.throttle,
            "brake": control.brake,
            "steer": control.steer
        },
        "world": {
            "weather": {
                    "cloudiness": weather.cloudiness,
                    "precipitation": weather.precipitation,
                    "precipitation_deposits": weather.precipitation_deposits,
                    "wind_intensity": weather.wind_intensity,
                    "sun_azimuth_angle": weather.sun_azimuth_angle,
                    "sun_altitude_angle": weather.sun_altitude_angle,
                    "fog_density": weather.fog_density,
                    "fog_distance": weather.fog_distance,
                    "fog_falloff": weather.fog_falloff,
                    "wetness": weather.wetness,
                    "scattering_intensity": weather.scattering_intensity,
                    "mie_scattering_scale": weather.mie_scattering_scale,
                    "rayleigh_scattering_scale": weather.rayleigh_scattering_scale,
                    "dust_storm": weather.dust_storm
                },
            "map": world.map.name,
            "speed_limit": getattr(speed_limit, 'current_speed_limit', None),
            "num_vehicles": len(vehicles),
            "simulation_time": getattr(world.hud, 'simulation_time', None),
        },
        "metrics": {
            "fuel_consumed": getattr(world, 'cumulative_fuel', None),
            "distance_traveled": getattr(world, 'distance_traveled', None),
            "collisions": {
                "latest_collision": collision_intensity,
                "flag": collision_flag
             }},
        "camera_image": img_b64,
        "nearby_vehicles": nearby_sorted,
        "environment": "Carla",
        "mode": args.mode,
        "summary": False
    }
    return data

def run_scenario(args, use_rl_throttle=False, model_path=None, client=None):
    try:
        global collision_flag, collision_flag_counter, collision_intensity
        HOST = '127.0.0.1'
        PORT = 65432
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setblocking(False)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Socket server listening on {HOST}:{PORT}")

        client_conn = None

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

        if args.scenario == 1:
            # Fixed start and end points
            start_point = carla.Transform(carla.Location(x=-13.6, y=5.8, z=11), carla.Rotation(yaw=180))
            end_point = carla.Location(x=-350.015564, y=5.184233, z=2)
        elif args.scenario == 2:
            spawn_points = sim_world.get_map().get_spawn_points()
            start_point = spawn_points[120].location
            start_point = carla.Transform(start_point, carla.Rotation(yaw=180))
            end_point = spawn_points[155].location

        world.player.set_transform(start_point)
        agent.set_destination(end_point)
        # if args.num_cars > 0:
        #     spawn_npc_vehicles(client, world.world, 10, world.player)

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

            # if collision_flag:
            #     collision_flag_counter += 1
            #     if collision_flag_counter > 30:
            #         collision_flag = False
            #         collision_flag_counter = 0

            # collision_happened = False
            # if world.collision_sensor.get_collision_history():
            #     collision_happened = True

            # if collision_happened:
            #     for entry in world.collision_sensor.history:
            #         collision_intensity = entry[1]
            #     collision_flag = True
            #     world.restart(args)
            #     agent = BehaviorAgent(world.player, behavior=args.behavior)
            #     agent.follow_speed_limits(True)
            #     agent.set_destination(random.choice(spawn_points).location)
            #     world.hud.notification("Collision! Restarted episode.", seconds=4.0)
            #     continue

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
            control = agent.run_step(debug=False)
            control.manual_gear_shift = False

            if use_rl_throttle:
                state = np.array([kmh, acceleration], dtype=np.float32)
                if args.dqn:
                    action = np.argmax(dqn_agent.get_qs(state))
                    throttle = [0.2, 0.5, 1.0][action]
                else: 
                    action = sac_agent.get_action(state, deterministic=True)
                    throttle = float(action[0])
                    # alpha = 0.7
                    # smoothed_throttle = alpha * throttle + (1 - alpha) * world.previous_throttle 
                    # throttle = smoothed_throttle

                if control.brake > 0.0: # Assume if user is braking, his leg is only on the brake pedal
                    throttle = 0.0
                control.throttle = throttle

            world.player.apply_control(control)

            # Try to accept a client connection if not already connected
            if client_conn is None:
                try:
                    client_conn, addr = server_socket.accept()
                    client_conn.setblocking(False)
                    print(f"Connected by {addr}")
                except BlockingIOError:
                    pass  # No client yet, continue simulation

            # If client is connected, send data
            if client_conn is not None:
                data = collect_step_data(world, args)
                try:
                    client_conn.sendall((json.dumps(data) + "\n").encode())
                except (BlockingIOError, BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    print("Client disconnected.")
                    client_conn.close()
                    client_conn = None

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

        summary_data = {
                "world": {
                    "simulation_time": total_time,
                },
                "metrics": {
                    "distance_traveled": total_distance,
                    "fuel_consumed": total_fuel,
                },
                "environment": "Carla",
                "mode": args.mode,
                "summary": True
            }
        if client_conn is not None:
                try:
                    client_conn.sendall((json.dumps(summary_data) + "\n").encode())
                except (BlockingIOError, BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    print("Client disconnected.")
                    client_conn.close()
                    client_conn = None
        
        return total_fuel, total_time, total_distance
    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.destroy()
        if client_conn:
            client_conn.close()
        server_socket.close()
        pygame.quit()

def test_model(args, model_path):
    try:
        print("Running RL model scenario...")
        pygame.init()
        pygame.font.init()
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        if model_path:
            result = run_scenario(args, use_rl_throttle=True, model_path=model_path, client=client)
            if result is None:
                print("RL scenario cancelled by user.")
                os._exit(0)
            fuel_rl, time_rl, dist_rl = result
            print(f"RL Model:     Fuel consumed = {fuel_rl:.2f} mg, Time = {time_rl:.2f} s, Distance = {dist_rl:.2f} m, Fuel/Distance = {fuel_rl/dist_rl if dist_rl > 0 else float('inf'):.2f} mg/m")

        else:
            print("Running Behavirol Agent scenario...")
            result = run_scenario(args, use_rl_throttle=False, client=client)
            if result is None:
                print("Behavioral Agent scenario cancelled by user.")
                os._exit(0)
            fuel_basic, time_basic, dist_basic = result
            print(f"Behavioral Agent:  Fuel consumed = {fuel_basic:.2f} mg, Time = {time_basic:.2f} s, Distance = {dist_basic:.2f} m, Fuel/Distance = {fuel_basic/dist_basic if dist_basic > 0 else float('inf'):.2f} mg/m")
        os._exit(0)
    except KeyboardInterrupt:
        print('\nTest cancelled by user. Bye!')