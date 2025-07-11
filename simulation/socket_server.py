import socket
import json
import pygame
import ast
import carla
import random
import time
from carla_utils.hud import HUD, KeyboardControl
from carla_utils.world import World
from agents.behavior_agent import BehaviorAgent
from carla_utils.utils import spawn_npc_vehicles

collision_intensity = -1
collision_flag = False
collision_flag_counter = 0

def decode_utf8(signal_list):
    try:
        # Remove trailing zeros and decode
        return bytes([b for b in signal_list if b != 0]).decode('utf-8')
    except Exception as e:
        return f"<decode error: {e}>"

def collect_step_data(world, agent):
    player = world.player
    control = player.get_control()
    location = player.get_location()
    velocity = player.get_velocity()
    acceleration = player.get_acceleration()
    speed_limit = getattr(world, 'speed_limit_detector', None)
    img_b64 = getattr(world.camera_manager, 'last_image_b64', None)  # Should be updated by camera callback

    ego_location = player.get_location()
    vehicles = world.world.get_actors().filter('vehicle.*')
    nearby = [
        (actor.type_id, ego_location.distance(actor.get_location()))
        for actor in vehicles
        if ego_location.distance(actor.get_location()) < 200 and actor.id != player.id
    ]
    nearby_sorted = sorted(nearby, key=lambda x: x[1])
    nearby_sorted_fixed = list(str(nearby_sorted).encode('utf-8')[:5000]) + [0] * (5000 - len(str(nearby_sorted).encode('utf-8')))
    decoded_nearby = decode_utf8(nearby_sorted_fixed).strip()

	# Treat “nothing” ('' or None) as an empty list
    if not decoded_nearby:
        nearby_list = []
    else:
        try:
            nearby_list = ast.literal_eval(decoded_nearby)
        except (SyntaxError, ValueError):
            nearby_list = []
	# Convert each element to a list (handles [], tuples, etc.)
    nearby_json = [list(item) for item in nearby_list]

    if img_b64 is not None:
            camera_image = bytes([b for b in img_b64 if b != 0]).decode('utf-8')
    else:
        camera_image = "" 

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
        "camera_image": camera_image,
        "nearby_vehicles": nearby_json,
        "environment": "Carla",
        "mode": 1,
        "summary": False
    }
    return data

def socket_server(args):
    global collision_flag, collision_flag_counter, collision_intensity
    pygame.init()
    pygame.font.init()
    world = None

    HOST = '127.0.0.1'
    PORT = 65432
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(False)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Socket server listening on {HOST}:{PORT}")

    client_conn = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        sim_world = client.load_world('Town04_Opt')

        # Clean up all actors before starting
        actors = sim_world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.') or actor.type_id.startswith('sensor.'):
                actor.destroy()
        time.sleep(1)  # Give CARLA a moment to clean up

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA RL Simulation")

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        agent = BehaviorAgent(world.player, behavior=args.behavior)
        agent.follow_speed_limits(True)  

        spawn_points = world.map.get_spawn_points()
        destination = carla.Location(x=-360.015564, y=5.184233, z=2)
        agent.set_destination(destination)
        
        if args.num_cars > 0:
            spawn_npc_vehicles(client, world.world, 50, world.player)

        clock = pygame.time.Clock()   

        running = True
        while running:
            try:
                clock.tick()
                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()
                if controller.parse_events():
                    running = False
                    break

                current_speed_limit = world.speed_limit_detector.update()
                world.tick(clock, current_speed_limit)
                world.render(display)
                pygame.display.flip()

                if collision_flag:
                    collision_flag_counter += 1
                    if collision_flag_counter > 30:
                        collision_flag = False
                        collision_flag_counter = 0

                collision_happened = False
                if world.collision_sensor.get_collision_history():
                    collision_happened = True

                if collision_happened:
                    for entry in world.collision_sensor.history:
                        collision_intensity = entry[1]
                    collision_flag = True
                    world.restart(args)
                    agent = BehaviorAgent(world.player, behavior=args.behavior)
                    agent.follow_speed_limits(True)
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Collision! Restarted episode.", seconds=4.0)
                    continue

                if agent.done():
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The episode finished, searching for another target")

                control = agent.run_step(debug=True)
                control.manual_gear_shift = False
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
                    data = collect_step_data(world, agent)
                    try:
                        client_conn.sendall((json.dumps(data) + "\n").encode())
                    except (BlockingIOError, BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        print("Client disconnected.")
                        client_conn.close()
                        client_conn = None

            except KeyboardInterrupt:
                print("KeyboardInterrupt received. Exiting simulation loop.")
                running = False

    finally:
        summary_data = {
                "world": {
                    "simulation_time": getattr(world.hud, 'simulation_time', None),
                },
                "metrics": {
                    "distance_traveled": getattr(world, 'distance_traveled', None),
                    "fuel_consumed": getattr(world, 'cumulative_fuel', None),
                },
                "environment": "Carla",
                "mode": 0,
                "summary": True
            }
        if client_conn is not None:
                try:
                    client_conn.sendall((json.dumps(summary_data) + "\n").encode())
                except (BlockingIOError, BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    print("Client disconnected.")
                    client_conn.close()
                    client_conn = None
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