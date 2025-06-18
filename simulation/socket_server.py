import socket
import json
import threading
import pygame
import carla
import random
import time
from carla_utils.hud import HUD, KeyboardControl
from carla_utils.world import World
from agents.behavior_agent import BehaviorAgent
from carla_utils.utils import spawn_npc_vehicles

def collect_step_data(world, agent):
    player = world.player
    control = player.get_control()
    location = player.get_location()
    velocity = player.get_velocity()
    acceleration = player.get_acceleration()
    gnss = getattr(world, 'gnss_sensor', None)
    speed_limit = getattr(world, 'speed_limit_detector', None)
    collision_sensor = getattr(world, 'collision_sensor', None)

    data = {
        "vehicle": {
            "speed": (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5,
            "acceleration": (acceleration.x**2 + acceleration.y**2 + acceleration.z**2) ** 0.5,
            "location": {"x": location.x, "y": location.y, "z": location.z},
            "heading": player.get_transform().rotation.yaw,
            "gnss": {
                "lat": getattr(gnss, 'lat', None),
                "lon": getattr(gnss, 'lon', None)
            },
            "throttle": control.throttle,
            "brake": control.brake,
            "steer": control.steer,
        },
        "world": {
            "weather": str(world.world.get_weather()),
            "map": world.map.name,
            "speed_limit": getattr(speed_limit, 'current_speed_limit', None),
            "num_vehicles": len(world.world.get_actors().filter('vehicle.*')),
            "simulation_time": getattr(world.hud, 'simulation_time', None),
        },
        "metrics": {
            "fuel_consumed": getattr(world, 'cumulative_fuel', None),
            "distance_traveled": getattr(world, 'distance_traveled', None),
            "collisions": collision_sensor.get_collision_history() if collision_sensor else [],
        }
    }
    return data

def socket_server(args):
    pygame.init()
    pygame.font.init()
    world = None

    HOST = '127.0.0.1'
    PORT = 65432
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(False)  # Non-blocking mode
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

                collision_happened = False
                if world.collision_sensor.get_collision_history():
                    collision_happened = True

                if collision_happened:
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