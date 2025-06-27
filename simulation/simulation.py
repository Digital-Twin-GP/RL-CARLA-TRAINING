import random
import pygame
import carla
from carla_utils.hud import HUD, KeyboardControl
from carla_utils.world import World
from agents.behavior_agent import BehaviorAgent
from carla_utils.utils import spawn_npc_vehicles

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

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

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = carla.Location(x=-360.015564, y=5.184233, z=2) 
        agent.set_destination(destination)

        if args.num_cars > 0:
            spawn_npc_vehicles(client, world.world, args.num_cars, world.player)

        clock = pygame.time.Clock()

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return
            
            # Get the current speed limit from the detector
            current_speed_limit = world.speed_limit_detector.update()
            
            # Update world with current speed limit
            world.tick(clock, current_speed_limit)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("Target reached", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step(debug = True)
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()
        
        pygame.quit()