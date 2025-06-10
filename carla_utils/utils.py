import random
import re
import carla

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def spawn_npc_vehicles(client, world, num_cars, ego_vehicle):
    """
    Spawns num_cars NPC vehicles at random spawn points, avoiding the ego vehicle's spawn.
    Returns a list of spawned NPC actors.
    """
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.tesla.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Avoid ego vehicle's spawn point
    ego_location = ego_vehicle.get_location()
    spawn_points = [sp for sp in spawn_points if sp.location.distance(ego_location) > 2.0]

    for n, spawn_point in enumerate(spawn_points[:num_cars]):
        blueprint = random.choice(vehicle_blueprints)
        blueprint.set_attribute('role_name', f'autopilot{n}')
        npc = world.try_spawn_actor(blueprint, spawn_point)
        if npc is not None:
            npc.set_autopilot(True, client.get_trafficmanager().get_port())