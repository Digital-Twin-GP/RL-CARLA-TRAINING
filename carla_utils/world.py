import sys
import time
from carla_utils.utils import find_weather_presets, get_actor_display_name
from energy_utils import calculate_power, calculate_fuel_rate
from carla_utils.sensors import CollisionSensor, LaneInvasionSensor, GnssSensor, CameraManager, SpeedLimitDetector
from config import SPEED_LIMIT
import carla
import math
import random

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        ############Fuel consumption#################
        self.cumulative_fuel = 0
        self.distance_traveled = 0
        self.previous_location = None
        self.current_velocity = 0
        self.last_velocity = 0
        self.acceleration = 0
        self.last_update_time = time.time()
        # Add speed limit detector
        self.speed_limit_detector = SpeedLimitDetector(self, SPEED_LIMIT)

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Instead of getting a random blueprint, specifically get the Tesla
        blueprint = self.world.get_blueprint_library().find('vehicle.chevrolet.impala')
        
        # Set color if desired
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '0,0,0') # Black Tesla
        
        blueprint.set_attribute('role_name', 'carla_driver')

        # Spawn the player.
        if self.player is not None:
            # spawn_point = self.player.get_transform()
            # spawn_point.location.z += 2.0
            # spawn_point.rotation.roll = 0.0
            # spawn_point.rotation.pitch = 0.
            self.destroy_sensors()
            self.destroy()
            spawn_point = carla.Transform(carla.Location(x=-13.6, y=5.8, z=11), carla.Rotation(yaw=180))
            spawn_points = self.map.get_spawn_points()
            self.player = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = carla.Transform(carla.Location(x=-13.6, y=5.8, z=11), carla.Rotation(yaw=180))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, speed_limit):
        # Update speed limit based on detected signs
        current_speed_limit = self.speed_limit_detector.update()
        
        # Pass the updated speed limit to the HUD
        self.hud.tick(self, clock, current_speed_limit)
        # --- Fuel and distance calculation ---
        v = self.player.get_velocity()
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

        # Calculate distance traveled
        current_location = self.player.get_location()
        if self.previous_location:
            self.distance_traveled += self.previous_location.distance(current_location)
        self.previous_location = current_location

        # Print stats
        # print(f"Speed: {self.current_velocity:.2f} km/h | "
        #     f"Fuel: {self.cumulative_fuel:.2f} mg | "
        #     f"Distance: {self.distance_traveled:.2f} m")

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None
        if self.collision_sensor and self.collision_sensor.sensor:
            self.collision_sensor.sensor.destroy()
            self.collision_sensor.sensor = None
        if self.lane_invasion_sensor and self.lane_invasion_sensor.sensor:
            self.lane_invasion_sensor.sensor.destroy()
            self.lane_invasion_sensor.sensor = None
        if self.gnss_sensor and self.gnss_sensor.sensor:
            self.gnss_sensor.sensor.destroy()
            self.gnss_sensor.sensor = None

    def destroy(self):
        """Destroys all actors"""
        self.destroy_sensors()
        if self.player:
            self.player.destroy()
            self.player = None