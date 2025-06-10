import os
import time
import random
import numpy as np
import pygame
import carla
from threading import Thread
from carla_utils.hud import HUD, KeyboardControl
from carla_utils.world import World
from agents.behavior_agent import BehaviorAgent
from carla_utils.utils import spawn_npc_vehicles
from energy_utils import calculate_power, calculate_fuel_rate
from rl.dqn_sac_agents import DQNAgent, SACAgent
from config import (
    EPSILON, MEMORY_FRACTION, EPISODES, MIN_EPSILON, EPSILON_DECAY, MODEL_NAME, SECONDS_PER_EPISODE
)
import tensorflow as tf
import math

def rl_training_loop(args):
    try:
        epsilon = EPSILON
        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_FRACTION * 1024)]
                )
            except RuntimeError as e:
                print(e)

        if not os.path.isdir('models'):
            os.makedirs('models')

        if args.dqn:
            agent_rl = DQNAgent()
        else:
            agent_rl = SACAgent()

        trainer_thread = Thread(target=agent_rl.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent_rl.training_initialized:
            time.sleep(0.01)

        # Dummy prediction to initialize model
        agent_rl.get_qs([0.0, 0.0])

        pygame.init()
        pygame.font.init()
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

        agent_nav = BehaviorAgent(world.player, behavior=args.behavior)
        agent_nav.follow_speed_limits(True)

        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent_nav.set_destination(destination)
        if args.num_cars > 0:
            spawn_npc_vehicles(client, world.world, args.num_cars, world.player)
        
        episode = 1
        while episode <= EPISODES:
            total_fuel_consumed = 0
            episode_reward = 0
            step = 1
            done = False
            collision_happened = False
            episode_transitions = []
            episode_distance_traveled = 0.0

            world.player.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            clock = pygame.time.Clock()

            # Get initial state: [speed, acceleration]
            v = world.player.get_velocity()
            kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            last_velocity = getattr(world, "last_velocity", 0)
            last_update_time = getattr(world, "last_update_time", time.time())
            delta_time = time.time() - last_update_time
            acceleration = (kmh - last_velocity) / 3.6 / delta_time if delta_time > 0 else 0
            world.last_update_time = time.time()
            world.last_velocity = kmh
            current_state = np.array([kmh, acceleration], dtype=np.float32)

            episode_start = time.time()

            while True:
                clock.tick()
                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()
                if controller.parse_events():
                    done = True
                    break

                # Update world and render
                current_speed_limit = world.speed_limit_detector.update()
                world.tick(clock, current_speed_limit)
                world.render(display)
                pygame.display.flip()

                # RL agent chooses throttle
                if args.dqn:
                    if np.random.random() > epsilon:
                        action = np.argmax(agent_rl.get_qs(current_state))
                    else:
                        action = np.random.randint(0, 3)
                        time.sleep(1/60)
                    throttle = [0.2, 0.5, 1.0][action]
                else:
                    action = agent_rl.get_action(current_state)
                    throttle = float(action[0])

                # Get navigation control and override throttle
                control = agent_nav.run_step(debug=True)
                if control.brake > 0.0: # Assume if user is braking, his leg is only on the brake pedal
                    throttle = 0.0
                control.throttle = throttle
                control.manual_gear_shift = False
                world.player.apply_control(control)

                # Get next state: [speed, acceleration]
                v = world.player.get_velocity()
                current_time = time.time()
                kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                last_velocity = getattr(world, "last_velocity", 0)
                last_update_time = getattr(world, "last_update_time", current_time)
                delta_time = current_time - last_update_time
                acceleration = (kmh - last_velocity) / 3.6 / delta_time if delta_time > 0 else 0
                world.last_update_time = current_time
                world.last_velocity = kmh
                new_state = np.array([kmh, acceleration], dtype=np.float32)

                # Compute reward and done
                collision = world.collision_sensor.get_collision_history()
                has_collision = any([x > 0 for x in collision.values()])
                power = calculate_power(kmh, acceleration)
                fuel_consumed = calculate_fuel_rate(power)
                total_fuel_consumed += fuel_consumed

                # Calculate distance traveled in this step
                d_trv = world.previous_location.distance(world.player.get_location()) 
                world.previous_location = world.player.get_location()

                # Calculate F_econ (avoid division by zero)
                f_econ = d_trv / fuel_consumed if fuel_consumed > 0 else 0.0

                # Get brake value
                control = world.player.get_control()
                b_ac = control.brake

                # Time spent in this step
                t_sp = delta_time

                if has_collision:
                    collision_happened = True
                    print("Collision detected! Neglecting episode.")
                    break  # Do not add this step to transitions, do not update anything
                else:
                    reward = (f_econ + kmh) - (b_ac + t_sp)

                if episode_start + SECONDS_PER_EPISODE < time.time():
                    done = True
                    print("Episode timeout! Ending episode.")

                episode_reward += reward
                episode_distance_traveled += d_trv

                # Buffer the transition
                if args.dqn:
                    episode_transitions.append((current_state, action, reward, new_state, done))
                else:
                    episode_transitions.append((current_state, throttle, reward, new_state, done))

                current_state = new_state
                step += 1

                if done or agent_nav.done():
                    print(f"Episode {episode} finished after {step} steps with reward: {episode_reward:.2f}, fuel consumed: {total_fuel_consumed:.2f} L")
                    break
            
            world.restart(args)

            agent_nav = BehaviorAgent(world.player, behavior=args.behavior)
            agent_nav.follow_speed_limits(True)
            
            agent_nav.set_destination(random.choice(spawn_points).location)
            if not collision_happened:
                world.hud.notification("Target reached", seconds=4.0)
                print("The episode finished, searching for another target")

            waypoint = world.map.get_waypoint(world.player.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is not None:
                new_transform = carla.Transform(
                    waypoint.transform.location,
                    carla.Rotation(
                        yaw=waypoint.transform.rotation.yaw,
                        pitch=0.0,
                        roll=0.0
                    )
                )
                world.player.set_transform(new_transform)
                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()
            else:
                print("[WARN] Could not align vehicle to lane at episode start.")

            # If collision happened, skip everything else
            if collision_happened:
                continue

            # Only update replay memory and stats for valid episodes
            for transition in episode_transitions:
                if args.dqn:
                    agent_rl.update_replay_memory(transition)
                else:
                    agent_rl.update_replay_memory(transition)

            agent_rl.tensorboard.step = episode
            fuel_per_distance = total_fuel_consumed / world.distance_traveled
            duration = time.time() - episode_start

            # Log per-episode stats to TensorBoard
            agent_rl.tensorboard.update_stats(
                reward=episode_reward,
                fuel_consumed=total_fuel_consumed,
                fuel_per_distance=fuel_per_distance,
                episode=episode,
                distance_traveled=episode_distance_traveled,
                steps=step,
                epsilon=epsilon,
                avg_speed= episode_distance_traveled/duration,
                duration=duration
            )

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            episode += 1

        agent_rl.terminate = True
        trainer_thread.join()
        if args.dqn:
            agent_rl.model.save(f'models/{MODEL_NAME}__{int(time.time())}.model')
        else:
            agent_rl.actor.save(f'models/{MODEL_NAME}__{int(time.time())}.model')

        world.destroy()
        pygame.quit()
    
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        try:
            if world is not None:
                world.destroy()
            pygame.quit()
        except Exception:
            pass