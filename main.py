import argparse
import logging
from simulation.simulation import game_loop
from rl.rl_training import rl_training_loop
from simulation.testing import test_model
from simulation.socket_server import socket_server


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
    '--rl',
    action='store_true',
    help='Enable RL fuel efficiency training mode')
    argparser.add_argument(
        '--test',
        action='store_true',
        help='Test trained RL model vs BasicAgent')
    argparser.add_argument(
        '--model-path',
        type=str,
        default='models/FuelOptimizer__1749141785.model',
        help='Path to trained RL model (default: models/FuelOptimizer__1749141785.model)'
    )
    argparser.add_argument(
        '--num-cars',
        type=int,
        default=50,
        help='Number of NPC vehicles to spawn in the simulation (default: 0)')
    argparser.add_argument(
    '--dqn',
    action='store_true',
    help='Use DQN agent instead of SAC (default: SAC)')
    argparser.add_argument(
        '--socket',
        action='store_true',
        help='Run simulation and serve data via TCP socket')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        if args.test:
            test_model(args, args.model_path)
        elif args.rl:
            rl_training_loop(args)
        elif args.socket:
            socket_server(args)
        else:
            game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()