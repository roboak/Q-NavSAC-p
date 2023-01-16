import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process


from SAC.sac_discrete import SacdAgent
from SAC.sac_discrete.shared_sacd import SharedSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = GIDASBenchmark(port=Config.port)
    test_env = GIDASBenchmark(port=Config.port + 100, setting="special")

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    # path = "_out/GIDASBenchmark/shared-sacd-seed0-20220303-1356/model/3000000/"
    config['num_steps'] = 2e6
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda, seed=args.seed, **config)
    agent.run()


def run_server():
    # train environment
    port = "-carla-port={}".format(Config.port)
    # subprocess.run(['cd C:/Users/aksi01/Documents/CARLA_0.9.13/WindowsNoEditor && SDL_VIDEODRIVER=offscreen CarlaUE4 -opengl ' + port], shell=True)
    subprocess.run("cd D:/CARLA_0.9.13/WindowsNoEditor && CarlaUE4.exe -RenderOffScreen "+port, shell=True)
    # Command to run in no rending mode on windows: CarlaUE4 -opengl -RenderOffScreen
    # Link: https://carla.readthedocs.io/en/latest/adv_rendering_options/


def run_test_server():
    # test environment
    port = "-carla-port={}".format(Config.port + 100)
    # subprocess.run(['cd C:/Users/aksi01/Documents/CARLA_0.9.13/WindowsNoEditor && SDL_VIDEODRIVER=offscreen ./CarlaUE4 -opengl ' + port], shell=True)
    subprocess.run("cd D:/CARLA_0.9.13/WindowsNoEditor && CarlaUE4.exe -RenderOffScreen "+port, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('SAC/sac_discrete/config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    Config.port = args.port
    print('Env. port: {}'.format(Config.port))

    p = Process(target=run_server)
    p.start()
    time.sleep(10)
    print('Run server started')

    p2 = Process(target=run_test_server)
    p2.start()
    time.sleep(8)

    run(args)
