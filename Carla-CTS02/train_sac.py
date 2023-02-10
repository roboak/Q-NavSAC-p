import os
import yaml
import argparse
import subprocess
import time
from datetime import datetime
from multiprocessing import Process


from SAC.sac_discrete.sacd.shared_sacd import SharedSacdAgent
from SAC.sac_discrete.qsacd.QuantumSharedSacdAgent import QuantumSharedSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = GIDASBenchmark(port=Config.port, mode="TRAINING")
    test_env = GIDASBenchmark(port=Config.port + 100, mode="PARTIAL_TESTING")

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    name += "-qsac-" if args.qsac else ""
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        '_out', args.env_id, f'{name}-seed{args.seed}-{time}')
    print("log_dir:", log_dir)
    # Create the agent.
    # path = "_out/GIDASBenchmark/shared-sacd-seed0-20220303-1356/model/3000000/"
    config['num_steps'] = 2e6
    config['env'] = env
    config['test_env'] = test_env
    config['log_dir'] = log_dir
    config['cuda'] = args.cuda
    config['seed'] = args.seed
    Agent = QuantumSharedSacdAgent if args.qsac else SharedSacdAgent
    agent = Agent(**config)
    agent.run()


def run_server(local: bool, mode = str):
    # train environment
    if mode == "train":
        port = "-carla-port={}".format(2000)
    else:
        port = "-carla-port={}".format(2100)
    if local:
        print("executing locally")
        subprocess.run("cd D:/CARLA_0.9.13/WindowsNoEditor && CarlaUE4.exe " + port, shell=True)
    else:
        print("executing on slurm cluster")
        subprocess.run(['cd /netscratch/sinha/carla && unset SDL_VIDEODRIVER && ./CarlaUE4.sh -vulkan -RenderOffscreen -nosound ' + port], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='SAC/sac_discrete/config/sacd.yaml')
    parser.add_argument('--qsac', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()

    Config.port = args.port
    print('Env. port: {}'.format(Config.port))

    # p = Process(target=run_server, args=(args.local, "train", ))
    # p.start()
    # time.sleep(12)
    # p = Process(target=run_server, args=(args.local, "test", ))
    # p.start()
    # time.sleep(12)

    run(args)
