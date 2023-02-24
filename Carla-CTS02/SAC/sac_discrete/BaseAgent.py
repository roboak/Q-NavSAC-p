from abc import ABC, abstractmethod
import os
import shutil
from time import sleep
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from SAC.sac_discrete.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from SAC.sac_discrete.utils import update_params, RunningMeanStats
from config import Config
from tqdm.auto import trange
# update_interval: network is trained after every 4 steps in an episode
class BaseAgent(ABC):
    # TODO: Identify what is per
    def __init__(self, env, test_env, log_dir, learning_steps, completed_steps, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=500, save_interval=100000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0, display=False, resume=False):
        super().__init__()
        self.env = env
        self.test_env = test_env
        self.resume = resume

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        # self.env.seed(seed)
        # self.test_env.seed(2**31-1-seed)

        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.file = open(self.summary_dir + "eval_results.log", "w")

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = learning_steps
        self.completed_steps = completed_steps
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.display = display
        # self.save_interval = save_interval

    def run(self):
        # pbar = trange(int(self.num_steps), unit=" episode", leave=True, position=0)
        # for i in pbar:
        #     pbar.set_description(f"Running step {i+1} of total_steps {int(self.num_steps)}")
        #     sleep(0.05)
        #     self.train_episode()
        #     pbar.refresh()
        #
        # pbar.close()
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    ## Rresponsible for executing a single epoisode  and storing the episode in replay buffer
    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        nearmiss = False
        accident = False
        goal = False

        # randomly initialise the state
        state = self.env.reset()
        action_count = {0: 0, 1: 0, 2: 0}
        action_count_critic = {0: 0, 1: 0, 2: 0}

        t = np.zeros(6)  # reward, vx, vy, onehot last action
        t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

        # while a single episode ends;
        # max_episode_steps: Max number of steps in anm
        pbar = trange(int(self.max_episode_steps), unit=" steps", leave=True, position=0)
        for i in pbar:
            pbar.set_description(f"Running step {i} of episode {self.episodes}")
            pbar.refresh()
            # sleep(0.05)

        # while (not done) and episode_steps < self.max_episode_steps:
            if(done):break
            if self.display:
                self.env.render()

            #get action corresponding to state = state
            #sample random actions until steps < start_steps or in other words the mnodel is used to sample actions
            # only once replay buffer has certain number of episodes
            if self.start_steps > self.steps:
                if self.resume:
                    action, critic_action = self.explore((state, t))
                else:
                    # Code Modification - Have more actions for acceleration than braking and maintain speed in the beginning.
                    action = np.random.choice([0,1,2],p=[0.5,0.3,0.2])
                    critic_action = action
            else:
                action, critic_action = self.explore((state, t))

            next_state, reward, done, info = self.env.step(action)
            action_count[action] += 1
            action_count_critic[critic_action] += 1

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)

            #TODO: Understand the role of mask

            if episode_steps + 1 == self.max_episode_steps:
                mask = False
            else:
                mask = done
            # mask = False if episode_steps + 1 == self.max_episode_steps else done

            t_new = np.zeros(6)
            t_new[0] = clipped_reward
            t_new[1] = info['velocity'].x / Config.max_speed
            t_new[2] = info['velocity'].y / Config.max_speed
            t_new[3 + action] = 1.0

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append((state, t), action, clipped_reward, (next_state, t_new), mask)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state
            t = t_new
            nearmiss = nearmiss or info['near miss']
            accident = accident or info['accident']
            goal = info['goal']
            done = done or accident

            # learning happens happens after certain interval and after there is enough data(after steps>start_steps)
            # Note: Learning happens multiple times in a single episode.

            if self.is_update():
                # print("learning the model")
                self.learn()
                # checkpointing the model after every Config.model_checkpointing_interval steps
                if(self.learning_steps % Config.model_checkpointing_interval == 0):
                    self.save_models(os.path.join(self.model_dir, str(self.learning_steps)+'_'+str(self.completed_steps+self.steps)))
                    self._clear_checkpoints()

            # else:
            #     print("not learning the model")

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                if self.steps > self.start_steps:
                    self.evaluate()



            # if self.steps % self.save_interval == 0:
            #     self.save_models(os.path.join(self.model_dir, str(self.steps)))


        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.completed_steps+self.steps)


        pbar.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            self.episodes, info['scenario'], info['ped_speed'], info['ped_distance']))
        pbar.write('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, accident, nearmiss))
        pbar.write('Total steps: {}, Episode steps: {}, Reward: {:.4f}'.format(self.steps, episode_steps, episode_return))
        pbar.write("Policy: {}, Critic: {}, Alpha: {:.4f}".format(action_count,action_count_critic,self.alpha.item()))
        pbar.update()
        pbar.close()

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1
        # print("learning steps: {}, steps: {}".format(self.learning_steps, self.steps))
        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            # TODO: What's the intent behind plotting mean_Q1 and mean_Q2 ?
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def evaluate(self):
        num_episodes = len(self.test_env.episodes)
        num_steps = 0
        total_return = 0.0
        total_goal = 0
        print('-' * 60)
        pbar = trange(num_episodes, unit="episode", leave=True, position=0)
        for ep_it in pbar:
            pbar.set_description(f"Evaluation Phase: running episode {ep_it} of {num_episodes}")
            # get new scene. In this case, the episodes re sequentially sampled from the episodes list.
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            action_count = {0: 0, 1: 0, 2: 0}
            t = np.zeros(6)  # reward, vx, vt, onehot last action
            t[3 + 1] = 1.0  # index = 3 + last_action(maintain)

            while (not done) and episode_steps < self.max_episode_steps:
                action = self.exploit((state, t))
                next_state, reward, done, info = self.test_env.step(action)
                action_count[action] += 1
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state


                t = np.zeros(6)
                t[0] = max(min(reward, 2.0), -2.0)
                t[1] = info['velocity'].x / Config.max_speed
                t[2] = info['velocity'].y / Config.max_speed
                t[3 + action] = 1.0
                done = done or info["accident"]

            # print("Evaluation episode number: {}/{}".format(ep_it,num_episodes))
            # num_episodes += 1 #TODO: Check if this was the right thing to do??
            total_return += episode_return
            total_goal += int(info['goal'])
            pbar.write("Scenario_Id: {}, Ped_Speed: {:.2f}m/s, Ped_Dist.: {:.2f}m, Return: {:.4f}".format(info['scenario_id'],
                info['ped_speed'], info['ped_distance'], episode_return))
            pbar.write("Goal: {}, Accident: {}, Act Dist.: {}".format(info['goal'], info['accident'], action_count))
            pbar.update()
            self.file.write("Ped_Speed: {:.2f}m/s, Ped_Dist.: {:.2f}m, Return: {:.4f}".format(
                info['ped_speed'], info['ped_distance'], episode_return))
            self.file.write("Goal: {}, Accident: {}, Act Dist.: {}".format(
                info['goal'], info['accident'], action_count))

            # if num_steps > self.num_eval_steps:
            #     break
        pbar.close()
        mean_return = total_return / num_episodes

        # if mean_return > self.best_eval_score:
        if total_goal > self.best_eval_score:
            self.best_eval_score = total_goal
            self.save_models(os.path.join(os.path.join(self.log_dir, 'best'), str(self.learning_steps)+'_'+str(self.steps)))
        self.writer.add_scalar(
            'reward/test', mean_return, self.completed_steps + self.steps)
        #  understand the meaning of total_goals -> total number of times, goal was achieved
        self.writer.add_scalar(
            'reward/goal', total_goal, self.completed_steps + self.steps)
        self.test_env.test_episodes_iter = iter(self.test_env.episodes)  #  This is done to initialise the iterator after each evaluation step.

        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print(f'Num steps: {self.steps:<5}  '
              f'goal return: {total_goal}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    def _clear_checkpoints(self):
        saved_models = sorted(Path(self.model_dir).iterdir(), key=os.path.getmtime)
        while len(saved_models) > Config.max_checkpoints:
            x = saved_models.pop(0)
            shutil.rmtree(x.as_posix())

    # def __del__(self):
    #     self.env.close()
    #     self.test_env.close()
    #     self.file.close()
