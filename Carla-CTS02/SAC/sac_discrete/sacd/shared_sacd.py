import math
import os
import numpy as np
import torch
from torch.optim import Adam

from SAC.sac_discrete.BaseAgent import BaseAgent
from SAC.sac_discrete.sacd.model import DQNBase, TwinnedQNetwork, CategoricalPolicy
from SAC.sac_discrete.utils import disable_gradients
from config import Config


class SharedSacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, learning_steps, completed_steps, display, tes=True, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0, path=None):
        super().__init__(
            env, test_env, log_dir, learning_steps, completed_steps, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed, display=display)

        self.createNetwork()
        if path != "None":
            self.resume = True
            if(str(self.device) == "cpu"):
                self.conv.load_state_dict(torch.load(os.path.join(path, "conv.pth"), map_location=torch.device("cpu")))
                self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pth"), map_location=torch.device("cpu")))
                self.online_critic.load_state_dict(torch.load(os.path.join(path, 'online_critic.pth'), map_location=torch.device("cpu")))
                self.target_critic.load_state_dict(torch.load(os.path.join(path, 'target_critic.pth'), map_location=torch.device("cpu")))
            else:
                self.conv.load_state_dict(torch.load(os.path.join(path, "conv.pth")))
                self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pth")))
                self.online_critic.load_state_dict(torch.load(os.path.join(path, 'online_critic.pth')))
                self.target_critic.load_state_dict(torch.load(os.path.join(path, 'target_critic.pth')))

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)
        # policy_optimiser should update the params for Conv as well.
        self.policy_optim = Adam(list(self.policy.parameters()) + list(self.conv.parameters()), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.tes = tes
        if(tes):
            self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n)
            self.std = 0
            self.flag_for_first_run = False
        else:
            self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr) # Testing with reduced learning rate for alpha to increase it's exploration capability.

    def createNetwork(self):
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CategoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)
        # observation_space = (0, 255, (400, 400, 3))
        # shape(observation_space) = (400, 400, 3)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n).to(device=self.device).eval()
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        print("DQNBase", self.conv)
        print("Online Critic", self.online_critic)
        print("Policy", self.policy)


    def explore(self, state):
        # Act with randomness.
        # (state(400x400x3) is the car intention, t is the record to be stored in replay buffer).
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            # here state is a 512 dimension vector
            state = self.conv((state, t))
            # state = torch.cat([state, t], dim=1)
            action, _, _ = self.policy.sample(state)
            curr_q1 = self.online_critic.Q1(state)
            curr_q2 = self.online_critic.Q2(state)
            q = torch.min(curr_q1, curr_q2)
            critic_action = torch.argmax(q, dim=1)
        return action.item(), critic_action.item()

    def exploit(self, state):
        # Act without randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv((state, t))
            # state = torch.cat([state, t], dim=1)
            # the following line makes the difference between explore and exploit.
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        states, t = states
        states = self.conv((states, t))
        # states = torch.cat([states, t], dim=-1)
        curr_q1 = self.online_critic.Q1(states).gather(1, actions.long())
        curr_q2 = self.online_critic.Q2(states.detach()).gather(1, actions.long())
        #  these are the values corresponding to (s,a) stored in the replay buffer
        #  -> output = batch_size x 1 (storing the Q value of the action in reply buffer corresponding to s)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_states, t_new = next_states
            next_states = self.conv((next_states, t_new))
            # next_states = torch.cat([next_states, t_new], dim=1)
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    # weights_dim = 128x1
    def calc_critic_loss(self, batch, weights):
        # curr_q1 is the Q value of (s,a) in replay buffer; dim = 128 x 1
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        # Mean of All q bvalues in the batch: (batch_size x 1) -> 1x1
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        states, t = states

        with torch.no_grad():
            states = self.conv((states, t))
        # states = torch.cat([states, t], dim=1)

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        # maximising -> policy_loss  = minimising -> -1*policy_loss
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    # Implementation of target entropy scheduler
    # policy_entropy = mean of the entropy of the batch.
    def _update_target_entropy(self, policy_entropy):
        #  Intialise mean_entropy to entropy of the policy when the training starts (max_entropy)
        if(not self.flag_for_first_run):
            # Initialisation of self.mean_entropy happens here - first time when this method is called.
            self.mean_entropy = policy_entropy
            self.flag_for_first_run = True
        else:
            delta = policy_entropy - self.mean_entropy
            self.mean_entropy = Config.exp_win_discount*self.mean_entropy + (1 - Config.exp_win_discount)*policy_entropy
            #Intitialised self.std to 0 in init()
            self.std = Config.exp_win_discount*(self.std + (1-Config.exp_win_discount)*delta**2)
            var = math.sqrt(self.std)
            if not ( self.target_entropy - Config.avg_threshold < self.mean_entropy < self.target_entropy + Config.avg_threshold) or var > Config.std_threshold:
                # target entropy is not changed in this case
                return
            else:
                # update target_entropy
                self.target_entropy = Config.entropy_discount_factor * self.target_entropy

    # Method defined here as a getter used to log target_entropy in the BaseAgent class
    def get_target_entropy(self):
        return self.target_entropy

    # Entropy loss is used to tune alpha.
    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad
        # Id tes( Target Entropy Scheduler is activated then update target entropy).
        if self.tes:
            self._update_target_entropy(entropies.mean().item())
        # Intuitively, we increase alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.conv.save(os.path.join(save_dir, 'conv.pth'))
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
