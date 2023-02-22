import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

        self.dim_reduction = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True))

    def forward(self, states):
        # this helps reorganise the states from (1x400x400x3) -> (1x3x400x400)
        state, t = states
        state = state.permute(0, 3, 1, 2)
        out = self.net(state)
        out = torch.cat([out, t], dim=1)
        out = self.dim_reduction(out)
        return out

class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        # self.conv = DQNBase(num_channels)
        # self.head = nn.Sequential(
        #         nn.Linear(46 * 46 * 64 + 6, 512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512, num_actions))

        self.head = nn.Linear(512, num_actions)


    def forward(self, states):
        # states = self.conv(states)
        return self.head(states)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CategoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        # if not shared:
        #     self.conv = DQNBase(num_channels)

        # self.head = nn.Sequential(
        #     nn.Linear(46 * 46 * 64 + 6, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_actions))
        self.head = nn.Linear(512, num_actions)


    def act(self, states):
        # if not self.shared:
        #     states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, probs=None, steps=None):
        # if not self.shared:
        #     states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        if probs is not None and steps is not None:
            action_probs = probs
            # if steps < Config.pre_train_steps:
            #     action_probs = probs
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
