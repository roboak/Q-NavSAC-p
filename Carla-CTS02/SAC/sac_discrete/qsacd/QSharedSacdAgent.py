from SAC.sac_discrete import DQNBase, CategoricalPolicy, TwinnedQNetwork
from SAC.sac_discrete.sacd.shared_sacd import SharedSacdAgent


class QSharedSacdAgent(SharedSacdAgent):
    def __init__(self):
        super()

    def createNetwork(self):
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CategoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)

        # TODO: Need to replace these two networks.

        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=self.dueling_net, shared=True).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2], self.env.action_space.n,
            dueling_net=self.dueling_net, shared=True).to(device=self.device).eval()


