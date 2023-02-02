from SAC.sac_discrete import DQNBase, CategoricalPolicy, TwinnedQNetwork
from SAC.sac_discrete.sacd.shared_sacd import SharedSacdAgent
from SAC.sac_discrete.qsacd.qnn_model import TwinnedQuantumQNetwork

class QuantumSharedSacdAgent(SharedSacdAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def createNetwork(self):
        self.conv = DQNBase(
            self.env.observation_space.shape[2]).to(self.device)
        self.policy = CategoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n,
            shared=True).to(self.device)
        #TODO: fix input_dimensions.
        self.online_critic = TwinnedQuantumQNetwork(input_dim=512, num_actions=self.env.action_space.n, qnn_layers=2, qnn_type="NormalVQC", device=self.device).to(device=self.device)
        self.target_critic = TwinnedQuantumQNetwork(input_dim=512, num_actions=self.env.action_space.n, qnn_layers=2, qnn_type="NormalVQC", device=self.device).to(device=self.device).eval()


