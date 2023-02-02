import math
from enum import Enum
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pennylane as qml

from SAC.sac_discrete.sacd.model import BaseNetwork
class DataEncodingForQVC(Enum):
  AMPLITUDE_ENCODING = 1
  ANGLE_ENCODING = 2

def layer_init(layer, init_type='default', nonlinearity='relu', w_scale=1.0):
  nonlinearity = nonlinearity.lower()
  # Initialize all weights and biases in layer and return it
  if init_type in ['uniform_', 'normal_']:
    getattr(nn.init, init_type)(layer.weight.data)
  elif init_type in ['xavier_uniform_', 'xavier_normal_', 'orthogonal_']:
    # Compute the recommended gain value for the given nonlinearity
    gain = nn.init.calculate_gain(nonlinearity)
    getattr(nn.init, init_type)(layer.weight.data, gain=gain)
  elif init_type in ['kaiming_uniform_', 'kaiming_normal_']:
    getattr(nn.init, init_type)(layer.weight.data, mode='fan_in', nonlinearity=nonlinearity)
  else: # init_type == 'default'
    return layer
  layer.weight.data.mul_(w_scale)
  nn.init.zeros_(layer.bias.data)
  return layer

# TODO: define this method for different experiment scenarios

def GetVQC(n_qubits: int, qnn_layers: int, qnn_type: str):
  if qnn_type == 'ReUploadingVQC':
    # TODO: Change this for ReUploadingVQC
    def ReUploadingVQC(inputs, entangling_weights, embedding_weights): # these are the shapes and not the actual weights.
      '''
      A variational quantum circuit (VQC) with data re-uploading
      '''
      # Prepare all zero state
      all_zero_state = torch.zeros(n_qubits)
      qml.BasisStatePreparation(all_zero_state, wires=range(n_qubits))
      for i in range(qnn_layers):
        # Variational layer
        qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
        # Encoding layer
        features = inputs * embedding_weights[i]
        qml.AngleEmbedding(features=features, wires=range(n_qubits))
      # Last varitional layer
      qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
      return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    # Get weight shape

    entangling_weights_shape = (qnn_layers+1, ) + qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
    embedding_weights_shape = (qnn_layers, n_qubits)
    weight_shapes = {
      'entangling_weights': entangling_weights_shape,
      'embedding_weights': embedding_weights_shape
    }
    return ReUploadingVQC, weight_shapes
  elif qnn_type == 'NormalVQC':
    def NormalVQC(inputs, entangling_weights):
      '''
      A variational quantum circuit (VQC) (without data re-uploading)
      '''
      # if(encoding == DataEncodingForQVC.AMPLITUDE_ENCODING):
      qml.AmplitudeEmbedding(features=inputs, wires= range(n_qubits), normalize=True, pad_with=0)
      # if(encoding == DataEncodingForQVC.ANGLE_ENCODING):
      #   qml.AngleEmbedding(features=inputs, wires=range(n_qubits))

      qml.StronglyEntanglingLayers(entangling_weights, wires=range(n_qubits))
      return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    entangling_weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=qnn_layers, n_wires=n_qubits)
    weight_shapes = {'entangling_weights': entangling_weights_shape}
    return NormalVQC, weight_shapes


class HybridQuantumQNetwork(torch.nn.Module):
  def get_n_qubits(self, input_dim: str, encoding: DataEncodingForQVC) -> int:
    if(encoding == DataEncodingForQVC.AMPLITUDE_ENCODING):
      return math.ceil(math.log(input_dim,2))
    else:
      return -1
    #TODO: implement this function for the rest

  def __init__(self, input_dim, output_dim, qnn_layers, qnn_type, last_w_scale, device, encoding=DataEncodingForQVC.AMPLITUDE_ENCODING):
    super().__init__()
    # Create a QNode
    n_qubits = self.get_n_qubits(input_dim=input_dim, encoding=encoding)
    if(device == 'cuda'):
      dev = qml.device('default.qubit.torch', wires=n_qubits, torch_device='cuda')
    else:
      dev = qml.device('default.qubit', wires=n_qubits)
    VQC, weight_shapes = GetVQC(n_qubits, qnn_layers, qnn_type)
    qnode = qml.QNode(VQC, dev, interface='torch', diff_method='best')
    self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    # Create a output layer : output_dim = n_actions
    self.output = layer_init(nn.Linear(n_qubits, output_dim), init_type='kaiming_uniform_', w_scale=last_w_scale)

  def forward(self, x):
    st = time.time()
    x = self.qlayer(x)
    x = self.output(x)
    elapsed_time = time.time()-st
    print('Execution time for Q network:', elapsed_time, 'seconds')
    return x




class TwinnedQuantumQNetwork(BaseNetwork):
  def __init__(self, input_dim, num_actions, qnn_layers, qnn_type, device, last_w_scale=1e-3):
    super().__init__()
    self.Q1 = HybridQuantumQNetwork(input_dim=input_dim, output_dim=num_actions, qnn_layers=qnn_layers, qnn_type=qnn_type, last_w_scale = last_w_scale,
                                    device=device)
    self.Q2 = HybridQuantumQNetwork(input_dim=input_dim, output_dim=num_actions, qnn_layers=qnn_layers, qnn_type=qnn_type, last_w_scale = last_w_scale,
                                    device=device)

  def forward(self, states):
    q1 = self.Q1(states)
    q2 = self.Q2(states)
    return q1, q2


