import math
import os
from typing import Dict, List, Tuple

# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch.optim as optim
import torch.autograd as autograd 
import traceback
# USE_CUDA = torch.cuda.is_available()


class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, device = "cuda:0"):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.device = device
    self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if device=="cuda:0" else autograd.Variable(*args, **kwargs)
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features)).to(device)  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features)).to(device)  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features)).to(self.device)
      init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features)).to(self.device)
      init.constant_(self.sigma_weight, self.sigma_init).to(self.device)
      init.constant_(self.sigma_bias, self.sigma_init).to(self.device)

  def forward(self, input):
    return F.linear(input.to(self.device), self.weight.to(self.device) + self.sigma_weight.to(self.device) * self.Variable(self.epsilon_weight).to(self.device), self.bias.to(self.device) + self.sigma_bias.to(self.device) * self.Variable(self.epsilon_bias).to(self.device))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features).to(self.device)
    self.epsilon_bias = torch.randn(self.out_features).to(self.device)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features).to(self.device)
    self.epsilon_bias = torch.zeros(self.out_features).to(self.device)