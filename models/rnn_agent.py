import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.NatureVisualEncoder import NatureVisualEncoder
import torch
from torch.nn import init
# from models.ICMModel import ICMModel
from models.ICMModel_2 import ICMModel
import numpy as np
import time
from torchrl.modules import NoisyLinear as NoisyLinearTorch
from models.NoisyLinear import NoisyLinear

from utils.utils import conv_output_shape
import traceback

class RNNAgent(nn.Module):
    def __init__(self, config, input_shape, device = None) -> None:
        super(RNNAgent, self).__init__()
        # drqn agent part
        self.config = config
        self.n_agents = config["num_agents"]
        self.device = device

        #Working
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape, config["rnn_hidden_dim"])

        if self.config["action_selector"] == "noisy":
            self.fc2 = nn.Sequential(
                NoisyLinear(config["rnn_hidden_dim"], config["rnn_hidden_dim"], sigma_init=0.017, device=self.device),
                nn.ReLU(),
                NoisyLinear(config["rnn_hidden_dim"], config["n_actions"], sigma_init=0.017, device=self.device)
            )
        else:
            # Working
            # self.fc1 = nn.Linear(input_shape, config["rnn_hidden_dim"])
            # self.fc2 = nn.Linear(config["rnn_hidden_dim"], config["n_actions"])

            # Removing all FFNN between CNN and RNN:
            self.fc2 = nn.Linear(config["rnn_hidden_dim"], config["n_actions"])


        # Working
        # self.rnn = nn.GRUCell(config["rnn_hidden_dim"], config["rnn_hidden_dim"])

        # Removing all FFNN between CNN and RNN:
        self.rnn = nn.GRUCell(self.input_shape , config["rnn_hidden_dim"])



    def init_hidden(self, hidden_state=None):

        # make hidden states on same device as model
        if self.config["use_burnin"] and hidden_state is not None:
            # copy the hidden state that was stored in the replay buffer into fc1
            try:
                pass
                self.fc1.weight.copy_(hidden_state)
            except Exception as e:
                traceback.print_exc()
        else:
            # return self.fc1.weight.new(1, self.config["rnn_hidden_dim"]).zero_()
            return torch.zeros(1, self.config["rnn_hidden_dim"])
            


    def forward(self, inputs, hidden_state, t, training = False):

        # Original way
        # x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.config["rnn_hidden_dim"])
        # h = self.rnn(x, h_in)
        # q = self.fc2(h)

        # Removing all FFNN between CNN and RNN:
        h_in = hidden_state.reshape(-1, self.config["rnn_hidden_dim"])
        h = self.rnn(inputs, h_in)
        q = self.fc2(h)



        return q, h
    
    def sample_noise(self):
        for layer in self.fc2:
            if isinstance(layer, NoisyLinear):
                layer.sample_noise()

    
    def _build_training_inputs(self, batch, t, feature):
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []

        feature = feature[:, t]

        inputs.append(feature)  # b1av

        # if self.args.obs_last_action:
        #     if t == 0:
        #         inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
        #     else:
        #         inputs.append(batch["actions_onehot"][:, t-1])

        if self.config["obs_agent_id"]:
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            


        inputs = torch.cat([x.reshape(bs*self.n_agents, -1).cuda() for x in inputs], dim=1)

        return inputs.cuda()


    def _build_batch_inputs(self, flat_obs, batch):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        next_obs = []
        obs = []
        
        # the feature extractor is used here to get features from the observations:
        next_obs.append(flat_obs[:,1:])  # b1av
        # B,T,N,V - [2,1000,2,512]

        obs.append(flat_obs[:,:-1])

        next_obs = torch.cat([x.reshape(bs, flat_obs.shape[1]-1, self.n_agents,-1).cuda() for x in next_obs], dim=-1)
        obs = torch.cat([x.reshape(bs, flat_obs.shape[1]-1, self.n_agents,-1).cuda() for x in obs], dim=-1)

        return next_obs.cuda(), obs.cuda()
    

    

    def min_max_norm(self,input):
        minval = torch.min(input, dim=-1, keepdim=True).values
        maxval = torch.max(input, dim=-1, keepdim=True).values
        
        normed = 2*((input-minval)/(maxval-minval))-1
        # normed = (scaled*2)-1

        return normed
    

    @staticmethod
    def argmaxed_action_from_probs(p, axis=1):
        # get the maximum action
        action = torch.argmax(p, dim=axis, keepdim=True)

        # convert it to a one-hotted version
        y_onehot = action.new(*action.shape[:-1], p.shape[-1]).zero_()
        y_onehot.scatter_(-1, action.long(), 1)
        return y_onehot.float()
        

