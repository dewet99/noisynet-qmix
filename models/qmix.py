import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

class QMixer(nn.Module):
    def __init__(self, config):
        super(QMixer, self).__init__()

        self.config = config
        self.n_agents = config["num_agents"]

        # State dim include available actions, so add the above again:
        self.state_dim=self.n_agents*config["n_actions"]

        self.state_dim+=config["state_shape"]

        self.embed_dim = config["mixing_embed_dim"]

        if config["hypernet_layers"] == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        elif config["hypernet_layers"] == 2:
            hypernet_embed = self.config["hypernet_embed"]
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            
        elif config["hypernet_layers"]  > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim).to(th.float32)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))     
        b1 = self.hyper_b_1(states)      
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
