from models.NatureVisualEncoder import NatureVisualEncoder
# from modules.agents import RNNAgent, Advanced_RNNAgent
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
# from utils.rl_utils import z_score_norm

class ICMModel(nn.Module):
    """
        encoder_output_size should be equal to visual_encoder output size
        output_size = n_actions in environment
        """
    def __init__(self, output_size, observation_size, device) -> None:
        
        super(ICMModel, self).__init__()
        self.output_size = output_size
        self.encoder_output_size = observation_size
        self.device = device

        self.inverse_net = nn.Sequential(
            nn.Linear(observation_size*2, observation_size),
            nn.ReLU(),
            nn.Linear(observation_size, output_size)
        )

        # self.residual = [nn.Sequential(
        #     nn.Linear(output_size+observation_size, observation_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(observation_size, observation_size),
        # )]*8

        # self.residual = nn.ModuleList()
        # for i in range(8):
        #     self.residual.append(nn.Sequential(
        #         nn.Linear(output_size+observation_size, observation_size),
        #         nn.LeakyReLU(),
        #         nn.Linear(observation_size, observation_size),
        #     ))

        self.residual = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(output_size + observation_size, observation_size),
                    nn.LeakyReLU(),
                    nn.Linear(observation_size, observation_size),
                ) for _ in range(4)
            ]
        )

        self.residual.to(self.device)
        # for res in self.residual:
        #     res.cuda()

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size+observation_size, observation_size),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size+observation_size, observation_size),
        )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.xavier_uniform_(p.weight)
                p.bias.data.zero_()

        

    def forward(self, inputs):
        """
        inputs is a tuple of:
            observation: encoded visual observation
            next_observation: encoded visual observation
            action: one hotted actions
        Returns:
            real_next_obs: The actual next observation, encoded
            pred_next_obs: the predicted next observation, encoded
            pred_action: the predicted action
        """
        encode_observation, encode_next_observation, action = inputs

        # Test without this to see if it influences anything
        # encode_observation = z_score_norm(encode_observation)
        # encode_next_observation = z_score_norm(encode_next_observation)
        

        # ---------------
        # predict action
        pred_action = torch.cat((encode_observation, encode_next_observation), dim = -1)
        pred_action = self.inverse_net(pred_action)
        # ----------------

        # get predicted next observation
        # print(f"concat {encode_observation.shape} and {action.shape} in dim 1")
        # action = torch.squeeze(action, dim=0).cuda()
        action = action.to(self.device)
        pred_next_obs_feature_orig = torch.cat((encode_observation, action), dim =-1).to(self.device)
        pred_next_obs_feature_orig = self.forward_net_1(pred_next_obs_feature_orig).to(self.device)
        # ----------------

        # residual
        for i in range(2):
            res = torch.cat((pred_next_obs_feature_orig, action),dim=-1).to(self.device)
            pred_next_obs_feature = self.residual[i*2](res)
            pred_next_obs_feature_orig = self.residual[i*2+1](
                torch.cat((pred_next_obs_feature, action), dim=-1)) + pred_next_obs_feature_orig
        
        pred_next_obs_feature = self.forward_net_2(torch.cat((pred_next_obs_feature_orig, action), dim=-1))

        real_next_obs_feature = encode_next_observation

        return real_next_obs_feature, pred_next_obs_feature, pred_action