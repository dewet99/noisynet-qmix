from models.NatureVisualEncoder import NatureVisualEncoder
# from modules.agents import RNNAgent, Advanced_RNNAgent
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional, Union

from utils.utils import conv_output_shape
import copy

class ICMModel(nn.Module):
    """
        encoder_output_size should be equal to visual_encoder output size
        output_size = n_actions in environment
        """
    def __init__(self, output_size, observation_size, device, input_obs_shape, config, encoder = None) -> None:
        
        super().__init__()
        self.output_size = output_size #num actions available
        self.encoder_output_size = observation_size
        self.device = device
        self.initial_channels = input_obs_shape[2]
        self.obs_h = input_obs_shape[0]
        self.obs_w = input_obs_shape[1]
        self.config = config

        print(f"Obs height is: {self.obs_h}")
        print(f"Obs width is: {self.obs_w}")

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        # deep copy the encoder because we don't want the MAC encoder to be updated from the ICM encoder.
        # self.icm_encoder = copy.deepcopy(encoder)
        # assert encoder is not None, "Encoder is None, pass an encoder as reference to the ICM"
        
        # This is replaced by a reference to the feature extractor. This is so we can isolate the encoder to
        # save its parameters to the parameter server, so the encoder is update from the ICM and then also used to extract
        # features for the agent to train from

        conv_1_hw = conv_output_shape((input_obs_shape[0] , input_obs_shape[1]), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 32


        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.initial_channels, 32, [8, 8], [4, 4]),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, [3, 3], [1, 1]),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_flat, self.encoder_output_size),
            nn.ReLU(),
            # nn.Linear(self.encoder_output_size, self.encoder_output_size),
            # nn.Tanh()
        )


        self.inverse_net = nn.Sequential(
            nn.Linear(observation_size*2, observation_size),
            nn.ELU(),
            nn.Linear(observation_size, output_size),
            nn.ELU()
        )

        self.phi_hat_new = nn.Sequential(
            nn.Linear(output_size+observation_size, observation_size), 
            nn.ELU(),
            nn.Linear(observation_size, observation_size),
            nn.ELU()
        )

    
        # self.residual = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(output_size + observation_size, observation_size),
        #             nn.LeakyReLU(),
        #             nn.Linear(observation_size, observation_size),
        #         ) for _ in range(4)
        #     ]
        # )

        # self.residual.to(self.device)
        # # for res in self.residual:
        # #     res.cuda()

        # self.forward_net_1 = nn.Sequential(
        #     nn.Linear(output_size+observation_size, observation_size),
        #     nn.LeakyReLU()
        # )
        # self.forward_net_2 = nn.Sequential(
        #     nn.Linear(output_size+observation_size, observation_size),
        # )

        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)

        

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
        # encode_observation, encode_next_observation, action = inputs
        obs, next_obs, action = inputs
        batch_size = obs.shape[0]
        

        # reshape to [num_agents, num_timesteps, obs_size]
        # phis = self.dense(self.conv_layers(batch_of_observations.reshape(-1, self.config["obs_shape"][2],self.config["obs_shape"][0],self.config["obs_shape"][1]).cuda())).reshape(self.config["num_agents"], -1, self.config["encoder_output_size"]) 
        # print(f"obs shape: {obs.shape}")
        # print(f"next obs: {next_obs.shape}")
        if obs.dtype == torch.uint8:
            obs = obs.to(torch.float32)/255

        if next_obs.dtype == torch.uint8:
            next_obs = next_obs.to(torch.float32)/255
        
        # np.save("obs_debug", obs.detach().cpu().numpy())
        # obs = self.dense(self.conv_layers(obs.reshape(-1, self.config["obs_shape"][2],self.config["obs_shape"][0],self.config["obs_shape"][1]).cuda()))
        # next_obs = self.dense(self.conv_layers(next_obs.reshape(-1, self.config["obs_shape"][2],self.config["obs_shape"][0],self.config["obs_shape"][1]).cuda()))

        # prep obs
        obs = obs.cuda().reshape(-1, self.obs_w , self.obs_h, self.initial_channels).permute(0,3,1,2)
        next_obs = next_obs.cuda().reshape(-1, self.obs_w , self.obs_h, self.initial_channels).permute(0,3,1,2)

        obs = self.dense(self.conv_layers(obs)).reshape(batch_size, -1, self.encoder_output_size)
        next_obs = self.dense(self.conv_layers(next_obs)).reshape(batch_size, -1, self.encoder_output_size)

        #encodes the entire batch of observations

        # self.conv_layers(batch["obs"][:, t].squeeze().permute([0,3,1,2]).cuda()))

        # Test without this to see if it influences anything
        # encode_observation = z_score_norm(encode_observation)
        # encode_next_observation = z_score_norm(encode_next_observation)
        
        # TEMP
        # next_obs.append(flat_obs[:,1:])  # b1av
        # # B,T,N,V - [2,1000,2,512]

        # obs.append(flat_obs[:,:-1])

        # ---------------
        # predict action
        pred_action = torch.cat((obs, next_obs), dim = -1)
        pred_action = self.inverse_net(pred_action)

        # print(f"pred action shape: {pred_action.shape}")
        # print(f"real action shape: {action.shape}")
        # ----------------

        # get predicted next observation
        # print(f"concat {encode_observation.shape} and {action.shape} in dim 1")
        # action = torch.squeeze(action, dim=0).cuda()
        action = action.to(self.device)


        # print(phis[:, :-1].squeeze().permute(1,0,2).shape)
        # print(action.shape)

        

        predict_phi__input = torch.cat((obs, action.squeeze()), dim =-1).to(self.device)

        

        phi_hat_new = self.phi_hat_new(predict_phi__input)

        # print(f"phi_hat_new shape: {phi_hat_new.shape}")

        return next_obs, phi_hat_new, pred_action
    
    def calculate_icm_reward(self, inputs):
        next_obs, pred_next_obs, _ = self.forward(inputs)

        # L_I = (1-self.config["beta"])*self.ce(pred_action.contiguous().view(-1, self.config["n_actions"]).cuda(), inputs[2].contiguous().view(-1,self.config["n_actions"]).cuda())
        # L_F = self.config["beta"]*self.mse(pred_next_obs, next_obs)

        intrinsic_reward = self.config["lamda"] * 0.5 * ((pred_next_obs-next_obs).pow(2).mean(dim=-1).mean(dim=0))

        return intrinsic_reward
    
    def calculate_icm_loss(self, inputs):
        next_obs, pred_next_obs, pred_action = self.forward(inputs)

        L_I = (1-self.config["beta"])*self.ce(pred_action.contiguous().view(-1, self.config["n_actions"]).cuda(), inputs[2].contiguous().view(-1,self.config["n_actions"]).cuda())
        L_F = self.config["beta"]*self.mse(pred_next_obs, next_obs)

        intrinsic_reward = self.config["lamda"] * 0.5 * ((pred_next_obs-next_obs).pow(2))

        return L_I, L_F, intrinsic_reward
    


