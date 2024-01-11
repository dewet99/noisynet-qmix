import ray
import pdb
from collections import deque
import numpy as np
import gc
from functools import partial
from utils.unity_utils import get_worker_id
import torch
from controllers.custom_controller import CustomMAC
from models.ICMModel_2 import ICMModel
from components.replay_buffer import EpisodeBatch
from utils.utils import OneHot, RunningMeanStdTorch
import torch.nn.functional as F
import torch.nn as nn 
import time

import traceback


import os
import datetime
import sys

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from models.NatureVisualEncoder import NatureVisualEncoder


class TestExecutor(object):
    def __init__(self,config, worker_id):
        super().__init__()
        # Set config items
        self.time_scale = config["test_timescale"]
        self.env_path = config["test_executable_path"]
        self.episode_limit = config["episode_limit"]
        self.config = config
        self.batch_size = config["batch_size_run"]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.worker_id = worker_id

        # Set class variables
        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(time_scale=self.time_scale)

        try:
            self.env.close()
            self.unity_env.close()
        except:
            print("No envs open")

        unity_env = UnityEnvironment(file_name=self.env_path, worker_id=get_worker_id(), seed=np.int32(np.random.randint(0, 120)), side_channels=[config_channel])
        unity_env.reset()

        self.env = UnityWrapper(unity_env, config_channel, episode_limit=self.episode_limit, config = self.config)
        self.env.reset()

        self.get_env_info()
        self.setup()
        self.setup_logger()


    def collect_experience(self):
        self.reset()
        terminated = False
        episode_return = 0

        self.mac.init_hidden(batch_size=self.batch_size, hidden_state=None)
        raw_observations = 0

        reward_episode = []
        try:
            while not terminated:
                raw_observations = np.uint8(self.env._get_observations()*255)
                state = self.env._get_global_state_variables()

                pre_transition_data = {
                    "state": state,
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }

                self.batch.update(pre_transition_data, ts=self.t)

                with torch.no_grad():
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, test_mode=True)
                    reward, terminated, env_info = self.env.step(actions[0])

            
                reward_episode.append(reward)


                episode_return += reward


                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }

                self.batch.update(post_transition_data, ts=self.t)

                self.t += 1

            raw_observations = np.uint8(self.env._get_observations()*255)

            last_data = {
                    "state": self.env._get_global_state_variables(),
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }
            
            self.batch.update(last_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=0, test_mode=True)

            self.batch.update({"actions": actions}, ts=self.t)
            self.test_episode+=1

            if self.config["save_obs_for_debug"]:
                np.save("testing_obs", self.batch["obs"].cpu().numpy())

            return self.batch["reward"], self.t

        except Exception as e:
            traceback.print_exc()
    

    def run(self):
        try:
            time.sleep(3)

            reward_data = []
            episode_length_data = []
            episode_id = []

            while self.test_episode<self.config["num_test_episodes"]:    
                rewards, episode_length = self.collect_experience()
                reward_data.append(np.array(rewards))
                episode_length_data.append(episode_length)
                episode_id.append(self.test_episode)
                
            
            ID = self.config["test_models_path"].split('/')[2]

            os.mkdir(f"./eval_results/{ID}")
            np.save(f"eval_results/{ID}/rewards", reward_data)
            np.save(f"eval_results/{ID}/episode_length_data", episode_length_data)
            np.save(f"eval_results/{ID}/episode_id", episode_id)
            sys.exit(1)
        except Exception as e:
            traceback.print_exc()


    def set_remote_objects(self, remote_buffer, parameter_server):
        self.remote_buffer = remote_buffer
        self.parameter_server = parameter_server


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
    
    def setup(self):
        scheme, groups, preprocess = self.generate_scheme()
        

        
        self.encoder = NatureVisualEncoder(self.config["obs_shape"][0],
                                           self.config["obs_shape"][1],
                                           self.config["obs_shape"][2],
                                           self.config,
                                           device = self.device
                                           )


        self.mac = CustomMAC(self.config, encoder = self.encoder, device = self.device)

        # Load models:
        path = self.config["test_models_path"]

        if path!= "":
            self.encoder.load_state_dict(torch.load(path+"/encoder.th"))
            self.mac.load_models(path)

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.config["batch_size_run"], self.config["episode_limit"]+1, preprocess = preprocess, device = "cpu")
        self.test_episode = 0


    def get_env_info(self):
        self.config["obs_shape"] = self.env.obs_shape
        self.env_info = self.env.get_init_env_info()
        self.config["n_actions"] = self.env_info["n_actions"]


    def generate_scheme(self):
        self.config["state_shape"] = self.env_info["state_shape"]

        scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents", "dtype": torch.uint8},
            "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
            "avail_actions": {"vshape": (self.env_info["n_actions"],), "group": "agents", "dtype": torch.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            }
        
        if self.config["curiosity"]:
            icm_reward = {"icm_reward": {"vshape": (1,)},}
            scheme.update(icm_reward)

        if self.config["use_burnin"]:
            hidden_states = {"hidden_state": {"vshape": (1, 2,self.config["rnn_hidden_dim"]), "dtype": torch.float32}}
            scheme.update(hidden_states)

        groups = {
        "agents": self.config["num_agents"]
        }

        preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=self.config["n_actions"])])
        }

        return scheme, groups, preprocess
    
