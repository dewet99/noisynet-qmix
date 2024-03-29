import ray
from collections import deque
import numpy as np
from envs import REGISTRY as env_REGISTRY
from functools import partial
import os
import torch
from controllers.custom_controller import CustomMAC

from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from components.replay_buffer import EpisodeBatch
from utils.utils import OneHot
import torch.nn.functional as F
import torch.nn as nn 
import time

import traceback
import datetime
import sys
import pdb

from torch.utils.tensorboard import SummaryWriter

if sys.platform == "linux":
    parent_directory = os.path.join(os.getcwd(), os.pardir)
    sc2_path = os.path.join(parent_directory, "3rdparty", "StarCraftII")
    os.environ.setdefault("SC2PATH",sc2_path)


@ray.remote(num_cpus = 1,num_gpus=0.0001, max_restarts=20)
class Test_Executor(object):
    def __init__(self,config, worker_id):
        super().__init__()
        # Set config items
        self.config = config
        
        self.batch_size = config["batch_size_run"]

        env_args = self.config["env_args"]

        # The executor processes should always run on CPU, to save GPU resources for training
        self.device = torch.device("cpu")
        self.worker_id = worker_id
        
        self.env = StarCraftCapabilityEnvWrapper(
            **env_args,
        )
        
        self.get_env_info()
        self.setup()
        self.setup_logger()

        self.total_t = 0
        self.last_test_T = -config["test_interval"]-1


    def run(self):
        try:
            # Sleep for a few seconds to give everything time to be initialised
            time.sleep(3)
            param_updates = 0

            while param_updates<self.config["t_max"] + self.config["worker_parameter_sync_frequency"]:
                # Check if a test episode needs to be run, every second
                time.sleep(1)
                param_updates = ray.get(self.parameter_server.get_parameter_update_steps.remote())

                # Perform test episodes based on the total average of steps done by all workers
                num_steps = 0
                for i in range(self.config["num_executors"]):
                    num_steps += ray.get(self.parameter_server.get_worker_steps_by_id.remote(i))
                
                num_steps= int(num_steps/self.config["num_executors"])

                self.total_t = num_steps

                if (self.total_t - self.last_test_T)/self.config["test_interval"] >= 1.0:
                    # First get the latest parameters from the parameter server
                    self.sync_with_parameter_server()

                    n_test_ep = self.config["n_test_episodes"]
                    print(f"Testing Executor {self.worker_id} running {n_test_ep} test episodes at local step {self.total_t}")
                    self.last_test_T = self.total_t
                    

                    # Run test episodes
                    test_rewards = 0
                    test_episode_lengths = 0

                    for _ in range(n_test_ep):
                        reward_episode, ep_length = self.collect_experience(test_mode=True)

                        test_rewards+=reward_episode
                        test_episode_lengths+=ep_length

                    stats_dict = self.env.get_stats()
                    self.parameter_server.log_test_stats.remote(test_rewards, test_episode_lengths, stats_dict, self.total_t, n_test_ep)

                    self.parameter_server.set_can_log_test.remote(True)
                    
                    # Reset environment battle stats so win rate can be calculated based only on current iteration of evaluation episodes
                    self.reset_testing_executor_battle_stats()
                    print(f"Executor {self.worker_id} done testing")

            sys.exit(1)

        except Exception as e:
            traceback.print_exc()

    def collect_experience(self, test_mode = False):
        """
        Runs an episode and stores the collected information in the replay buffer
        """
        self.reset()
        episode_start = time.time()

        try:
            # THIS HAS POTENTIALLY BEEN DEPRECATED
            pass
            # This way, epsilon is calculated based on the worker's individual steps.I.e epsilon is independent of the number of workers, so each worker
            # has to step eg 500k times (depending on config) for epsilon to reach its min value
            # global_steps = ray.get(self.parameter_server.get_worker_steps_by_id.remote(self.worker_id))

            # This way, epsilon decays based on the total number of steps. So if you have four workers and each has stepped 125k times, then
            # epsilon is calculated fort each worker as if the total number of steps is 500k
            # global_steps = ray.get(self.parameter_server.return_total_steps_all_workers.remote(self.worker_id))
        except Exception as e:
            print(e)
        terminated = False
        episode_return = 0

        self.mac.init_hidden(batch_size=self.batch_size, hidden_state=None)
        raw_observations = 0

        reward_episode = []
        try:
            while not terminated:
                # Convert the observations to uint8 to reduce the RAM requirements, this allows you to set a significantly larger replay size than with float32
                raw_observations = self.env.get_obs()
                
                state = self.env.get_state()

                pre_transition_data = {
                    "state": state,
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }
                
                try:
                    if self.config["use_burnin"]:
                        # store the hidden state in the replay buffer so that we can use it during training to init hidden
                        h_s = self.mac.hidden_states.detach()
                        hidden_state = {"hidden_state": h_s.unsqueeze(0)}
                        pre_transition_data.update(hidden_state)
                except Exception as e:
                    traceback.print_exc()
                

                self.batch.update(pre_transition_data, ts=self.t)


                # Pass the entire batch of experiences up till now to the mac to select actions
                with torch.no_grad():
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.total_t, test_mode=test_mode)
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

            raw_observations = self.env.get_obs()

            last_data = {
                    "state": self.env.get_state(),
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": raw_observations
                }
            
            self.batch.update(last_data, ts=self.t)

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.total_t, test_mode=test_mode)

            self.batch.update({"actions": actions}, ts=self.t)


            # Accumulate test reward stats in parameter server
            # get_stats_dict = self.env.get_stats()
            # self.parameter_server.accumulate_test_stats.remote(sum(reward_episode), self.t, get_stats_dict, self.total_t)


            return episode_return, self.t
        
        except Exception as e:
            traceback.print_exc()


    def set_remote_objects(self, parameter_server):
        self.parameter_server = parameter_server


    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        

    def setup(self):
        scheme, groups, preprocess = self.generate_scheme()
    
        self.mac = CustomMAC(self.config, device = self.device)
        # THe agent used in testing shouldn't have any noise in their 2 layers
        self.mac.remove_agent_noise()

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.config["batch_size_run"], self.config["episode_limit"]+1, preprocess = preprocess, device = "cpu")

    def get_env_info(self):
        self.env_info = self.env.get_env_info()

        self.config["obs_shape"] = self.env_info["obs_shape"]
        self.config["n_actions"] = self.env_info["n_actions"]
        self.config["n_agents"] = self.env_info["n_agents"]
        self.config["episode_limit"] = self.env_info["episode_limit"]

    def setup_logger(self):
        self.log_dir = "results/" + self.config["name"] + str(self.config["env_args"]["capability_config"]["n_units"]) \
            + "_vs_" + str(self.config["env_args"]["capability_config"]["n_enemies"]) + "_" + datetime.datetime.now().strftime("%d_%m_%H_%M")
    def close_env(self):
        self.env.close()

    def generate_scheme(self):
        self.config["state_shape"] = self.env_info["state_shape"]

        scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
            "avail_actions": {"vshape": (self.env_info["n_actions"],), "group": "agents", "dtype": torch.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            }
        
        if self.config["use_burnin"]:
            hidden_states = {"hidden_state": {"vshape": (1, self.env_info["n_agents"],self.config["rnn_hidden_dim"]), "dtype": torch.float32}}
            scheme.update(hidden_states)
        
        groups = {
        "agents": self.env_info["n_agents"]
        }

        preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=self.env_info["n_actions"])])
        }


        return scheme, groups, preprocess
    
    def retrieve_updated_config(self):
        return self.config
    
    def sync_with_parameter_server(self):
        # receive the stored parameters from the server using ray.get()

        new_params = ray.get(self.parameter_server.return_params.remote())

        for param_name, param_val in self.mac.named_parameters():
            if param_name in new_params:
                param_data = torch.tensor(ray.get(new_params[param_name])).to(self.device)
                param_val.data.copy_(param_data)

    def reset_testing_executor_battle_stats(self):
        self.env.env.battles_won = 0
        self.env.env.battles_game = 0




