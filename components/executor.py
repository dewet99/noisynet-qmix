import ray
from collections import deque
import numpy as np

from functools import partial

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



@ray.remote(num_cpus = 1,num_gpus=0.0001, max_restarts=20)
class Executor(object):
    def __init__(self,config, worker_id):
        super().__init__()
        # Set config items
        self.time_scale = config["time_scale"]
        self.env_path = config["executable_path"]
        self.episode_limit = config["episode_limit"]
        self.config = config
        self.batch_size = config["batch_size_run"]



        # The executor processes should always run on CPU, to save GPU resources for training
        self.device = torch.device("cpu")
        self.worker_id = worker_id

        capability_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }
        
        # print('================================')
        # capability_config["n_enemies"]
        # print('================================')
        
        env = StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            map_name="10gen_terran",
            debug=True,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
        )

        
        self.env_info = env.get_env_info()
        
        # self.get_env_info()
        self.setup()
        self.setup_logger()


    def collect_experience(self):
        """
        Runs an episode and stores the collected information in the replay buffer
        """
        self.reset()
        episode_start = time.time()

        try:
            global_steps = ray.get(self.parameter_server.get_worker_steps_by_id.remote(self.worker_id))
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
                raw_observations = np.uint8(self.env._get_observations()*255)

                state = self.env._get_global_state_variables()

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
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=global_steps, test_mode=False)
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

            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=global_steps, test_mode=False)

            self.batch.update({"actions": actions}, ts=self.t)

            
            # Parameter server keeps track of global steps and episodes
            # Add the number of steps in this executor's episode to the global count
            self.parameter_server.add_environment_steps.remote(self.t)
            
            # Increment global episode count
            self.parameter_server.increment_total_episode_count.remote()

            # Accumulate reward stats in parameter server
            self.parameter_server.accumulate_stats.remote(sum(reward_episode), time.time() - episode_start, self.t)
            self.parameter_server.accumulate_worker_steps_by_id.remote(self.worker_id, self.t)

            return self.batch
        
        except Exception as e:
            traceback.print_exc()
    

    def run(self):
        try:
            # Sleep for a few seconds to give everything time to be initialised
            time.sleep(3)
            param_updates = 0

            # add parameter sync frequency otherwise there will be one less logging point than we want. Or something. Don't overthink it.
            while param_updates<self.config["t_max"] + self.config["worker_parameter_sync_frequency"]:
                param_updates = ray.get(self.parameter_server.get_parameter_update_steps.remote())

                if param_updates % self.config["worker_parameter_sync_frequency"] == 0:
                    # These two can be the same function but I leave as is for now
                    self.sync_with_parameter_server()
                
                episode_batch = self.collect_experience()

                if self.config["save_obs_for_debug"]:
                    # Saves the observations of a single episode if you want to load it into a notebook and see what the agents see
                    np.save("epbatch_obs", episode_batch["obs"])
                    self.config["save_obs_for_debug"] = False

                self.remote_buffer.insert_episode_batch.remote(episode_batch)

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
    
        self.mac = CustomMAC(self.config, scheme, device = self.device)

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.config["batch_size_run"], self.config["episode_limit"]+1, preprocess = preprocess, device = "cpu")

    # def get_env_info(self):
    #     self.config["obs_shape"] = self.env.obs_shape
    #     self.env_info = self.env.get_init_env_info()
    #     self.config["n_actions"] = self.env_info["n_actions"]

    def setup_logger(self):
        self.log_dir = "results/" + self.config["name"] +"_" + datetime.datetime.now().strftime("%d_%m_%H_%M")

    def close_env(self):
        self.env.close()

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
        
        if self.config["use_burnin"]:
            hidden_states = {"hidden_state": {"vshape": (1, 2,self.config["rnn_hidden_dim"]), "dtype": torch.float32}}
            scheme.update(hidden_states)
        
        groups = {
        "agents": self.config["num_agents"]
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



