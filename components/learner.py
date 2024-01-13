import torch
import torch.nn as nn 
from models.qmix import QMixer
from utils.utils import RunningMeanStdTorch
from torch.optim import Adam
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import numpy as np
import ray
from controllers.custom_controller import CustomMAC
import time
from components.replay_buffer import EpisodeBatch
import traceback

import os
import yaml
from utils.utils import signed_hyperbolic, signed_parabolic
from models.ICMModel_2 import ICMModel

@ray.remote(num_gpus = 0.99, num_cpus=1, max_restarts=10)
class Learner(object):
    def __init__(self, config):
        # super().__init__()
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.global_training_steps = config["t_max"]

        
        self.mac = CustomMAC(self.config)

        if self.config["use_transfer"]:
            self.mac.load_models(self.config["models_2_transfer_path"])

    
        # Create parameterlist to train:
        self.trainable_parameters = nn.ParameterList(self.mac.agent.parameters())
            
        # Parameter server lists - stores the names of the parameters that should be stored in the parameter server
        self.parameter_server_list = list(self.mac.agent.state_dict())

        # Setup Mixer
        self.mixer = QMixer(config)

        if self.config["use_transfer"]:
            self.mixer.load_state_dict(torch.load(self.config["models_2_transfer_path"] + "/mixer.th"))

        self.trainable_parameters+= nn.ParameterList(self.mixer.parameters())
        

        # Reward Standardisation
        if self.config["standardise_rewards"]:
            self.reward_rms = RunningMeanStdTorch(shape=(1,), device=self.device)

        # Optimiser
        self.optimiser = Adam(params=self.trainable_parameters, lr=self.config["lr"], eps = self.config["optim_eps"])

        self.log_base_dir = "results/" + self.config["name"] +"_" + datetime.datetime.now().strftime("%d_%m_%H_%M")


        # Target Networks:
        self.target_mac = CustomMAC(self.config)
        self.target_mixer = deepcopy(self.mixer)

        # Initialise timing variables
        self.previous_target_update_episode = 0
        self.trainer_steps = 0
        self.training_start_time = time.time()

        # Cuda learner if available
        if torch.cuda.is_available():
            self.cuda() 

        # Create TB logger        
        self.setup_writer()
        self.time_info = {"number_log_steps": 0}
        self.log_stats_dict = {}
        
        # Create episolon decay schedule
        if self.config["action_selector"] == "epsilon_greedy":
            self.delta = (self.config["epsilon_start"]-self.config["epsilon_finish"])/self.config["epsilon_anneal_time"]

        # Save config to results for recreating later
        file = open(self.log_base_dir + "/config.yaml", "w")
        yaml.dump(self.config, file)
        file.close()
  
    
    def train(self, log_this_step = False):
        log_dict = {}

        try:
            if self.config["use_per"]:
                try:
                    sample_indices, episode_sample_reference = ray.get(self.remote_buffer.sample.remote(self.config["batch_size"]))
                except Exception as e:
                    traceback.print_exc()

            else:
                try:
                    episode_sample_reference = ray.get(self.remote_buffer.sample.remote(self.config["batch_size"]))
                except Exception as e:
                    traceback.print_exc()
            try:
                max_ep_t_reference = episode_sample_reference.max_t_filled()
                max_ep_t = max_ep_t_reference
            except Exception as e:
                traceback.print_exc()

            if not self.config["random_update"]:
                # Select entirety of episode, i.e all steps, from replay - bootstrapped sequential updates 
                episode_sample_reference = episode_sample_reference[:, :max_ep_t]
                
            else:
                # Slice steps from episode, i.e bootstrapped random updates - Ideally, we want to change the replay buffer the negate the need for slicing,
                # since slicing from the replay reduces the effectiveness of PER. However, I did not get around to that so I will leave it as is.
                try:
                    # Include additional steps for burning in the RNN layer
                    if self.config["use_burnin"]: 
                        recurrent_sequence_length = self.config["recurrent_sequence_length"] + self.config["burn_in_step_count"]
                    else:
                        recurrent_sequence_length = self.config["recurrent_sequence_length"]

                    if max_ep_t>recurrent_sequence_length:
                        # Start from a random index (this is what reduces PER effectiveness. The high-value transitions could be omitted in training due to
                        # the below code)
                        start_idx = np.random.randint(0, max_ep_t-recurrent_sequence_length)
                        episode_sample_reference = episode_sample_reference[:, start_idx[0]:start_idx[0]+recurrent_sequence_length]

                    else:
                        episode_sample_reference = episode_sample_reference[:, :max_ep_t]

                except Exception as e:
                    traceback.print_exc()

            

            if self.config["use_per"]:
                try:
                    global_eps = ray.get(self.parameter_server.return_total_episode_count.remote())
                    td_error, log_dict = self.training_loop(episode_sample_reference,global_eps, log_this_step)
                    return log_dict, td_error, sample_indices
                except Exception as e:
                    traceback.print_exc()
                
            else:
                global_eps = ray.get(self.parameter_server.return_total_episode_count.remote())
                log_dict = self.training_loop(episode_sample_reference,global_eps, log_this_step)
                return log_dict
                
        except Exception as e:
            traceback.print_exc()

    def training_loop(self, batch, num_global_episodes, log_this_step):

        try:
            
            if self.config["use_burnin"]:
                do_burn_in = False
                # Get the hidden state from the replay buffer to start burn-in
                init_hidden_state = batch["hidden_state"][0,0]

                # If the episode is long enough, slice the burn-in steps from the episode:
                if batch["obs"].shape[1] > 2*self.config["burn_in_step_count"]:
                    do_burn_in = True
                    burnin_batch = batch[:, :self.config["burn_in_step_count"]]
                    batch = batch[:, self.config["burn_in_step_count"]:]

            else:
                init_hidden_state = None

            rewards = batch["reward"][:, :-1].to('cuda:0', non_blocking=True)
            actions = batch["actions"][:, :-1].to('cuda:0', non_blocking=True, dtype = torch.int64)
            actions_onehot = batch["actions_onehot"][:,:-1].to('cuda:0', non_blocking=True)
            terminated = batch["terminated"][:, :-1].float().to('cuda:0', non_blocking=True)
            mask = batch["filled"][:, :-1].float().to('cuda:0', non_blocking=True)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]).to('cuda:0', non_blocking=True)
            avail_actions = batch["avail_actions"].to('cuda:0', non_blocking=True)
            batch_state = batch["state"].to('cuda:0', non_blocking=True)

            B = batch["obs"].shape[0]
            T = batch["obs"].shape[1]
            N = batch["obs"].shape[2]

            
            mac_out = []
            self.mac.init_hidden(batch.batch_size, hidden_state=init_hidden_state)

            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size, hidden_state=init_hidden_state)

            # Burn in the rnn of both the online and target networks, if there are enough steps in the episode
            if self.config["use_burnin"] and do_burn_in:
                with torch.no_grad():
                    for t in range(burnin_batch["obs"].shape[1]):
                        _, _ = self.mac.forward(burnin_batch, t=t)
                        _, _ = self.target_mac.forward(burnin_batch, t=t)

            # Calculate q-values
            for t in range(T):
                agent_outs, _ = self.mac.forward(batch, t=t)
                target_agent_outs, _ = self.target_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
                target_mac_out.append(target_agent_outs) 

            mac_out = torch.stack(mac_out, dim=1)  # Concat over time
        

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            # We don't need the first timestep's Q-Value estimate for calculating targets
            target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -999999


            # Max over target Q-Values
            if self.config["double_q"]:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]




            if self.config["standardise_rewards"]:
                self.reward_rms.update(rewards)
                rewards_normed = (rewards-self.reward_rms.mean)/torch.sqrt(self.reward_rms.var)
            else:
                rewards_normed = rewards

            # Not necessary for any of the environments, but do it anyways
            torch.clamp_(rewards_normed, max = self.config["reward_clip_max"], min=self.config["reward_clip_min"])

            rewards_total = rewards_normed

            # If extra state information is available (used by the mixer)
            state = torch.cat([avail_actions.reshape([B,T,-1]), batch_state], dim=-1).to(self.device)

            chosen_action_qvals = self.mixer(chosen_action_qvals.to(torch.float32), state[:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals.to(torch.float32), state[:, 1:])

            # do scaling on target_max_qvals here
            if self.config["value_fn_rescaling"]:
                target_max_qvals = signed_parabolic(target_max_qvals)
            
    
            # Calculate Q-Learning targets
            if self.config["n_step_return"]:
                targets = self.n_step_targets(rewards_total, terminated, target_max_qvals)
            else:
                targets = rewards_total + self.config["gamma"] * (1 - terminated) * target_max_qvals

            # apply inverse scaling to targets here now
            if self.config["value_fn_rescaling"]:
                    targets = signed_hyperbolic(targets)
            

            # Td-error
            # TD error as done in the pymarl implementations
            td_error = (chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()
        
            # Optimise
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.config["grad_norm_clip"])
            self.optimiser.step()

            if self.config["action_selector"] == "noisy":
                self.mac.agent.sample_noise()
                self.target_mac.agent.sample_noise()

            # Update target networks:
            if (self.trainer_steps - self.previous_target_update_episode) / (self.config["target_update_interval"]) >= 1.0:
                try:
                    print(f"Updating target networks at global episode {num_global_episodes}, total steps: {ray.get(self.parameter_server.return_total_steps_all_workers.remote())} ")
                    self._update_targets()
                    self.parameter_server.track_target_network_updates.remote()
                    self.previous_target_update_episode = self.trainer_steps
                except Exception as e:
                    traceback.print_exc()

            if self.trainer_steps%self.config["save_models_interval"] == 0 and self.config["save_models"] and self.trainer_steps>1:
                self.save_models()

            losses_dict = {}
            if log_this_step:
                losses_dict = {"total_loss" : loss.item()}
                
                mask_elems = mask.sum().item()
                other_log_stuff = {
                    "td_error_abs": (masked_td_error.abs().sum().item()/mask_elems),
                    "q_taken_mean": (chosen_action_qvals * mask).sum().item()/(mask_elems * self.config["n_agents"]),
                    "target_mean": (targets * mask).sum().item()/(mask_elems * self.config["n_agents"]),
                    "learner_total_return_mean": rewards_total.sum().item()/B
                }
                losses_dict.update(other_log_stuff)


            if self.config["use_per"]:
                td_er = self.calculate_r2d2_like_priorities(td_error)
                return td_er, losses_dict

            else:
                return losses_dict
                   
            
        except Exception as e:
            traceback.print_exc()

    def run(self):
        try:
            while not ray.get(self.remote_buffer.can_sample.remote(self.config["batch_size"])):
                time.sleep(1)
                continue

            print("READY TO START TRAINING!!")

            while self.trainer_steps < self.config["t_max"]+5:
                # Log time taken per training step:
                current_training_step_start_time = time.time()
                
                if self.trainer_steps%self.config["log_every"] == 0:
                    print(f"Logging at training step {self.trainer_steps}")
                    log_this_step = True
                else:
                    log_this_step = False

                if self.config["use_per"]:
                    log_dict, td_error, sample_indices = self.train(log_this_step)
                    self.remote_buffer.update_priorities.remote(sample_indices, td_error)
                else:
                    log_dict = self.train(log_this_step)
    
                self.trainer_steps += 1

                self.update_parameter_server()

                self.cuda()


                training_took = time.time() - current_training_step_start_time
                self.store_time_stats("Mean_training_loop_time", training_took)

                if log_this_step:
                    self.log_stats(log_dict)
            sys.exit(1)

        except Exception as e:
            traceback.print_exc()
            

        
    def setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_base_dir + "/tb_logs")

    def save_models(self):
        path = self.log_base_dir + f"/models/{self.trainer_steps}"

        os.makedirs(path, exist_ok=True)

        self.mac.save_models(path)

        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def log_stats(self, stats_to_log:dict):
        # rather than global env steps, use trainer steps as a more consistent measure of performance over time.
        global_environment_steps_for_logging = ray.get(self.parameter_server.return_total_steps_all_workers.remote())
        trainer_steps = self.trainer_steps

        # Stats obtained from the trainer:
        for key, value in stats_to_log.items():
            self.writer.add_scalar(f"Training_Stats/{key}", value, self.trainer_steps)

        # Trainer time info
        self.writer.add_scalar("Time_Stats/mean_training_step_time", self.time_info["Mean_training_loop_time"]/self.time_info["number_log_steps"], trainer_steps)
        self.writer.add_scalar("Time_Stats/environment_steps_taken", global_environment_steps_for_logging, trainer_steps)

        # Log train stats
        # acc_stats_dict= ray.get(self.parameter_server.get_accumulated_stats.remote())

        # for key, value in acc_stats_dict.items():
        #     self.writer.add_scalar(f"Reward_Stats/{key}", value, trainer_steps)

        # Log test results:
        # First check if all the executors have completed a set of testing episodes
        can_log = ray.get(self.parameter_server.get_can_log_test.remote())
        
        if can_log:
            test_acc_stats_dict, total_t = ray.get(self.parameter_server.get_accumulated_test_stats.remote())
            
            for key, value in test_acc_stats_dict.items():
                self.writer.add_scalar(f"Test_stats/{key}", value, total_t)

            self.parameter_server.set_can_log_test.remote(False)              
            
        # self.writer.add_scalar("Reward_Stats/mean_total_reward", acc_stats_dict["mean_total_reward"], trainer_steps)

        # self.writer.add_scalar("Reward_Stats/mean_extrinsic_reward", acc_stats_dict["mean_extrinsic_reward"], trainer_steps)

        # self.writer.add_scalar("Reward_Stats/mean_ep_length", ["mean_ep_length"], trainer_steps)

        # self.writer.add_scalar("Reward_Stats/average_train_win_rate", acc_stats_dict["avg_win_rate"], trainer_steps)
        # self.writer.add_scalar("Reward_Stats/average_train_win_rate_vs_global_steps", acc_stats_dict["avg_win_rate"], global_environment_steps_for_logging)

        # self.writer.add_scalar("Time_Stats/mean_episode_duration", acc_stats_dict["mean_episode_duration"], trainer_steps)

        # log worker steps
        worker_steps_dict = ray.get(self.parameter_server.get_worker_steps_dict.remote())
        for key, value in worker_steps_dict.items():
            self.writer.add_scalar(f"{key}/total_worker_steps", value, value)
            if self.config["action_selector"] == "epsilon_greedy":
                self.writer.add_scalar(f"{key}/epsilon", self.calculate_epsilon(value), value)

        global_eps = ray.get(self.parameter_server.return_total_episode_count.remote())
        self.writer.add_scalar("Time_Stats/total_num_episodes", global_eps, trainer_steps)

        
    def calculate_epsilon(self, steps_value):
        return max(self.config["epsilon_finish"], 1-(self.delta*steps_value))


    def store_time_stats(self, key, value):
        if key not in self.time_info:
            self.time_info[key] = 0
            self.time_info[key] += value
            self.time_info["number_log_steps"] +=1
        else:
            self.time_info[key] += value
            self.time_info["number_log_steps"] +=1
    

    def n_step_targets(self, rewards, terminated, target_max_qvals):
        """
        Code modified from https://github.com/michaelnny/deep_rl_zoo/blob/1b5450eb403976be223daf80e29461bedd5d835b/deep_rl_zoo/multistep.py#L28
        """
        bellman_target = torch.cat(
            [torch.zeros_like(target_max_qvals[:,0:1]), target_max_qvals]+[target_max_qvals[:,-1:]/self.config["gamma"]**k for k in range(1, self.config["n_step"])], dim = 1
            )
        
        terminated_n = torch.cat([terminated] + [torch.zeros_like(terminated[:,0:1])] * self.config["n_step"], dim = 1)
        rewards_n = torch.cat([rewards] + [torch.zeros_like(rewards[:, 0:1])] * self.config["n_step"], dim = 1)

        # let us lolol the n_step targets now
        for _ in range(self.config["n_step"]):
            rewards_n = rewards_n[:, :-1]
            terminated_n = terminated_n[:,:-1]
            bellman_target = rewards_n + self.config["gamma"]*(1.0-terminated_n.float())*bellman_target[:, 1:]

        return bellman_target


    def set_remote_objects(self, remote_buffer, parameter_server):
        self.remote_buffer = remote_buffer
        self.parameter_server = parameter_server

        self.parameter_server.define_param_list.remote(self.parameter_server_list)



    def update_parameter_server(self):
        params = {}
        state_dicts_to_save = self.mac.agent.state_dict()

        # The below two could be merged into a single method but meh, makes very little difference
        self.parameter_server.update_params.remote(state_dicts_to_save)

    def return_parameter_list(self):
        return self.parameter_server_list

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def calculate_r2d2_like_priorities(self, td_errors):
        """
        Shamelessly stolen from https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/distributed.py#L31
        """
        td_errors = torch.clone(td_errors).detach()
        abs_td_errors = torch.abs(td_errors)
        priorities = self.config["per_eta"] * torch.max(abs_td_errors, dim=1)[0] + (1 - self.config["per_eta"]) * torch.mean(abs_td_errors, dim=1)
        priorities = torch.clamp(priorities, min=0.0001, max=1000)  # Avoid NaNs
        priorities = priorities.cpu().numpy()


        return priorities
