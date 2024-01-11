import ray
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import traceback

@ray.remote(num_cpus=1, num_gpus=0.0001)
class ParameterServer(object):
    def __init__(self, config) -> None:
        # self.params = []
        self.params = {}
        self.encoder_params = {}
        self.ICM_encoder_params = {}
        self.target_network_update_tracker = []
        self.environment_steps = 0
        
        self.config = config
        self.define_worker_schedule_tracker()
        self.total_episodes = 0
        self.parameter_updates = 0

        self.cumulative_rewards = 0
        self.icm_reward = 0
        self.num_episodes_accumulated_over = 0

        self.episode_duration = 0

        self.ep_length = 0

        self.log_dir = "results/" + datetime.datetime.now().strftime("%d_%m_%H_%M")


        self.reset_accumulated_rewards()
        

    def define_param_list(self, parameter_server_list):
        # self.param_list contains only the names of the parameters
        try:
            self.parameter_server_list = parameter_server_list
            # self.param_list = param_list
        except Exception as e:
            print (f"In {__file__}: {e}")

    def define_param_list_encoder(self, parameter_server_list_encoder):
        # self.param_list contains only the names of the parameters
        try:
            self.parameter_server_list_encoder = parameter_server_list_encoder
            # self.param_list = param_list
        except Exception as e:
            print (f"In {__file__}: {e}")
    
    def define_param_list_ICM_encoder(self, parameter_server_list_ICM_encoder):
        # self.param_list contains only the names of the parameters
        try:
            self.parameter_server_list_ICM_encoder = parameter_server_list_ICM_encoder
            # self.param_list = param_list
        except Exception as e:
            print (f"In {__file__}: {e}")

    def track_target_network_updates(self):
        self.target_network_update_tracker.append(self.environment_steps)
        dir = self.log_dir + "/target_updates"
        np.save(dir, self.target_network_update_tracker)

    def define_worker_schedule_tracker(self):
        self.worker_schedule_tracker = {}
        for worker_id in range(self.config["num_executors"]):
            self.worker_schedule_tracker[f"worker_{worker_id}"] = 0

    def get_worker_steps_by_id(self, worker_id):
        try:
            return self.worker_schedule_tracker[f"worker_{worker_id}"]
        except Exception as e:
            traceback.print_exc()

    def get_worker_steps_dict(self):
        return self.worker_schedule_tracker
    
    def accumulate_worker_steps_by_id(self, worker_id, steps_to_accumulate):
        self.worker_schedule_tracker[f"worker_{worker_id}"]+=steps_to_accumulate




    def update_params(self, state_dicts_to_save):
        try:
            params = {param: state_dicts_to_save[param].cpu().numpy() for param in self.parameter_server_list}
            # params = {param: state_dicts_to_save[param] for param in self.parameter_server_list}

            for param_name, param_value in params.items():
                # Use ray.put directly for the value, no need for an intermediate dictionary
                self.params[param_name] = ray.put(param_value)

            self.parameter_updates += 1

        except Exception as e:
            print(f"In {__file__}: {e}")

    def update_encoder_params(self, state_dicts_to_save):
        
        try:
            params = {param: state_dicts_to_save[param].cpu().numpy() for param in self.parameter_server_list_encoder}
            # params = {param: state_dicts_to_save[param] for param in self.parameter_server_list_encoder}

            for param_name, param_value in params.items():
                # Use ray.put directly for the value, no need for an intermediate dictionary
                self.encoder_params[param_name] = ray.put(param_value)


        except Exception as e:
            print(f"In {__file__}: {e}")

    def update_ICM_encoder_params(self, state_dicts_to_save):
        
        try:
            params = {param: state_dicts_to_save[param].cpu().numpy() for param in self.parameter_server_list_ICM_encoder}

            for param_name, param_value in params.items():
                # Use ray.put directly for the value, no need for an intermediate dictionary
                self.encoder_params[param_name] = ray.put(param_value)


        except Exception as e:
            print(f"In {__file__}: {e}")

    def return_params(self):
        return self.params
    
    def return_encoder_params(self):
        return self.encoder_params
    
    def return_ICM_encoder_params(self):
        return self.ICM_encoder_params
    
    def get_parameter_update_steps(self):
        return self.parameter_updates
    
    def add_environment_steps(self, num_steps_to_add):
        self.environment_steps+=num_steps_to_add

    def return_environment_steps(self):
        return self.environment_steps
    
    def increment_total_episode_count(self):
        self.total_episodes+=1

    def return_total_episode_count(self):
        return self.total_episodes
    
    def accumulate_stats(self, reward, episode_time, ep_length, icm_reward = None):
        self.cumulative_rewards+=reward
        self.num_episodes_accumulated_over+=1
        self.episode_duration += episode_time
        self.ep_length+=ep_length

        # self.L_I+=L_I
        # self.L_F+=L_F
        # self.grad_norm = grad_norm
        
        if icm_reward is not None:
            self.icm_reward += icm_reward


    def reset_accumulated_rewards(self):
        self.cumulative_rewards = 0
        self.num_episodes_accumulated_over = 0
        self.episode_duration = 0
        self.ep_length = 0
        # self.L_I=0
        # self.L_F=0
        # self.grad_norm = 0

        if self.icm_reward is not None:
            self.icm_reward = 0

    def get_accumulated_stats(self):
        mean_reward = self.cumulative_rewards/self.num_episodes_accumulated_over
        mean_episode_duration = self.episode_duration/self.num_episodes_accumulated_over
        mean_icm_reward = self.icm_reward/self.num_episodes_accumulated_over
        mean_episode_length = self.ep_length/self.num_episodes_accumulated_over
        # mean_lf = self.L_F/self.num_episodes_accumulated_over
        # mean_li = self.L_I/(self.ep_length)
        # mean_grad = self.grad_norm/self.num_episodes_accumulated_over
        mean_total_ep_reward = (self.cumulative_rewards+self.icm_reward)/self.num_episodes_accumulated_over

        self.reset_accumulated_rewards()
        
        return mean_reward,mean_icm_reward, mean_episode_duration, mean_episode_length, mean_total_ep_reward



