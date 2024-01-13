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
        self.total_steps_all_workers = 0
        
        self.config = config
        self.define_worker_schedule_tracker()
        
        self.total_episodes = 0
        self.parameter_updates = 0

        self.cumulative_rewards = 0

        # Items from the get_stats method from the starcraft env
        self.avg_win_rate = 0
        self.avg_battles_won = 0
        self.avg_battles_game = 0
        self.avg_battles_draw = 0
        self.avg_timeouts = 0
        self.avg_restarts = 0
        
        # Tracking test stats
        self.can_log_test = True
        self.test_cumulative_rewards = 0
        self.test_num_episodes_accumulated_over = 1
        self.test_ep_length = 0
        self.test_avg_win_rate = 0
        self.test_avg_battles_won = 0
        self.test_avg_battles_game = 0
        self.test_avg_battles_draw = 0
        self.test_avg_timeouts = 0
        self.test_avg_restarts = 0
        self.test_total_t = 0


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
        pass
        # self.target_network_update_tracker.append(self.environment_steps)
        # dir = self.log_dir + "/target_updates"
        # np.save(dir, self.target_network_update_tracker)

    def define_worker_schedule_tracker(self):
        self.worker_schedule_tracker = {}
        for worker_id in range(self.config["num_executors"]):
            self.worker_schedule_tracker[f"worker_{worker_id}"] = 0

    # def define_worker_test_tracker(self):
    #     """
    #     Tracks whether a worker has completed a set of testing episodes, and if all workers have completed a set,
    #     the learner will log the stats against the mean of the steps where the workers started tracking.
    #     """
    #     self.worker_test_tracker = {}
    #     for worker_id in range(self.config["num_executors"]):
    #         self.worker_test_tracker[f"worker_{worker_id}"] = False

    # def get_worker_test_tracker_dict(self):
    #     return self.worker_test_tracker
    
    # def set_worker_test_tracker_dict(self, worker_id, state):
    #     self.worker_test_tracker[f"worker_{worker_id}"] = state
    
    # def set_worker_test_tracker_dict_false(self):
    #     for key in self.worker_test_tracker:
    #         self.worker_test_tracker[key] = False
    #     # self.worker_test_tracker[f"worker_{worker_id}"] = False
    
        

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
        self.total_steps_all_workers+=num_steps_to_add

    def return_total_steps_all_workers(self):
        return self.total_steps_all_workers
    
    def increment_total_episode_count(self):
        self.total_episodes+=1

    def return_total_episode_count(self):
        return self.total_episodes
    
    def accumulate_stats(self, reward, episode_time, ep_length, get_stats_dict):
        self.cumulative_rewards+=reward
        self.num_episodes_accumulated_over+=1
        self.episode_duration += episode_time
        self.ep_length+=ep_length

        self.avg_win_rate += get_stats_dict["win_rate"]
        self.avg_battles_won += get_stats_dict["battles_won"]
        self.avg_battles_game += get_stats_dict["battles_game"]
        self.avg_battles_draw += get_stats_dict["battles_draw"]
        self.avg_timeouts += get_stats_dict["timeouts"]
        self.avg_restarts += get_stats_dict["restarts"]
        

    def reset_accumulated_rewards(self):
        self.cumulative_rewards = 0
        self.num_episodes_accumulated_over = 0
        self.episode_duration = 0
        self.ep_length = 0


    def get_accumulated_stats(self):
        mean_reward = self.cumulative_rewards/self.num_episodes_accumulated_over
        mean_episode_length = self.ep_length/self.num_episodes_accumulated_over
        

        acc_stats_dict = {
            "mean_train_reward": mean_reward,
            "mean_train_episode_length": mean_episode_length,
        }

        self.reset_accumulated_rewards()
        return acc_stats_dict


    def log_test_stats(self, reward, ep_length, win_rate, total_t, num_eps):
        self.test_cumulative_rewards=reward
        self.test_num_episodes_accumulated_over = num_eps
        self.test_ep_length=ep_length

        self.test_avg_win_rate = win_rate
        self.test_total_t = total_t
        # self.test_avg_battles_won += get_stats_dict["battles_won"]
        # self.test_avg_battles_game += get_stats_dict["battles_game"]
        # self.test_avg_battles_draw += get_stats_dict["battles_draw"]
        # self.test_avg_timeouts += get_stats_dict["timeouts"]
        # self.test_avg_restarts += get_stats_dict["restarts"]

    def get_accumulated_test_stats(self):
        mean_reward = self.test_cumulative_rewards/self.test_num_episodes_accumulated_over
        mean_episode_length = self.test_ep_length/self.test_num_episodes_accumulated_over
        win_rate = self.test_avg_win_rate

        acc_stats_dict = {
            "test_mean_reward": mean_reward,
            "test_mean_episode_length": mean_episode_length,
            "test_avg_win_rate": win_rate,
        }

        self.reset_test_stats()
        return acc_stats_dict, self.test_total_t
        

    def reset_test_stats(self):
        self.test_avg_win_rate = 0
        self.test_avg_battles_won = 0
        self.test_avg_battles_game = 0
        self.test_avg_battles_draw = 0
        self.test_avg_timeouts = 0
        self.test_avg_restarts = 0
        self.test_cumulative_rewards = 0
        self.test_num_episodes_accumulated_over = 1
        self.test_ep_length = 0

    def get_can_log_test(self):
        return self.can_log_test
    
    def set_can_log_test(self, state):
        self.can_log_test = state






