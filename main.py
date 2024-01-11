import ray
from components.replay_buffer import Remote_ReplayBuffer, generate_replay_scheme, CustomPrioritized_ReplayBuffer
from components.parameter_server import ParameterServer
from utils.read_config import merge_yaml_files
from components.executor import Executor
from components.testing_executor import TestExecutor
from components.learner import Learner
import yaml

import sys

ray.init()

def main(args):
    """
    args:
        0: main.py
        1: test/train
        2: config file name
    """

    default_cfg = "./config/default.yaml"

    file2 = f"./config/{args[2]}.yaml"
    
    config = merge_yaml_files(default_cfg, file2)
    
    if args[1] == "train":
        workers = [Executor.remote(config, i) for i in range (config["num_executors"])]
        config_ref = workers[0].retrieve_updated_config.remote()
        config = ray.get(config_ref)


        scheme, groups, preprocess = generate_replay_scheme(config)

        if config["use_per"]:
            remote_buffer = CustomPrioritized_ReplayBuffer.remote(scheme, groups, config["buffer_size"], config["episode_limit"]+1, preprocess=preprocess, device="cpu", alpha = config["prioritized_buffer_alpha"])
        else:
            remote_buffer = Remote_ReplayBuffer.remote(scheme, groups, config["buffer_size"], config["episode_limit"]+1, preprocess=preprocess, device="cpu")
        parameter_server = ParameterServer.remote(config)

        learner = Learner.remote(config)

        # set remote objects
        for worker in workers:
            worker.set_remote_objects.remote(remote_buffer, parameter_server)

        learner.set_remote_objects.remote(remote_buffer, parameter_server)

        param_list = ray.get(learner.return_parameter_list.remote())

        ray.get(parameter_server.define_param_list.remote(param_list))
        ray.get(learner.update_parameter_server.remote())

        all_actors = workers + [learner]

        ray.wait([worker.run.remote() for worker in all_actors])       

        # For debugging purposes:
        # ray.timeline(filename="timeline.json")
        
        sys.exit(1)
    elif args[1] == "test":
        worker = TestExecutor(config, 0)

        worker.run()
        sys.exit(1)
    else:
        print ("No argument specified, exiting")
        sys.exit(1)





if __name__ == "__main__":
    args = sys.argv
    main(args)

