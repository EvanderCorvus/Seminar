import configparser
import numpy as np
import ast
from tensordict import TensorDict
import torch as tr
from torchrl.data import ReplayBuffer, LazyTensorStorage
import time
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')


def hyperparams_dict(section, path):
    config = configparser.ConfigParser()
    config.read(path)
    if not config.read(path):
        raise Exception("Could not read config file")

    params = config[section]
    typed_params = {}
    for key, value in params.items():
        try:
            # Attempt to evaluate the value to infer type
            typed_params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback to the original string value if evaluation fails
            typed_params[key] = value

    return typed_params



class SingleAgentPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, beta):
        super().__init__(
            storage=LazyTensorStorage(int(size)),
            sampler=PrioritizedSampler(
                max_capacity=int(size),
                alpha=alpha,
                beta=beta)
            )

    def add_transition(self, obs, act, reward, next_obs):
        data = TensorDict({
            'obs': obs,
            'action': act,
            'reward': reward,
            'next_obs': next_obs
            },
            batch_size=1,
        )
        self.extend(data)


def time_it(once=True):
    def decorator(func):
        stop = False

        def wrapper(*args, **kwargs):
            nonlocal stop
            if not stop:
                start_time = time.time()  # Record start time
                result = func(*args, **kwargs)
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time
                print(f"{func.__name__} took {elapsed_time:.5f} seconds to run.")

                if once: stop = True
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator