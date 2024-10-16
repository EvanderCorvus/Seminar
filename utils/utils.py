import configparser
import numpy as np
import ast
from tensordict import TensorDict
import torch as tr
from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage, LazyMemmapStorage
import time

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



class MultiAgentReplayBuffer(ReplayBuffer):
    def __init__(self, size, batch_dim):
        self.batch_dim = batch_dim
        super().__init__(storage=LazyTensorStorage(size),
                         # batch_size=batch_size,
                         )

    def add_transition(self, obs, act, reward, next_obs):
        data = self._create_td_transition(obs, act, reward, next_obs)
        self.add(data)

    # ToDo: Check if this can be done with torch.view or other reshaping methods
    def sample_agent_batches(self, batch_size):
        data = self.sample(batch_size).float().to(device)
        # print(data)
        transitions = []
        for i in range(data['observation'].shape[1]):
            observation = data['observation'][:, i, :]
            action = data['action'][:, i, :]
            reward = data['reward'][:, i, :]
            next_observation = data['next_observation'][:, i, :]

            transitions.append((observation, action, reward, next_observation))

        return transitions

    def _create_td_transition(self, obs, act, reward, next_obs):
        transition = TensorDict({
            'observation': obs,
            'action': act,
            'reward': reward,
            'next_observation': next_obs
        },
            batch_size=self.batch_dim,  # Here Batch dim is n_agents
            # device=device,
        )
        return transition


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