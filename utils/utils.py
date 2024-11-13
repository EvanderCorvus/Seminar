import configparser
import numpy as np
import ast
from tensordict import TensorDict
import torch as tr
from torchrl.data import ReplayBuffer, LazyTensorStorage
import time
from torchrl.data.replay_buffers.samplers import PrioritizedSampler



if tr.backends.mps.is_available():
    device = tr.device("mps")
elif tr.cuda.is_available():
    device = tr.device("cuda")
else:
    device = tr.device("cpu")

print(f"Using device: {device}")


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
    def __init__(self, size, alpha=0.6, beta=0.4):
        super().__init__(
            storage=LazyTensorStorage(int(size)),
            sampler=None  # We'll implement our own sampling
        )
        self.size = int(size)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(self.size, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add_transition(self, obs, act, reward, next_obs, done):
        data = TensorDict({
            'obs': obs,
            'action': act,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        }, batch_size=1)
        
        self.extend(data)
        
        # Set priority for new sample
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None

        # Calculate sampling probabilities
        probs = self.priorities[:len(self)] ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (len(self) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = tr.from_numpy(weights.astype(np.float32))

        # Get samples
        samples = self[indices]

        return samples, indices, weights

    def update_priority(self, indices, priorities):
        # Konvertiere priorities zu numpy array falls es ein Tensor ist
        if tr.is_tensor(priorities):
            priorities = priorities.detach().cpu().numpy()
        
        # Stelle sicher, dass priorities ein numpy array ist
        priorities = np.asarray(priorities)
        
        # Update priorities für die gegebenen indices
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
        
        # Update max_priority
        self.max_priority = float(max(self.max_priority, np.max(priorities)))


    def extend(self, tensordicts):
        super().extend(tensordicts)
        new_samples = len(tensordicts)
        self.priorities[self.pos:self.pos+new_samples] = self.max_priority
        self.pos = (self.pos + new_samples) % self.size



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