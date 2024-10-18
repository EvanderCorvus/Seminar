import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import cKDTree, distance
from numba import njit, jit
from utils.utils import time_it


# ToDo: See if i can just use the add method in rb and then the data doesn't need a batch dim
class ActiveBrownianEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.size = config["size"]
        self.dt = config["dt"]
        v_0 = config['peclet_number'] * config['rotational_diffusion'] * config['sigma'] / 3
        self.noise_std = np.sqrt(self.dt) * np.sqrt(config['rotational_diffusion'] * config['typical_length'] / v_0)

        self.current_step = None
        self.global_step = 0
        self.train_step = 0
        self.boundary_condition_type = config['boundary_condition_type']

        self.lower_bound = 0
        self.upper_bound = self.size

        self.observation_space = spaces.Box(
            self.lower_bound,
            self.upper_bound,
            shape=(1, 2),
            dtype=np.float64,
        )

        self.action_space = spaces.Box(
            -np.pi, np.pi,
            shape=(1, 1),
            dtype=np.float64,
        )
        self.states = None
        self.obs = None

    def reset(self, **kwargs):
        pass

    def step(self, actions, frame_skip):
        self.current_step += 1
        next_obs, e, nns, dist = self.active_brownian_rollout(actions, frame_skip)
        rewards = self._get_reward()

        return next_obs, rewards, e

    def active_brownian_rollout(self, actions, frame_skip):
        std = self.noise_std * np.ones((self.n_agents, 1), dtype=np.float64)
        e = np.zeros((self.n_agents, 2), dtype=np.float64)
        for _ in range(int(frame_skip)):
            # This performs a single time step of langevin dynamics
            thetas = np.random.normal(loc=actions, scale=std)
            # Calculate Forces
            forces = np.zeros((self.n_agents, 2), dtype=np.float64)

            # Update positions
            e = np.array([np.cos(thetas), np.sin(thetas)],
                         dtype=np.float64).reshape(self.n_agents, 2)

            v = e + forces
            next_states = self.states + v * self.dt

            if not self.state_space.contains(next_states):
                next_states = self._apply_boundary_conditions(
                    self.boundary_condition_type,
                    self.lower_bound,
                    self.upper_bound,
                    next_states
                )

            self.states = next_states
            actions = thetas


        self.obs = next_obs

        return next_obs, e


    def _get_reward(self):
        pass


    @staticmethod
    @njit
    def _apply_boundary_conditions(boundary_condition_type, lower_bound, upper_bound, states):
        if boundary_condition_type == 'hard':
            states = np.clip(states, lower_bound, upper_bound)
        elif boundary_condition_type == 'periodic':
            states = np.mod(states, upper_bound)
        # elif boundary_condition_type == 'reflective':
        #     mask_neg = states < lower_bound
        #     mask_pos = states > upper_bound
        #     states[mask_neg] = -states[mask_neg]
        #     states[mask_pos] = 2*upper_bound - states[mask_pos]

        return states
