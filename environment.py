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
        #self.size = config["size"]
        self.dt = config["dt"]
        
        self.noise_std = config["char_size"]*np.sqrt(self.dt)

        # Potential parameters
        self.U0 = config["u0"]


        self.current_step = None
        self.global_step = 0
        self.train_step = 0
        self.boundary_condition_type = config['boundary_condition_type']

        self.lower_bound = config["lower_bound"]
        self.upper_bound = config["upper_bound"]

        self.state_space = spaces.Box(
            self.lower_bound,
            self.upper_bound,
            shape=(1, 2),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            -np.pi, np.pi,
            shape=(1, 1),
            dtype=np.float32,
        )

        self.goal_space = spaces.Box(
            np.array([[0.45,-0.05]]),
            np.array([[0.55,0.05]]),
            shape=(1, 2),
            dtype=np.float32,
               )
        
        self.state = None
        self.obs = None

    def reset(self, **kwargs):
        self.state = np.array([[-0.5, 0.]], dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, actions, frame_skip):
        self.current_step += 1
        next_obs, e = self.active_brownian_rollout(actions, frame_skip)
        rewards = self._get_reward()
        done = np.ones((1,1),dtype=np.int32)*self.goal_space.contains(next_obs)

        return next_obs, rewards, e, done
    
    # def forces(self):
    #     # Calculate Forces: mexican hat potential
    #     r = distance.cdist(self.state, self.state, 'euclidean')
        

    #     return np.zeros((1, 2), dtype=np.float32)
        
    def forces(self):
        # Get current position
        pos = self.state
        
        # Calculate radial distance ρ from center
        rho = np.sqrt(np.sum(pos**2, axis=1))
        
        # Initialize force array
        force = np.zeros_like(pos)
     
        
        # Calculate forces only where ρ ≤ 1/2
        mask = rho <= 0.5
        
        if np.any(mask):
            # Avoid division by zero by adding small epsilon
            eps = 1e-10
            safe_rho = np.maximum(rho[mask], eps)
            
            # Calculate force magnitude
            force_magnitude = -32 * self.U0 * (safe_rho**2 - 0.25) * safe_rho
            
            # Safely normalize the position vectors
            norm_pos = pos[mask] / (safe_rho[:, np.newaxis])
            
            # Convert to vector force
            force[mask] = force_magnitude[:, np.newaxis] * norm_pos
            
            # Handle the case where rho is exactly zero
            zero_mask = rho[mask] < eps
            if np.any(zero_mask):
                # At the center, force points radially outward with magnitude based on the potential
                force[mask][zero_mask] = np.array([eps, 0]) * (-32 * self.U0 * 0.0625)  # 0.0625 = 1/16
        
        return force

    def active_brownian_rollout(self, actions, frame_skip):
        std = self.noise_std * np.ones((1, 1), dtype=np.float32)
        e = np.zeros((1, 2), dtype=np.float32)
        for _ in range(int(frame_skip)):
            # This performs a single time step of langevin dynamics
            thetas = np.random.normal(loc=actions, scale=std)
            # Calculate Forces
            forces = self.forces()

            # Update positions
            e = np.array([np.cos(thetas), np.sin(thetas)],
                         dtype=np.float32).reshape(1, 2)

            v = e + forces
            next_states = self.state + v * self.dt

            if not self.state_space.contains(next_states):
                next_states = self._apply_boundary_conditions(
                    self.boundary_condition_type,
                    self.lower_bound,
                    self.upper_bound,
                    next_states
                )

            self.state = next_states
            actions = thetas


        self.obs = next_states
        return next_states, e


    def _get_reward(self):
        
        reward = -np.ones((1, 1), dtype=np.float32)*self.dt
        if self.goal_space.contains(self.state):
            reward = np.ones((1, 1), dtype=np.float32)
        # check if the agent is at one border of the environment
        if np.any(self.state == self.lower_bound) or np.any(self.state == self.upper_bound):
            reward = -0.5*np.ones((1, 1), dtype=np.float32)

        return reward


    @staticmethod
    #s@njit
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
