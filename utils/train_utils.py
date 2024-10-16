import numpy as np
from tqdm import tqdm
from utils.utils import time_it
from numba import njit, jit


def train_episode(
        env,
        agents,
        replay_buffer,
        current_episode,
        config,
        writer=None
    ):
    observation = env.reset()
    log_states = [env.states]

    for step in range(config['n_steps']):
        # Perform Actions
        actions = np.array([agents[f'agent_{i}'].act(observation[i]) for i in range(env.n_agents)])

        # Environment Steps
        next_obs, rewards, velocities = env.step(actions, config['frame_skip'])

        replay_buffer.add_transition(
            observation,
            actions,
            rewards,
            next_obs
        )
        observation = next_obs
        log_states.append(env.states)
        env.global_step += 1
        if writer is not None:
            writer.add_scalar(
                'Reward (Group Average)',
                rewards.mean(),
                config['n_steps'] * current_episode + step
            )
        if step >= config['n_optim']:
            # Update Agents
            loss_actors, loss_critics, entropies = [], [], []
            for _ in range(config['n_optim']):
                sample = replay_buffer.sample_agent_batches(
                    min(config['batch_size'], len(replay_buffer))
                )
                for i in range(env.n_agents):
                    loss_actor, loss_critic, entropy = agents[f'agent_{i}'].update(sample[i])
                    loss_actors.append(loss_actor)
                    loss_critics.append(loss_critic)
                    entropies.append(entropy)
                if writer is not None:
                    writer.add_scalar(
                        'Loss/Actors',
                        np.mean(loss_actors),
                        env.train_step
                    )
                    writer.add_scalar(
                        'Loss/Critics',
                        np.mean(loss_critics),
                        env.train_step
                    )
                    writer.add_scalar(
                        'Entropy Regularization (Group Average)',
                        np.mean(entropies),
                        env.train_step
                    )
                env.train_step += 1
    return log_states


def test_episode(
        env,
        agents,
        config,
        writer=None
):
    observations = env.reset()
    list_states = [env.states]
    for step in tqdm(range(config['test_steps']), leave=False):
        # actions = np.zeros((env.n_agents, 1), dtype=np.float64)
        actions = np.array([agents[f'agent_{i}'].act(observations[i]) for i in range(env.n_agents)])
        next_obs, e, _, _ = env.active_brownian_rollout(actions, config['frame_skip'])
        observations = next_obs
        list_states.append(env.states)
        if writer is not None:
            writer.add_scalar(
                'Test/Polar Order',
                group_polar_order(e),
                step,
            )
            writer.add_scalar(
                'Test/Angular Momentum',
                group_angular_momentum(e, env.states),
                step,
            )
            writer.add_scalar(
                'Test/Group Average MSD',
                group_average_MSD(list_states),
                step,
            )
        else:
            print('No writer')

    return list_states


def group_polar_order(velocities):
    return np.linalg.norm(np.mean(velocities, axis=0))


def group_angular_momentum(velocities, positions):
    mean_pos = np.mean(positions, axis=0)
    relative_positions = positions - mean_pos
    # 2D cross product
    angular_momentum = relative_positions[:, 0] * velocities[:, 1] - relative_positions[:, 1] * velocities[:, 0]
    return np.linalg.norm(np.mean(angular_momentum, axis=0))


# State Shape: (time_step, n_agents, 2)

def group_average_MSD(states):
    states_array = np.array(states)
    initial_positions = states_array[0]

    squared_displacement = np.linalg.norm(
        (states_array[1:] - initial_positions),
        axis=-1
    ) ** 2
    return np.mean(squared_displacement)