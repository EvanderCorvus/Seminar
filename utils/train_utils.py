import numpy as np
from tqdm import tqdm
from utils.utils import time_it
from numba import njit, jit


def train_episode(
        env,
        agent,
        replay_buffer,
        current_episode,
        config,
        writer=None
    ):
    observation = env.reset()
    log_states = [env.state]
    reward_total = 0
    for step in range(config['n_steps']):
        # Perform Actions
        actions = np.array(agent.act(observation))

        # Environment Steps
        next_obs, rewards, e, done = env.step(actions, config['frame_skip'])
        replay_buffer.add_transition(
            observation,
            actions,
            rewards,
            next_obs,
            done
        )
        reward_total += rewards
        observation = next_obs
        log_states.append(env.state)
        env.global_step += 1
        if writer is not None:
            writer.add_scalar(
                'Reward (Group Average)',
                rewards.mean(),
                config['n_steps'] * current_episode + step
            )
        if step >= config['batch_size']:
            # Update Agents
            loss_actors, loss_critics, entropies = [], [], []
            for _ in range(config['n_optim']):
                #print(replay_buffer.sample(config['batch_size']))
                sample, info, weights = replay_buffer.sample(config['batch_size'])
                loss_actor, loss_critic, entropy, td_error = agent.update(sample)

                replay_buffer.update_priority(
                    info,
                    td_error
                )
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
        if done.any(): break
    return log_states, reward_total


def test_episode(
        env,
        agents,
        config,
        writer=None
):
    observations = env.reset()
    list_states = [env.state]
    for step in tqdm(range(config['test_steps']), leave=False):
        # actions = np.zeros((env.n_agents, 1), dtype=np.float64)
        actions = agents.act(observations, deterministic=True)
        next_obs, e  = env.active_brownian_rollout(actions, config['frame_skip'])
        observations = next_obs
        list_states.append(env.state)
        if env.goal_space.contains(env.state): break

    return list_states

