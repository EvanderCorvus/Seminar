from utils.utils import *
from environment import ActiveBrownianEnv
from utils.train_utils import train_episode
# from agents import SAC_Agent
from models import SoftActorCritic
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import os
import dotenv
from datetime import datetime


# from torch.profiler import profile, record_function, ProfilerActivity

@time_it()
def train():
    # Data
    print('Testing at', datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    dotenv.load_dotenv()
    experiments_dir = os.getenv('experiment_dir')
    hparam_path = os.getenv('hparam_path')
    writer_flag = os.getenv('writer_flag')

    # Check for CUDA or MPS availability
    if tr.backends.mps.is_available():
        device = tr.device("mps")
        print("Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif tr.cuda.is_available():
        device = tr.device("cuda")
        print("Using CUDA")
    else:
        device = tr.device("cpu")
        print("Using CPU")

    experiment_number = len(os.listdir(experiments_dir)) + 1
    writer = None

    # Device count (for CUDA only, MPS always has 1 device)
    if tr.cuda.is_available():
        num_cuda_devices = tr.cuda.device_count()
        print(f"Number of CUDA devices available: {num_cuda_devices}")
    else:
        print("CUDA is not available on this system")

    # Configuration
    env_config = hyperparams_dict("Environment", hparam_path)
    print(env_config)
    agent_config = hyperparams_dict("Agent", hparam_path)
    train_config = hyperparams_dict('Training', hparam_path)

    # Initialize Environment, Agents, and Replay Buffer
    env = ActiveBrownianEnv(env_config)
    agent = SoftActorCritic(
        agent_config,
        device
    ).to(device)

    replay_buffer = SingleAgentPrioritizedReplayBuffer(
        train_config['buffer_size'],
        train_config['alpha'],
        train_config['beta']
    )
    # Logging
    if writer_flag:
        writer = SummaryWriter(f'{experiments_dir}/{experiment_number}')
        for key, value in agent_config.items():
             writer.add_text(key, str(value))
        for key, value in env_config.items():
             writer.add_text(key, str(value))

    States = []


    for episode in tqdm(range(train_config['n_episodes']), leave=False):
        states, _ = train_episode(
            env, agent, replay_buffer,
            episode, train_config,
            writer
        )
        States += states

        if episode % train_config['save_freq'] == 0:
            np.save(f'{experiments_dir}/{experiment_number}/train_states.npy', np.array(States).squeeze())
            tr.save(agent, f'{experiments_dir}/{experiment_number}/agents.pth')
            if writer_flag: writer.flush()


    env.close()
    if writer_flag: writer.close()

    np.save(f'{experiments_dir}/{experiment_number}/train_states.npy', np.array(States).squeeze())
    tr.save(agent, f'{experiments_dir}/{experiment_number}/agents.pth')
    max_memory_used = tr.cuda.max_memory_allocated() / 1024 ** 3
    print(f'Maximum Memory Allocated: {max_memory_used: .2f} GB')


if __name__ == '__main__':
    #tr.set_precision(32)
    tr.set_default_dtype(tr.float32)
    
    train()