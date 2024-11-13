from environment import ActiveBrownianEnv
from models import SoftActorCritic
import ray
from ray import train, tune
from utils.utils import *
from utils.train_utils import train_episode
import json
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
from ray.tune.schedulers import ASHAScheduler
import torch as tr
import dotenv
import os

def costom_trial_dirname(trial):
    return f'{trial.trial_id}'

dotenv.load_dotenv()
work_dir = os.getenv('work_dir')
hparam_path = os.getenv('hparam_path')
train_config = hyperparams_dict('Training', hparam_path)
os.environ["RAY_TMPDIR"] = work_dir + '/tmp'

def objective(config):
    try:
        # Device Konfiguration
        if tr.backends.mps.is_available():
            device = tr.device("mps")
            print("Using MPS (Metal Performance Shaders) on Apple Silicon")
        elif tr.cuda.is_available():
            device = tr.device("cuda")
            print("Using CUDA")
        else:
            device = tr.device("cpu")
            print("Using CPU")

        env_config = hyperparams_dict("Environment", hparam_path)
        train_env = ActiveBrownianEnv(env_config)
        obs_dim = env_config['state_dim']*env_config['n_neighbors']+env_config['state_dim']
        config['obs_dim'] = obs_dim
        agents = SoftActorCritic(
            config,
            device
        ).to(device)
        
        replay_buffer = SingleAgentPrioritizedReplayBuffer(
            config['buffer_size'],
            config['batch_size'],
        )
        last_reward = 0
        for epoch in range(int(train_config['n_episodes'])):
            _, reward = train_episode(train_env, agents, replay_buffer, epoch, train_config)
            assert not np.isnan(reward).any(), 'Nan in reward'
            train.report({"reward" : reward.mean()})
            last_reward = reward

        train_env.close()
    except Exception as e: 
        raise e

    return {"reward" : last_reward}
    
config = hyperparams_dict("Agent", hparam_path)
search_space = {
    'learning_rate_critic': tune.loguniform(1e-5, 1e-3),
    'learning_rate_actor': tune.loguniform(1e-5, 1e-3),
    'entropy_coeff': tune.uniform(1e-4, 1e-2),
}
# Overwrite the default config with the search space
for key in search_space.keys():
    config[key] = search_space[key]

# Create an Optuna pruner instance
sampler = TPESampler()
algo = OptunaSearch(sampler=sampler)

scheduler = ASHAScheduler(
    max_t=int(train_config['n_episodes']),
    grace_period=int(train_config['n_episodes']//10),
    reduction_factor=train_config['reduction_factor'],
)

# Angepasste Ressourcenkonfiguration für Apple Silicon
resources_per_trial = {
    "cpu": 2,  # Anzahl der CPU-Kerne
}

# Füge MPS nur hinzu, wenn verfügbar
if tr.backends.mps.is_available():
    resources_per_trial["mps"] = 1  # Oder entfernen Sie dies, wenn Sie MPS nicht explizit zuweisen möchten

tuner = tune.Tuner(
    tune.with_resources(
        objective,
        resources=resources_per_trial
    ),
    tune_config=tune.TuneConfig(
        metric="reward",
        mode="max",
        search_alg=algo,
        num_samples=train_config['num_samples'],
        scheduler=scheduler,
        trial_dirname_creator=costom_trial_dirname
    ),
    run_config=train.RunConfig(
        verbose=1,
        failure_config=train.FailureConfig(fail_fast=True),
    ),
    param_space=config
)

# Initialisiere Ray mit angepasster Speicherkonfiguration
ray.init(
    object_store_memory=int(2e9),  # 2GB Speicher
    _system_config={
        "object_spilling_config": json.dumps({
            "type": "filesystem",
            "params": {
                "directory_path": os.path.join(work_dir, "ray_spill")
            }
        })
    }
)

results = tuner.fit()
fname = hparam_path + '/best_hyperparams.json'
with open(fname, 'w') as f:
    json.dump(results.get_best_result().config, f)

print(results.get_best_result().config)