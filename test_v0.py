import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

from gym.wrappers import Monitor

with open('maml-halfcheetah-vel/config.json', 'r') as f:
    config = json.load(f)

env = gym.make(config['env-name'], **config['env-kwargs'])


policy = get_policy_for_env(env,
                            hidden_sizes=config['hidden-sizes'],
                            nonlinearity=config['nonlinearity'])
# with open('maml-halfcheetah-vel/policy.th', 'rb') as f:
#     state_dict = torch.load(f, map_location=torch.device('cpu'))
#     policy.load_state_dict(state_dict)
policy.share_memory()



env = Monitor(env, "./video", force=True)

obs = env.reset()
done = False

while not done:
    # action = env.action_space.sample()
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # add batch dim
        pi = policy(obs_tensor)  # get action distribution
        action = pi.sample().squeeze(0).numpy()  # sample from it and remove batch dim
        print(action)

    obs, reward, done, info = env.step(action)
env.close()


# baseline = LinearFeatureBaseline(get_input_size(env))

# # Sampler
# sampler = MultiTaskSampler(config['env-name'],
#                            env_kwargs=config['env-kwargs'],
#                            batch_size=config['fast-batch-size'],
#                            policy=policy,
#                            baseline=baseline,
#                            env=env,
#                            seed=1,
#                            num_workers=1)

# logs = {'tasks': []}
# train_returns, valid_returns = [], []
# tasks = sampler.sample_tasks(40)
# # print(tasks)

# train_episodes, valid_episodes = sampler.sample(tasks,
#                                                 num_steps=config['num-steps'],
#                                                 fast_lr=config['fast-lr'],
#                                                 gamma=config['gamma'],
#                                                 gae_lambda=config['gae-lambda'],
#                                                 device='cpu')

# import pickle

# with open('episodes.pkl', 'wb') as f:
#     pickle.dump({
#         'tasks': tasks,
#         'train_episodes': train_episodes,
#         'valid_episodes': valid_episodes
#     }, f)

