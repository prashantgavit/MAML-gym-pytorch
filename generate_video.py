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




# import imageio
import os
import pickle

def render_episode_video( observations):
    with open('maml-halfcheetah-vel/config.json', 'r') as f:
        config = json.load(f)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    frames = []

    obs = env.reset()
    for i in range(len(observations)):
        env_state = observations[i]
        env.unwrapped.set_state_from_obs(env_state)  # ← We'll define this function below
        frame = env.render()
        frames.append(frame)

    env.close()

    # imageio.mimsave(filename, frames, fps=fps)
    # print(f"Saved video to {filename}")

with open('episodes.pkl', 'rb') as f:
    data = pickle.load(f)

# Access
tasks = data['tasks']
train_episodes = data['train_episodes']
valid_episodes = data['valid_episodes']

a = train_episodes[0][0]

obs = a.observations

render_episode_video( obs)