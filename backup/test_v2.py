import torch
import yaml
import json
import gym
import os

from maml_rl.envs import *  # Register custom envs
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import reinforce_loss


def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config format. Use .yaml, .yml, or .json")


def single_gradient_update(config_path, seed=None, use_cuda=False, checkpoint_path=None):
    # Load configuration
    config = load_config(config_path)

    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create environment
    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))

    # Create policy
    policy = get_policy_for_env(env,
                                 hidden_sizes=config['hidden-sizes'],
                                 nonlinearity=config['nonlinearity'])

    # Load policy checkpoint if provided
    # if checkpoint_path is not None:
    #     with open(checkpoint_path, 'rb') as f:
    #         state_dict = torch.load(f, map_location=device)
    #         policy.load_state_dict(state_dict)
    #     print(f"🔁 Loaded policy from checkpoint: {checkpoint_path}")

    policy.share_memory()
    policy.to(device)

    # Create baseline and sampler
    baseline = LinearFeatureBaseline(get_input_size(env))
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=seed,
                               num_workers=1)

    # Sample tasks and rollouts
    tasks = sampler.sample_tasks(1)
    (train_batches, valid_batches) = sampler.sample(tasks,
        num_steps=1,
        fast_lr=config['fast-lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae-lambda'],
        device=device)

    train_episodes = train_batches[0][0]
    valid_episodes = valid_batches[0]

    # Inner loop: Adapt policy
    train_loss = reinforce_loss(policy, train_episodes)
    adapted_params = policy.update_params(train_loss, step_size=config['fast-lr'], first_order=True)

    # Outer loop: Evaluate and update policy
    valid_loss = reinforce_loss(policy, valid_episodes, params=adapted_params)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    valid_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=40.0)
    optimizer.step()

    print(f"✔️ Single gradient update done")
    print(f"Train loss: {train_loss.item():.4f} | Valid loss: {valid_loss.item():.4f}")


# Example usage
if __name__ == '__main__':
    config_path = 'maml-halfcheetah-vel-v1/config.json'  # or YAML path
    checkpoint = 'maml-halfcheetah-vel-v1/policy.th'     # optional checkpoint
    single_gradient_update(config_path, seed=42, use_cuda=False, checkpoint_path=checkpoint)
