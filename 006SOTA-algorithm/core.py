import scipy
from torch import nn
import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np
from gym.spaces import Box, Discrete
import pandas as pd
import core


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    xx = np.array(x, dtype=np.float)
    for i in reversed(range(len(x) - 1)):
        xx[i] += xx[i + 1] * discount
    return xx


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class VCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class QCritic(nn.Module):
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes, activation) -> None:
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [1]) 


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = VCritic(obs_dim, hidden_sizes, activation)

    def forward(self, obs):

        pi = self.pi._distribution(obs)
        a = pi.sample()
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)
        return a, v, logp_a

    # def act(self, obs):
    #     return self.step(obs)[0]


class Trainer:

    def select_action(self, state):
        raise NotImplementedError

    def train(self, buffer, batch_size):
        raise NotImplementedError

    def save(self, path):
        torch.save(self._module.state_dict(), path)

    def load(self, path):
        self._module.load_state_dict(torch.load(path))

class Buffer:

    def push(self, **kwargs):
        raise NotImplementedError

    def after_episode(self, next_v=0):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class Tracer(object):
    '''用于集中管理训练过程的记录'''

    def __init__(self, logfile_path, modelfile_path, log_freq=1, log_write_freq=1, model_write_freq=1, max_step=1000, get_env=None,
                 reuse_model=False) -> None:

        self.logfile_path = logfile_path
        self.modelfile_path = modelfile_path
        self.log_freq = log_freq
        self.log_write_freq = log_write_freq
        self.model_write_freq = model_write_freq
        self.max_step = max_step
        self.get_env = get_env
        self.reuse_model = reuse_model

        self.trace_counter = 0
        self.trainer: Trainer = None  # model, not module
        self.log_df: pd.DataFrame = None

    def register_trainer(self, trainer):
        self.trainer = trainer
        self.log_df = pd.DataFrame(columns=['epoch', 'eval', 'vloss'])
        self.log_df.to_csv(self.logfile_path, header=(not self.reuse_model))

    def trace(self, iterable):
        if self.reuse_model:
            self.trainer.load(self.modelfile_path)

        for item in iterable:
            if self.trace_counter % self.log_freq == 0:
                _env = self.get_env()
                o = _env.reset()
                rews, vals = [], []
                for _ in range(self.max_step):
                    with torch.no_grad():
                        _a, v, _ = self.trainer.select_action(o)
                    o, r, d, _ = _env.step(_a)
                    _env.render()
                    rews.append(r)
                    vals.append(v)

                    if d: break
                _env.reset()

                rews, vals = np.array(rews), np.array(vals)
                gains = core.discount_cumsum(rews, 1)
                vloss = (gains - vals).mean()

                new_line = pd.DataFrame(data=[[self.trace_counter, gains[0], vloss]],
                                        columns=['epoch', 'eval', 'vloss'])
                self.log_df = pd.concat([self.log_df, new_line], ignore_index=False)
            if self.trace_counter % self.log_write_freq == 0:
                self.log_df.to_csv(self.logfile_path, mode='a', header=False)
                self.log_df = pd.DataFrame(columns=['epoch', 'eval', 'vloss'])
            if self.trace_counter % self.model_write_freq == 0:
                self.trainer.save(self.modelfile_path)

            self.trace_counter += 1
            yield item

