import numpy as np
import torch

import core
from core import MLPActorCritic, Trainer


class PPOBuffer(object):

    def __init__(self, capacity, obs_dim, act_dim, discount_rate, gae_weight) -> None:
        self.size = 0
        self.capacity = capacity
        self.starting_ptr = 0

        # recorded buf
        self.obs_buf = np.zeros(core.combined_shape(capacity, obs_dim), dtype=np.float64)
        self.act_buf = np.zeros(core.combined_shape(capacity, act_dim), dtype=np.float64)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float64)
        self.val_buf = np.zeros((capacity, 1), dtype=np.float64)
        self.logp_buf = np.zeros((capacity, 1), dtype=np.float64)

        # computed buf
        self.adv_buf = np.zeros((capacity, 1), dtype=np.float64)
        self.ret_buf = np.zeros((capacity, 1), dtype=np.float64)

        # config
        self.discount_rate = discount_rate
        self.gae_weight = gae_weight
    
    def push(self, o, a, r, v, logp):
        assert self.size < self.capacity

        self.obs_buf[self.size] = o
        self.act_buf[self.size] = a
        self.rew_buf[self.size] = r
        self.val_buf[self.size] = v
        self.logp_buf[self.size] = logp

        self.size += 1

    def after_episode(self, next_v=0):
        rew = np.append(self.rew_buf[self.starting_ptr:self.size], next_v)
        val = np.append(self.val_buf[self.starting_ptr:self.size], next_v)

        deltas = rew[:-1] + self.discount_rate*val[1:] - val[:-1]
        self.adv_buf[self.starting_ptr:self.size] = core.discount_cumsum(deltas, self.discount_rate * self.gae_weight).reshape(-1, 1)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[self.starting_ptr:self.size] = core.discount_cumsum(rew, self.discount_rate).reshape(-1, 1)[:-1]

        self.starting_ptr = self.size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.obs_buf[ind]), 
            torch.FloatTensor(self.act_buf[ind]), 
            torch.FloatTensor(self.logp_buf[ind]),
            torch.FloatTensor(self.adv_buf[ind]),
            torch.FloatTensor(self.ret_buf[ind])
        )

    def clear(self):
        self.size = 0
        self.starting_ptr = 0


class PPO(Trainer):

    def __init__(
        self, observation_space, action_space,
        clipping_param=0.2, policy_lr=3e-8, value_lr=1e-3,
        train_pi_iter=80, train_v_iter=80, target_kl=0.01
    ) -> None:
        super().__init__()

        self._module = MLPActorCritic(observation_space, action_space)
        self.policy_optimizer = torch.optim.Adam(self._module.pi.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self._module.v.parameters(), lr=value_lr)

        ## hook for checking farward & backward
        # self.grad_block, self.fmap_block = dict(), dict()
        # def backward_hook(module, grad_in, grad_out):
        #     self.grad_block['grad_in'] = grad_in
        #     self.grad_block['grad_out'] = grad_out
        # def farward_hook(module, inp, outp):
        #     self.fmap_block['input'] = inp
        #     self.fmap_block['output'] = outp
        # self._module.pi.mu_net.register_forward_hook(farward_hook)
        # self._module.pi.mu_net.register_full_backward_hook(backward_hook)

        # hyperparameter config
        self.clipping_param = clipping_param
        self.train_pi_iter = train_pi_iter
        self.train_v_iter = train_v_iter
        self.target_kl = target_kl

    def select_action(self, state):
        state = torch.from_numpy(np.array(state))
        a, v, logp = self._module.forward(state)
        return a, v, logp

    def train(self, buffer, batch_size):
        obs, act, logp, adv, ret = buffer.sample(batch_size)

        # update policy 
        for _ in range(self.train_pi_iter):
            # ppo policy loss with kl constraint
            self.policy_optimizer.zero_grad()
            _, new_logp = self._module.pi.forward(obs, act=act)
            approx_kl = (logp - new_logp).mean().item()
            ratio = torch.exp(new_logp - logp)
            clipped_ratio = torch.clamp(ratio, 1-self.clipping_param, 1+self.clipping_param)*adv
            policy_loss = -(torch.min(clipped_ratio, ratio*adv)).mean()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            if approx_kl > 1.5*self.target_kl:
                break

        # update value evaluation module
        for _ in range(self.train_v_iter):
            self.value_optimizer.zero_grad()
            val = self._module.v.forward(obs)
            loss_v = ((val - ret)**2).mean()
            loss_v.backward()
            self.value_optimizer.step()

        # print(self.fmap_block)
        # print(self.grad_block)

    def save(self, path):
        torch.save(self._module.state_dict(), path)
    
    def load(self, path):
        self._module.load_state_dict(torch.load(path))
