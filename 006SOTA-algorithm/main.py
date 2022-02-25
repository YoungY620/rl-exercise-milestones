import gym
import torch
import numpy as np
from alive_progress import alive_bar
from ppo import PPO, PPOBuffer
import argparse

import core

'''暂时使用硬编码的超参数'''
MODEL_PATH = 'D:\\MyZone\\Study\\AI\\周博磊-强化学习概论\\exercise-note\\006SOTA-algorithm\\temp\\model\\ppo.pt'
LOG_PATH = 'D:\\MyZone\\Study\\AI\\周博磊-强化学习概论\\exercise-note\\006SOTA-algorithm\\temp\\log\\ppo.csv'
SAVE_FREQ = 50
EVALUATION_FREQ = 10
REUSE_MODULE = False

VEL_GRADING = 0.5
MAX_EPISODES = 1000
BUFFER_CAPACITY = 3000
TRAIN_BATCH_PER_EPOCH = 10
BATCH_SIZE = 256

DISCOUNT_RATE = 0.99
GAE_WEIGHT = 0.95

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--algor', default='ppo', choices=['trpo', 'ppo', 'acktr', 'td3', 'ddpg', 'sac'])
    args = parser.parse_args()

    env = gym.make("Pendulum-v1")

    # seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_kwargs = {'act_dim': env.action_space.shape, 'obs_dim': env.observation_space.shape}
    b_kwargs = {'act_dim': env.action_space.shape, 'obs_dim': env.observation_space.shape}
    if args.algor == 'ppo':
        trainer = PPO(t_kwargs)
        b_kwargs['capacity'], b_kwargs['discount_rate'], b_kwargs['gae_weight'] = \
            BUFFER_CAPACITY, DISCOUNT_RATE, GAE_WEIGHT
        buffer = PPOBuffer(b_kwargs)
    else:
        raise NotImplementedError

    if args.eval:
        trainer.load(MODEL_PATH)
        o = env.reset()
        for _ in range(BUFFER_CAPACITY):
            with torch.no_grad():
                a, v, _ = trainer.select_action(o)
            o, r, d, _ = env.step(a)
            env.render() 
    else:
        def get_env(): return env
        tracer = core.Tracer(
            logfile_path=LOG_PATH, modelfile_path=MODEL_PATH, max_step=BUFFER_CAPACITY,
            get_env=get_env, reuse_model=REUSE_MODULE)
        tracer.register_trainer(trainer)

        with alive_bar(MAX_EPISODES*(BUFFER_CAPACITY+TRAIN_BATCH_PER_EPOCH)) as bar:
            for epoch in tracer.trace(range(MAX_EPISODES)):
                ob = env.reset()
                # repeat, until buffer is filled, but not just until ONE env scenarios is done
                for step in range(BUFFER_CAPACITY):
                    with torch.no_grad():
                        a, v, logp_a = trainer.select_action(ob)

                    ob, r, d, i = env.step(a)
                    buffer.push(ob, a, r, v, logp_a)

                    if d or step == BUFFER_CAPACITY-1:
                        if d:
                            v = 0.0
                            ob = env.reset()
                        else:
                            with torch.no_grad():
                                _, v, _ = trainer.select_action(ob)
                            v = v.numpy()
                        buffer.after_episode(next_v=v)
                    bar()
                
                for _ in range(TRAIN_BATCH_PER_EPOCH):  
                    trainer.train(buffer, batch_size=BATCH_SIZE)
                    bar()

                buffer.clear()
            
