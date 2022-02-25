import gym
import numpy as np
from math import floor
from alive_progress import alive_bar

'''
蒙特卡洛法 实现的 （失败的） policy control with epsilon greedy

由于 MountainCar 的 MDP 树叶节点规模为 (3*N_STATE^2)^MAX_STEP = 10^700
同时遍历时 未 及时 优化 policy，因此是不可能成功的

TD 的算法 本质上是及时 对 MDP 进行了 剪枝
'''

N_STATE = 40

MAX_STEP = 200
MAX_EPISODE = int((3*N_STATE*N_STATE)**MAX_STEP * 0.7)

INIT_EPSILON = 1.0
GAMMA = 1.0


def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    for i in range(MAX_STEP):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += GAMMA ** i * reward
        if done:
            break
    return total_reward


def epsilon_greedy_action(q_table, p, v, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, np.array(q_table).shape[2])
    else:
        return np.argmax(q_table[p][v])


def obs_to_state(obs):
    '''observation to state'''
    obs_high = env.observation_space.high
    obs_low = env.observation_space.low
    obs_interval = (obs_high - obs_low) / N_STATE

    state_p = floor((obs[0] - obs_low[0]) / obs_interval[0])
    state_v = floor((obs[1] - obs_low[1]) / obs_interval[1])

    return state_p, state_v


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    q_table = np.zeros((N_STATE, N_STATE, 3))
    sample_counter = np.zeros((N_STATE, N_STATE, 3))

    with alive_bar(MAX_EPISODE) as bar:
        for epi in range(MAX_EPISODE):
            rs = []
            ps = []
            vs = []
            actions = []
            epsilon = 1 / epi if epi > 0 else INIT_EPSILON
            obs = env.reset()
            for t in range(MAX_STEP):
                p, v = obs_to_state(obs)
                action = epsilon_greedy_action(q_table, p, v, epsilon)
                obs, reward, done, _ = env.step(action)
                rs.append(reward)
                ps.append(p)
                vs.append(v)
                actions.append(action)
            
            total_reward = 0
            for i in range(t, 0, -1):
                p, v, a, r = ps[i], vs[i], actions[i], rs[i]
                sample_counter[p][v][a] += 1
                total_reward = r + GAMMA * total_reward
                # running mean
                q_table[p][v][a] += total_reward + (1 / sample_counter[p][v][a]) * q_table[p][v][a]
        
            bar()

    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    for _ in range(2):
        run_episode(env, solution_policy, True)
    env.close()   
    
