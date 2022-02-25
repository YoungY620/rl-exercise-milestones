from math import floor
import gym
import numpy as np
from alive_progress import alive_bar

'''
MountainCar-v0 临摹
实现 off/on-policy TD0 的 module-free policy control
''' 

# -- hyper-perameters:
N_STATE = 30
ON_POLICY = True
MAX_EPISODE = 5000
MAX_STEP = 200

MIN_LR = 0.01
INIT_LR = 1

EPSILON = 0.1
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


def obs_to_state(obs):
    '''observation to state'''
    obs_high = env.observation_space.high
    obs_low = env.observation_space.low
    obs_interval = (obs_high - obs_low) / N_STATE

    state_p = floor((obs[0] - obs_low[0]) / obs_interval[0])
    state_v = floor((obs[1] - obs_low[1]) / obs_interval[1])

    return state_p, state_v


def epsilon_greedy_action(q_table, p, v):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.randint(0, np.array(q_table).shape[2])
    else:
        return np.argmax(q_table[p][v])


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    q_table = np.zeros((N_STATE, N_STATE, 3))

    with alive_bar(MAX_EPISODE) as bar:
        for epi in range(MAX_EPISODE):
            lr = max(MIN_LR, INIT_LR*(0.85**(epi//100)))
            obs = env.reset()

            for t in range(MAX_STEP):
                p, v = obs_to_state(obs)
                action = epsilon_greedy_action(q_table, p, v)
                obs, reward, done, _ = env.step(action)
                p_, v_ = obs_to_state(obs)

                if done:
                    break

                if ON_POLICY:
                    # on-policy SARSA method
                    action_ = epsilon_greedy_action(q_table, p_, v_)
                    td_target = reward + GAMMA * q_table[p_][v_][action_]
                else:
                    # off-policy q-learning
                    td_target = reward + GAMMA * np.max(q_table[p_][v_])
                
                td_error = td_target - q_table[p][v][action]
                q_table[p][v][action] += lr * td_error

            bar()

    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    for _ in range(2):
        run_episode(env, solution_policy, True)
    env.close()



