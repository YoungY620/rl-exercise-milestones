import gym 
import numpy as np
'''
使用 动态规划 实现 FrozenLake-v0
动态规划法基于 贝尔曼公式 的迭代，包括 policy/value iteration
具体，包括 policy evaluation(PI)/optimization(VI) + policy improvement 两步

另外需要注意，动态规划法 需要完整的 MDP 
只有 FrozenLake 等 环境 提供了 完整的 MDP
'''

POLICY_ITERATION = True     # 控制使用 policy 或 value iteration
MAX_ITER = 500000
GAMMA = 1


def run_episode(env, policy, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (GAMMA ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward



def evaluate_policy(env, policy, n = 100):
    scores = [run_episode(env, policy) for _ in range(n)]
    return np.mean(scores)



def compute_values(policy, env):
    eps = 1e-10
    v = np.zeros((env.env.nS,))
    while True:
        pre_v = np.copy(v)
        for s in range(env.env.nS):
            if POLICY_ITERATION:
                action = policy[s]
                v[s] = sum([p * (r + GAMMA * pre_v[s_]) for p, s_, r, _ in env.env.P[s][action]])
            else:
                v[s] = 0                # 仅因为，这里价值不可能为负数
                for action in env.env.P[s]:
                    tmp_v = sum([p * (r + GAMMA * pre_v[s_]) for p, s_, r, _ in env.env.P[s][action]])
                    v[s] = max(tmp_v, v[s])

        if np.sum(np.fabs(pre_v - v)) < eps:
            break
    return v

        

def extract_policy(v, env):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.env.nS)   
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)     # 仅因为，这里价值不可能为负数
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + GAMMA * v[s_]) for p, s_, r, _ in  env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy



if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v1')
    
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))

    for _ in range(MAX_ITER):
        values = compute_values(policy, env)        # policy evaluation(PI)/optimization(VI)
        new_policy = extract_policy(values, env)    # policy improvement

        if np.all(new_policy == policy):
            break
            
        policy = new_policy
    
    scores = evaluate_policy(env, policy)
    print('Average scores = ', np.mean(scores))

