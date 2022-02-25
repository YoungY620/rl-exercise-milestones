import gym 

env = gym.make("CarRacing-v0")

print(env.action_space.shape)
print(env.observation_space.shape)
