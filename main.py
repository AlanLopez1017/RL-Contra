import gym
from nes_py.wrappers import JoypadSpace

from Contra.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

env = gym.make("Contra-v0")
env = JoypadSpace(env, RIGHT_ONLY)

print("actions", env.action_space)
print("observation_space ", env.observation_space.shape[0])


done = False
env.reset()
for step in range(5000):
    if done:
        print("Over")
        break
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
