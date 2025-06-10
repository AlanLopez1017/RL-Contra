
from collections import deque

import cv2
import gym
import numpy as np
import torch
from gym import spaces
from nes_py.wrappers import JoypadSpace

from Contra.actions import COMPLEX_MOVEMENT


# Wrappers (deben coincidir exactamente con los usados en entrenamiento)
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = spaces.Box(low=obs.low.min(), high=obs.high.max(), shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        new_obs = spaces.Box(obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0), dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = deque(maxlen=n_steps)

    def reset(self, *, seed=None, options=None):
        for _ in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        self.buffer.append(observation)
        return np.concatenate(self.buffer)

# Red neuronal (misma arquitectura)
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = torch.FloatTensor(x).to(next(self.parameters()).device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Función para crear entorno
def make_env(env_name):
    env = gym.make(env_name)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = FrameSkip(env, skip=4)
    env = WrapFrame(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env

# Carga del entorno y del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env("Contra-v0")

state = env.reset()
input_shape = state.shape
n_actions = env.action_space.n

model = DQN(input_shape, n_actions).to(device)
model.load_state_dict(torch.load("dqn_contra.pth", map_location=device))
model.eval()

# Reproducción de un episodio
total_reward = 0
done = False
while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(state_tensor).argmax().item()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    env.render()  # Quita si no quieres visualizar

print(f"Total reward (test run): {total_reward}")
env.close()
