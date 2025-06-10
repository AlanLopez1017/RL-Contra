import random
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from nes_py.wrappers import JoypadSpace

from Contra.actions import COMPLEX_MOVEMENT

# Hiperparámetros
EPISODES = 500
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEM_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
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
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (
            obs.shape[-1],
            obs.shape[0],
            obs.shape[1],
        )  # solo actualiza la “descripción” del entorno.
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(), shape=new_shape, dtype=obs.dtype
        )  # aseguran que los límites siguen siendo correctos

    def observation(self, observation):
        return np.moveaxis(
            observation, 2, 0
        )  # Esta función se llama automáticamente cada vez que el entorno devuelve una observación (por ejemplo al hacer env.step() o env.reset()).


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype,
        )
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


def make_env(env_name: str, **kwargs):

    env = gym.make(env_name, **kwargs)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = FrameSkip(env, skip=4)
    env = WrapFrame(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)

    return env


# Preprocesamiento de imagen: grayscale + resize
def preprocess(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    state = np.expand_dims(state, axis=0)  # (1, 84, 84)
    return state.astype(np.float32) / 255.0


# Red neuronal convolucional
"""
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        x = torch.FloatTensor(x).to(device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

"""


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
        # Calcular automáticamente el tamaño después de las convoluciones
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


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


# Acción con política epsilon-greedy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(state).argmax().item()


# Entrenamiento
def train():
    if len(memory) < BATCH_SIZE:
        return

    # batch = random.sample(memory, BATCH_SIZE)
    batch = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * GAMMA * max_next_q

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Inicialización del entorno
# env = gym.make("Contra-v0")  # usa "rgb_array" si no quieres visualizar
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = FrameSkip(env, skip=4)
env = make_env("Contra-v0")

action_size = env.action_space.n

state = env.reset()
# state = preprocess(state)
input_shape = state.shape

policy_net = DQN(input_shape, action_size).to(device)
target_net = DQN(input_shape, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# memory = deque(maxlen=MEM_SIZE)
memory = ExperienceBuffer(MEM_SIZE)

epsilon = EPS_START

# Entrenamiento principal
for episode in range(EPISODES):
    state = env.reset()
    # state = preprocess(state)
    total_reward = 0

    for t in range(1000):  # creo que es lo de batch
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        # next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        train()

        if done:
            break

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # env.render()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total reward = {total_reward}, Epsilon = {epsilon:.3f}")

torch.save(policy_net.state_dict(), "dqn_contra.pth")
env.close()
