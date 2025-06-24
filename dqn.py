import math
import random
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from gym.wrappers import RecordVideo
from nes_py.wrappers import JoypadSpace

from Contra.actions import BASIC_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# Hiperparámetros
EPISODES = 50
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEM_SIZE = 10000  # 10000
EPS_START = 1
EPS_END = 0.01
# EPS_DECAY = 0.995
EPS_DECAY_EPISODES = int(0.7 * EPISODES)
EPS_DECAY = (EPS_START - EPS_END) / EPS_DECAY_EPISODES
TARGET_UPDATE = 10
MIN_REPLAY_SIZE = 1000

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
    env = JoypadSpace(env, RIGHT_ONLY)
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


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # control de cuánta prioridad se usa (0 = uniforme, 1 = 100% prioritario)
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # pesos de importancia para corregir sesgo
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalizar

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority


# Acción con política epsilon-greedy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(state).argmax().item()


# Entrenamiento
def train():
    if len(memory) < MIN_REPLAY_SIZE:
        return

    # batch = random.sample(memory, BATCH_SIZE)
    batch = memory.sample(BATCH_SIZE)
    # batch, indices, weights = memory.sample(BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    # weights = torch.FloatTensor(weights).unsqueeze(1).to(device)  #############

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * GAMMA * max_next_q

    loss = nn.MSELoss()(q_values, target_q)

    """
    td_errors = (q_values - target_q).squeeze()  # forma [batch] ############
    losses = td_errors**2  ######

    # 6. Aplicar pesos de importancia y optimizar
    loss = (losses * weights.squeeze()).mean()  ########

    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # new_priorities = (
    #    td_errors.abs().detach().cpu().numpy() + 1e-6
    # )  # evitar prioridad 0 #############
    # memory.update_priorities(indices, new_priorities)  ################


# Inicialización del entorno
# env = gym.make("Contra-v0")  # usa "rgb_array" si no quieres visualizar
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = FrameSkip(env, skip=4)


def get_epsilon(episode):
    if episode < EPS_DECAY_EPISODES:
        decay = (EPS_START - EPS_END) / EPS_DECAY_EPISODES
        return max(EPS_END, EPS_START - decay * episode)
    else:
        return EPS_END


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
# memory = PrioritizedReplayBuffer(MEM_SIZE)

epsilon = EPS_START

best_reward = -float("inf")

reward_list = []
mean_rewards = []
SAVE_EVERY = 10


#######
# eval_env = make_env('Contra-v0')
# eval_env = RecordVideo(
#    eval_env,
#    video_folder = './videos',
#    episode_trigger = lambda episode_id: episode_id % 100 == 0
# )

######

# Entrenamiento principal
for episode in range(EPISODES):
    state = env.reset()
    # state = preprocess(state)
    total_reward = 0

    if episode % 100 == 0 and episode > 0:
        epsilon_override = 0.5
        print(f"Exploración forzada")
    else:
        epsilon_override = get_epsilon(epsilon)

    done = False
    steps = 0
    while not done:  # for t in range(1000):  # Limite de pasos por episodio
        action = select_action(state, epsilon_override)
        next_state, reward, done, _ = env.step(action)

        # next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        train()

        #    if done:
        #        break
        steps += 1

    print(f"steps: {steps}, episode -> {episode}")
    # epsilon = max(EPS_END, epsilon * EPS_DECAY)
    epsilon = max(EPS_END, EPS_START - EPS_DECAY * episode)

    # env.render()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}: Total reward = {total_reward}, Epsilon = {epsilon:.3f}")

    reward_list.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), "best_dqn_contra.pth")
        print(f"New best reward: {best_reward:.2f} — model saved.")

        # try:
        #    video_env = make_env("Contra-v0")
        #    video_env = RecordVideo(
        #        video_env, video_folder="./videos", name_prefix=f"SM-best_ep{episode}"
        #    )
        #    state_v = video_env.reset()
        #    done = False
    #
    #           while not done:
    #              state_tensor = torch.from_numpy(state_v).float().unsqueeze(0).to(device)
    #             with torch.no_grad():
    #                action = policy_net(state_tensor).argmax().item()
    # state_v, _, done, _ = video_env.step(action)

    #      video_env.close()
    #  except Exception as e:
    #     print(f"Error al grabar el video del episidio {episode}")

    if (episode + 1) % SAVE_EVERY == 0:
        mean = np.mean(reward_list[-SAVE_EVERY:])
        mean_rewards.append(mean)
        np.save("rewards_contra.npy", np.array(mean_rewards))


torch.save(policy_net.state_dict(), "dqn_contra.pth")
env.close()
