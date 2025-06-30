import argparse
import os
import shutil
from random import random, randint, sample
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris


def get_args():
    parser = argparse.ArgumentParser("DQN Training for Tetris")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    return parser.parse_args()


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if device.type == 'cuda':
        torch.cuda.manual_seed(123)

    # Limpieza de logs previos
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path, exist_ok=True)
    writer = SummaryWriter(opt.log_path)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset().to(device)
    replay_memory = deque(maxlen=opt.replay_memory_size)

    best_score = float('-inf')  # Inicializar mejor score
    epoch = 0

    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()

        # ε-greedy exploration
        epsilon = opt.final_epsilon + max(opt.num_decay_epochs - epoch, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs
        u = random()
        random_action = u <= epsilon

        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(next_states).squeeze(-1)
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index]
        action = next_actions[index]

        reward, done = env.step(action, render=True)

        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines

            state = env.reset().to(device)

            #  Guardar mejor modelo si supera
            if final_score > best_score:
                best_score = final_score
                os.makedirs(opt.saved_path, exist_ok=True)
                torch.save(model.state_dict(), f"{opt.saved_path}/best_model.pth")
                print(f"✔️ Nuevo mejor modelo guardado con score {best_score} en epoch {epoch}")

        else:
            state = next_state
            continue

        # Solo empezar entrenamiento cuando haya suficiente en el buffer
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.stack(state_batch).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=device).unsqueeze(1)
        next_state_batch = torch.stack(next_state_batch).to(device)

        q_values = model(state_batch)

        model.eval()
        with torch.no_grad():
            next_q_values = model(next_state_batch)
        model.train()

        y_batch = torch.cat([
            reward.unsqueeze(0) if done else reward.unsqueeze(0) + opt.gamma * next_q_val.unsqueeze(0)
            for reward, done, next_q_val in zip(reward_batch, done_batch, next_q_values)
        ], dim=0)

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{opt.num_epochs}, Action: {action}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Lines: {final_cleared_lines}")
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared_lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            os.makedirs(opt.saved_path, exist_ok=True)
            torch.save(model.state_dict(), f"{opt.saved_path}/tetris_{epoch}.pth")

    torch.save(model.state_dict(), f"{opt.saved_path}/tetris_final.pth")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
