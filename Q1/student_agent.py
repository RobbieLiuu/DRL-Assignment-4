import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Define the same Actor model used in training
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return 2.0 * self.net(x)  # scale to [-2, 2]

# Agent class
class Agent(object):
    def __init__(self):
        self.actor = Actor(3, 1)
        self.actor.load_state_dict(torch.load('actor.pth', map_location='cpu'))
        self.actor.eval()

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).numpy()[0]
        return np.clip(action, -2.0, 2.0)
