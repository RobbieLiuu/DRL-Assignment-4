# student_agent.py
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

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

    def forward(self, obs):
        return self.net(obs)

class Agent(object):
    def __init__(self):
        self.obs_dim = 5  # CartPole observation: [x, theta, x_dot, theta_dot, time]
        self.act_dim = 1
        self.actor = Actor(self.obs_dim, self.act_dim)
        self.actor.load_state_dict(torch.load("best_actor.pth", map_location=torch.device("cpu")))
        self.actor.eval()

    def act(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)
