# train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from dmc import make_dmc_env

import GPUtil
torch.cuda.empty_cache()
def get_gpu_with_most_memory():
    gpus = GPUtil.getGPUs()
    max_free_mem = -1
    gpu_id_with_max_mem = -1
    for gpu in gpus:
        free_mem = gpu.memoryFree
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            gpu_id_with_max_mem = gpu.id
    return gpu_id_with_max_mem, max_free_mem

# ==== Actor ====
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

# ==== Critic ====
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, size=1000000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(act, dtype=torch.float32),
            torch.tensor(rew, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# ==== Ornstein-Uhlenbeck Noise ====
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# ==== Training ====
def train():
    env = make_dmc_env("cartpole-balance", seed=np.random.randint(1e6), flatten=True, use_pixels=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim, act_dim)
    target_actor = Actor(obs_dim, act_dim)
    target_critic = Critic(obs_dim, act_dim)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = ReplayBuffer()
    noise = OUNoise(act_dim)

    gamma = 0.99
    tau = 0.005
    batch_size = 128
    total_steps = 100_000
    warmup_steps = 1000

    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []
    best_avg_reward = -float('inf') 
    for step in tqdm(range(total_steps), desc="Training"):
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor(obs_tensor).squeeze(0).numpy()
                action += noise.sample()
                action = np.clip(action, -1.0, 1.0)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        episode_reward += reward






        if done:
            obs, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            noise.reset()
        
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"AvgReward (last 10): {avg_reward:.2f}")
        
      
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(actor.state_dict(), "best_actor.pth")
                    print(f"New best model saved with avg reward: {avg_reward:.2f}")










        if step >= warmup_steps and len(buffer) >= batch_size:
            o, a, r, no, d = buffer.sample(batch_size)

            with torch.no_grad():
                target_q = target_critic(no, target_actor(no))
                target_value = r + gamma * (1 - d) * target_q

            critic_loss = nn.MSELoss()(critic(o, a), target_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(o, actor(o)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    torch.save(actor.state_dict(), "actor.pth")
    print("Training complete. Saved actor to actor.pth.")

if __name__ == "__main__":
    train()
