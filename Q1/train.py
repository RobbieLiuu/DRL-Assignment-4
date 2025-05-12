import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import os




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



# Define Actor and Critic
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

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size=100000):
        self.ptr = 0
        self.size = size
        self.len = 0
        self.obs = np.zeros((size, 3), dtype=np.float32)
        self.act = np.zeros((size, 1), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, 3), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        self.len = min(self.len + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.len, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])

# OU Noise
class OUNoise:
    def __init__(self, mu=0, sigma=0.2, theta=0.15, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = 0

    def sample(self):
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.x += dx
        return self.x

# Training setup
env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

actor = Actor(obs_dim, act_dim)
critic = Critic(obs_dim, act_dim)
actor_target = Actor(obs_dim, act_dim)
critic_target = Critic(obs_dim, act_dim)
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

buffer = ReplayBuffer()
noise = OUNoise()

batch_size = 64
gamma = 0.99
tau = 0.005
steps = 100000
warmup = 1000
obs, _ = env.reset()

episode_reward = 0
recent_rewards = []

for step in range(steps):
    if step < warmup:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            action += noise.sample()
            action = np.clip(action, -2.0, 2.0)

    next_obs, reward, term, trunc, _ = env.step(action)
    done = term or trunc
    buffer.add(obs, action, reward, next_obs, float(done))
    episode_reward += reward
    obs = next_obs if not done else env.reset()[0]

    if done:
        recent_rewards.append(episode_reward)
        episode_reward = 0
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)

    if step >= warmup:
        o, a, r, no, d = buffer.sample(batch_size)
        o = torch.tensor(o)
        a = torch.tensor(a)
        r = torch.tensor(r)
        no = torch.tensor(no)
        d = torch.tensor(d)

        with torch.no_grad():
            target_q = critic_target(no, actor_target(no))
            target = r + gamma * (1 - d) * target_q

        critic_loss = nn.MSELoss()(critic(o, a), target)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        actor_loss = -critic(o, actor(o)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        for tp, p in zip(actor_target.parameters(), actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(critic_target.parameters(), critic.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # Print training progress every 1000 steps
    if step % 1000 == 0 and step >= warmup:
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        print(f"Step {step}: Avg Reward = {avg_reward:.2f}, Actor Loss = {actor_loss.item():.4f}, Critic Loss = {critic_loss.item():.4f}")

# Save trained actor
torch.save(actor.state_dict(), 'actor.pth')
print("Training complete. Actor saved to actor.pth.")
