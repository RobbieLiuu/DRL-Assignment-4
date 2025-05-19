import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import os

# Actor Network definition (copied from your training code)
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that uses a trained DDPG model."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        
        # Set device (use CUDA if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Define observation dimension from DMC humanoid
        self.obs_dim = 67  # Based on your training code's observation space
        self.act_dim = 21  # From the action space dimension
        
        # Initialize the actor model
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        
        # Load the best saved model
        checkpoint_path = "best_actor_ddpg.pth"  # or "humanoid_actor.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"==> Loading actor model from {checkpoint_path}")
            self.actor.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.actor.eval()  # Set to evaluation mode
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. Using random actions.")
            self.use_random = True
            return
            
        self.use_random = False

    def act(self, observation):
        # If model wasn't loaded successfully, fall back to random actions
        if getattr(self, "use_random", False):
            return self.action_space.sample()
        
        # Convert observation to tensor and get prediction from model
        try:
            with torch.no_grad():  # No need to track gradients for inference
                obs_tensor = torch.tensor(observation.astype(np.float32), device=self.device).unsqueeze(0)
                action = self.actor(obs_tensor).cpu().numpy().squeeze(0)
                
            # Ensure the action is within bounds [-1.0, 1.0]
            action = np.clip(action, -1.0, 1.0)
            return action
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to random action if prediction fails
            return self.action_space.sample()

