import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """
    My DQN architecture.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize a Deep Q-Network (DQN) model.
        """
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        """
        Forward pass of the DQN model.
        """
        return self.fc(x)


class Agent:
    """
    My Agent Architecture
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize RL agent.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.prev_observation = None
        self.prev_action = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(observation_space.shape[0], action_space.n).to(self.device)
        self.target_model = DQN(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.buffer = deque(maxlen=200000)  # TODO: tune this
        self.gamma = 0.99
        self.batch_size = 64  # TODO: tune this
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # TODO: tune this
        self.epsilon_min = 0.05
        self.update_freq = 500  # TODO: tune this
        self.steps = 0

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Choose an action based on the given observation.
        """

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = self.model(observation_tensor).argmax().item()

        self.prev_observation = observation
        self.prev_action = action
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Update the agent.
        """

        if self.prev_observation is not None:
            self.buffer.append(
                (
                    self.prev_observation,
                    self.prev_action,
                    reward,
                    observation,
                    terminated,
                )
            )

        if len(self.buffer) < self.batch_size:
            return

        if self.steps % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps += 1

        batch = random.sample(self.buffer, self.batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_values = self.model(obs).gather(1, action)
        target_q_values = (
            self.target_model(next_obs).max(dim=1, keepdim=True)[0].detach()
        )
        target_q_values = reward + self.gamma * target_q_values * (1 - done)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Double DQN
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions)
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
