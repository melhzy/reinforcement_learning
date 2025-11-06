"""
Deep Q-Network (DQN) Implementation
Neural network-based Q-learning for continuous state spaces
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dims: List[int] = [128, 128]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network and Target Network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training step counter
        self.steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class MedicalClassificationEnv:
    """
    Simplified medical classification environment for demonstration.
    
    The agent receives patient features and must classify or order additional tests.
    """
    
    def __init__(self, num_features: int = 10, num_classes: int = 3):
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Actions: classify as class 0, 1, 2, or request additional test (3)
        self.action_dim = num_classes + 1
        self.state_dim = num_features
        
        # Generate synthetic patient data
        self.patients = self._generate_synthetic_data(num_samples=100)
        self.current_patient_idx = 0
        self.current_state = None
        
    def _generate_synthetic_data(self, num_samples: int) -> List[Tuple[np.ndarray, int]]:
        """Generate synthetic patient data."""
        patients = []
        
        for _ in range(num_samples):
            # Generate features
            features = np.random.randn(self.num_features)
            
            # Generate label based on features
            if features[0] > 0.5:
                label = 0
            elif features[0] < -0.5:
                label = 1
            else:
                label = 2
            
            patients.append((features, label))
        
        return patients
    
    def reset(self) -> np.ndarray:
        """Reset environment with a new patient."""
        self.current_patient_idx = np.random.randint(len(self.patients))
        self.current_state, self.true_label = self.patients[self.current_patient_idx]
        return self.current_state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action in the environment.
        
        Args:
            action: Action to take (0-2: classify, 3: request test)
            
        Returns:
            next_state, reward, done
        """
        if action < self.num_classes:
            # Classification action
            if action == self.true_label:
                reward = 1.0  # Correct classification
            else:
                reward = -1.0  # Incorrect classification
            done = True
            next_state = self.current_state
        else:
            # Request additional test (not implemented in this simple version)
            reward = -0.1  # Small penalty for additional test
            done = False
            next_state = self.current_state
        
        return next_state, reward, done


def train_dqn(
    env,
    agent: DQNAgent,
    num_episodes: int = 1000,
    verbose: bool = True
) -> List[float]:
    """
    Train DQN agent on environment.
    
    Args:
        env: Environment to train on
        agent: DQN agent
        num_episodes: Number of episodes
        verbose: Whether to print progress
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards


if __name__ == "__main__":
    print("Training DQN Agent on Medical Classification Environment...")
    
    # Create environment
    env = MedicalClassificationEnv(num_features=10, num_classes=3)
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        batch_size=32,
        hidden_dims=[64, 64]
    )
    
    # Train agent
    rewards = train_dqn(env, agent, num_episodes=1000, verbose=True)
    
    # Test trained agent
    print("\nTesting trained agent...")
    test_rewards = []
    for _ in range(100):
        state = env.reset()
        action = agent.select_action(state, training=False)
        _, reward, _ = env.step(action)
        test_rewards.append(reward)
    
    print(f"Test Accuracy: {np.mean([r > 0 for r in test_rewards]):.2%}")
    print(f"Average Test Reward: {np.mean(test_rewards):.2f}")
