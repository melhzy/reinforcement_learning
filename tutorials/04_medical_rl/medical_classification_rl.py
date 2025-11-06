"""
Medical Classification using Reinforcement Learning
Feature selection and cost-aware diagnostic decision making
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
from collections import defaultdict


class MedicalDiagnosisEnv:
    """
    Medical diagnosis environment with multi-modal features.
    
    Agent must select which tests to perform and when to make a diagnosis.
    """
    
    def __init__(
        self,
        num_clinical_features: int = 10,
        num_biomarker_features: int = 5,
        num_microbiome_features: int = 20,
        num_classes: int = 3,
        max_steps: int = 10
    ):
        self.num_clinical = num_clinical_features
        self.num_biomarker = num_biomarker_features
        self.num_microbiome = num_microbiome_features
        self.num_classes = num_classes
        self.max_steps = max_steps
        
        # Feature costs (relative)
        self.clinical_cost = 1.0
        self.biomarker_cost = 5.0
        self.microbiome_cost = 10.0
        
        # Total features
        self.total_features = num_clinical_features + num_biomarker_features + num_microbiome_features
        
        # State: [collected_features, feature_availability_mask, step_count]
        self.state_dim = self.total_features + self.total_features + 1
        
        # Actions: [request_clinical, request_biomarker, request_microbiome, classify_0, classify_1, classify_2]
        self.action_dim = 3 + num_classes
        
        # Generate synthetic dataset
        self.dataset = self._generate_dataset(num_samples=1000)
        
        self.reset()
    
    def _generate_dataset(self, num_samples: int) -> List[Tuple[np.ndarray, int]]:
        """Generate synthetic medical data."""
        dataset = []
        
        for _ in range(num_samples):
            # True class
            true_class = np.random.randint(self.num_classes)
            
            # Generate features based on class
            clinical = np.random.randn(self.num_clinical) + true_class
            biomarker = np.random.randn(self.num_biomarker) + true_class * 0.5
            microbiome = np.random.randn(self.num_microbiome) + true_class * 0.3
            
            features = np.concatenate([clinical, biomarker, microbiome])
            dataset.append((features, true_class))
        
        return dataset
    
    def reset(self) -> np.ndarray:
        """Reset environment with new patient."""
        # Select random patient
        self.patient_features, self.true_class = self.dataset[np.random.randint(len(self.dataset))]
        
        # Initially, no features are collected
        self.collected_features = np.zeros(self.total_features)
        self.feature_mask = np.ones(self.total_features)  # 1 = not yet collected
        
        # Track which feature groups are available
        self.clinical_available = True
        self.biomarker_available = True
        self.microbiome_available = True
        
        self.steps = 0
        self.total_cost = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        return np.concatenate([
            self.collected_features,
            self.feature_mask,
            [self.steps / self.max_steps]
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in environment."""
        reward = 0
        done = False
        info = {}
        
        if action == 0 and self.clinical_available:
            # Request clinical features
            start_idx = 0
            end_idx = self.num_clinical
            self.collected_features[start_idx:end_idx] = self.patient_features[start_idx:end_idx]
            self.feature_mask[start_idx:end_idx] = 0
            self.clinical_available = False
            reward = -self.clinical_cost
            self.total_cost += self.clinical_cost
            
        elif action == 1 and self.biomarker_available:
            # Request biomarker features
            start_idx = self.num_clinical
            end_idx = self.num_clinical + self.num_biomarker
            self.collected_features[start_idx:end_idx] = self.patient_features[start_idx:end_idx]
            self.feature_mask[start_idx:end_idx] = 0
            self.biomarker_available = False
            reward = -self.biomarker_cost
            self.total_cost += self.biomarker_cost
            
        elif action == 2 and self.microbiome_available:
            # Request microbiome features
            start_idx = self.num_clinical + self.num_biomarker
            end_idx = self.total_features
            self.collected_features[start_idx:end_idx] = self.patient_features[start_idx:end_idx]
            self.feature_mask[start_idx:end_idx] = 0
            self.microbiome_available = False
            reward = -self.microbiome_cost
            self.total_cost += self.microbiome_cost
            
        elif action >= 3:
            # Classification action
            predicted_class = action - 3
            
            if predicted_class == self.true_class:
                reward = 10.0 - self.total_cost * 0.1  # Reward for correct classification
                info['correct'] = True
            else:
                reward = -10.0 - self.total_cost * 0.1  # Penalty for incorrect classification
                info['correct'] = False
            
            done = True
            info['predicted'] = predicted_class
            info['true'] = self.true_class
        
        self.steps += 1
        
        # Force termination after max steps
        if self.steps >= self.max_steps:
            done = True
            if 'correct' not in info:
                # Forced to classify
                reward = -5.0
        
        next_state = self._get_state()
        
        return next_state, reward, done, info


class DiagnosticAgent:
    """
    RL agent for medical diagnosis with feature selection.
    Uses DQN architecture.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = [self.memory[i] for i in np.random.choice(len(self.memory), self.batch_size, replace=False)]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, env, episodes: int = 5000, verbose: bool = True):
        """Train agent on environment."""
        episode_rewards = []
        episode_accuracy = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            correct = None
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.store_experience(state, action, reward, next_state, done)
                loss = self.train_step()
                
                state = next_state
                total_reward += reward
                
                if 'correct' in info:
                    correct = info['correct']
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            episode_rewards.append(total_reward)
            if correct is not None:
                episode_accuracy.append(1.0 if correct else 0.0)
            
            if verbose and (episode + 1) % 500 == 0:
                avg_reward = np.mean(episode_rewards[-500:])
                avg_accuracy = np.mean(episode_accuracy[-500:]) if episode_accuracy else 0
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Accuracy: {avg_accuracy:.2%}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards, episode_accuracy


def evaluate_agent(agent: DiagnosticAgent, env, num_episodes: int = 100) -> Tuple[float, float]:
    """Evaluate agent performance."""
    correct = 0
    total_cost = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if 'correct' in info:
                if info['correct']:
                    correct += 1
        
        total_cost += env.total_cost
    
    accuracy = correct / num_episodes
    avg_cost = total_cost / num_episodes
    
    return accuracy, avg_cost


if __name__ == "__main__":
    print("Training Medical Diagnosis Agent with RL...")
    print("=" * 60)
    
    # Create environment
    env = MedicalDiagnosisEnv(
        num_clinical_features=10,
        num_biomarker_features=5,
        num_microbiome_features=20,
        num_classes=3,
        max_steps=10
    )
    
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Number of classes: {env.num_classes}")
    print()
    
    # Create agent
    agent = DiagnosticAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001,
        gamma=0.99
    )
    
    # Train agent
    rewards, accuracy = agent.train(env, episodes=5000, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained agent...")
    test_accuracy, avg_cost = evaluate_agent(agent, env, num_episodes=100)
    
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Average Cost: ${avg_cost:.2f}")
    print(f"Cost-Effectiveness: {test_accuracy / avg_cost:.4f}")
