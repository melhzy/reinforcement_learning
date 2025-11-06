"""
Alzheimer's Disease Multi-Modal Classification using RL
Integrates clinical, microbiome, and biomarker data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from collections import deque
import random


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with separate pathways for each data type.
    """
    
    def __init__(
        self,
        clinical_dim: int,
        oral_microbiome_dim: int,
        gut_microbiome_dim: int,
        biomarker_dim: int,
        embedding_dim: int = 32
    ):
        super(MultiModalEncoder, self).__init__()
        
        # Separate encoders for each modality
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.oral_encoder = nn.Sequential(
            nn.Linear(oral_microbiome_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.gut_encoder = nn.Sequential(
            nn.Linear(gut_microbiome_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(biomarker_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Fusion layer with attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def forward(self, clinical, oral, gut, biomarker, modality_mask):
        """
        Forward pass with modality masking.
        
        Args:
            clinical, oral, gut, biomarker: Input features
            modality_mask: Binary mask indicating available modalities
        """
        batch_size = clinical.size(0)
        
        # Encode each modality
        clinical_emb = self.clinical_encoder(clinical) * modality_mask[:, 0:1]
        oral_emb = self.oral_encoder(oral) * modality_mask[:, 1:2]
        gut_emb = self.gut_encoder(gut) * modality_mask[:, 2:3]
        biomarker_emb = self.biomarker_encoder(biomarker) * modality_mask[:, 3:4]
        
        # Concatenate and fuse
        combined = torch.cat([clinical_emb, oral_emb, gut_emb, biomarker_emb], dim=1)
        
        return self.fusion(combined)


class MultiModalQNetwork(nn.Module):
    """
    Q-Network with multi-modal encoder for Alzheimer's classification.
    """
    
    def __init__(
        self,
        clinical_dim: int,
        oral_microbiome_dim: int,
        gut_microbiome_dim: int,
        biomarker_dim: int,
        action_dim: int,
        embedding_dim: int = 32
    ):
        super(MultiModalQNetwork, self).__init__()
        
        self.clinical_dim = clinical_dim
        self.oral_dim = oral_microbiome_dim
        self.gut_dim = gut_microbiome_dim
        self.biomarker_dim = biomarker_dim
        
        # Multi-modal encoder
        self.encoder = MultiModalEncoder(
            clinical_dim,
            oral_microbiome_dim,
            gut_microbiome_dim,
            biomarker_dim,
            embedding_dim
        )
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(64 + 5, 64),  # +5 for meta features (cost, steps, etc.)
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through network."""
        # Extract modalities
        clinical = state_dict['clinical']
        oral = state_dict['oral_microbiome']
        gut = state_dict['gut_microbiome']
        biomarker = state_dict['blood_biomarkers']
        modality_mask = state_dict['modality_mask']
        meta_features = state_dict['meta_features']
        
        # Encode multi-modal features
        encoded = self.encoder(clinical, oral, gut, biomarker, modality_mask)
        
        # Combine with meta features
        combined = torch.cat([encoded, meta_features], dim=1)
        
        # Compute Q-values
        return self.q_head(combined)


class AlzheimerEnv:
    """
    Alzheimer's disease diagnosis environment with multi-modal data.
    """
    
    def __init__(
        self,
        clinical_dim: int = 10,
        oral_microbiome_dim: int = 20,
        gut_microbiome_dim: int = 30,
        biomarker_dim: int = 15,
        num_classes: int = 3,  # CN, MCI, AD
        max_steps: int = 6
    ):
        self.clinical_dim = clinical_dim
        self.oral_dim = oral_microbiome_dim
        self.gut_dim = gut_microbiome_dim
        self.biomarker_dim = biomarker_dim
        self.num_classes = num_classes
        self.max_steps = max_steps
        
        # Test costs
        self.test_costs = {
            'clinical': 10,
            'oral_microbiome': 50,
            'gut_microbiome': 75,
            'blood_biomarkers': 100
        }
        
        # Actions: [order_clinical, order_oral, order_gut, order_biomarker, 
        #           classify_CN, classify_MCI, classify_AD]
        self.action_dim = 4 + num_classes
        
        # Generate synthetic dataset
        self.dataset = self._generate_dataset(1000)
        
        self.reset()
    
    def _generate_dataset(self, num_samples: int) -> List[Tuple]:
        """Generate synthetic multi-modal Alzheimer's data."""
        dataset = []
        
        for _ in range(num_samples):
            # True class: 0=CN, 1=MCI, 2=AD
            true_class = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
            
            # Clinical features (age, MMSE, etc.)
            clinical = np.random.randn(self.clinical_dim)
            clinical[0] += true_class * 0.8  # Age correlation
            clinical[1] -= true_class * 1.2  # MMSE decreases with severity
            
            # Oral microbiome (P. gingivalis increases with AD)
            oral_micro = np.abs(np.random.randn(self.oral_dim))
            oral_micro[0] += true_class * 0.5  # Key species
            
            # Gut microbiome (dysbiosis in AD)
            gut_micro = np.abs(np.random.randn(self.gut_dim))
            gut_micro[:5] += true_class * 0.4  # Multiple species affected
            
            # Blood biomarkers (Aβ42 decreases, tau increases)
            biomarkers = np.random.randn(self.biomarker_dim)
            biomarkers[0] -= true_class * 1.0  # Aβ42 decrease
            biomarkers[1] += true_class * 1.2  # tau increase
            biomarkers[2] += true_class * 1.5  # p-tau increase
            
            dataset.append({
                'clinical': clinical,
                'oral_microbiome': oral_micro,
                'gut_microbiome': gut_micro,
                'blood_biomarkers': biomarkers,
                'label': true_class
            })
        
        return dataset
    
    def reset(self) -> Dict:
        """Reset environment with new patient."""
        # Select random patient
        patient = self.dataset[np.random.randint(len(self.dataset))]
        
        self.true_features = {
            'clinical': patient['clinical'],
            'oral_microbiome': patient['oral_microbiome'],
            'gut_microbiome': patient['gut_microbiome'],
            'blood_biomarkers': patient['blood_biomarkers']
        }
        self.true_label = patient['label']
        
        # Initialize collected features (all zeros)
        self.collected = {
            'clinical': np.zeros(self.clinical_dim),
            'oral_microbiome': np.zeros(self.oral_dim),
            'gut_microbiome': np.zeros(self.gut_dim),
            'blood_biomarkers': np.zeros(self.biomarker_dim)
        }
        
        # Track which modalities are collected
        self.modality_collected = {
            'clinical': False,
            'oral_microbiome': False,
            'gut_microbiome': False,
            'blood_biomarkers': False
        }
        
        self.steps = 0
        self.total_cost = 0
        
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """Get current state representation."""
        modality_mask = np.array([
            float(self.modality_collected['clinical']),
            float(self.modality_collected['oral_microbiome']),
            float(self.modality_collected['gut_microbiome']),
            float(self.modality_collected['blood_biomarkers'])
        ])
        
        meta_features = np.array([
            self.steps / self.max_steps,
            self.total_cost / 250,  # Normalize cost
            np.sum(modality_mask) / 4,  # Fraction collected
            float(self.modality_collected['clinical']),
            float(self.modality_collected['blood_biomarkers'])  # Most important
        ])
        
        return {
            'clinical': self.collected['clinical'].copy(),
            'oral_microbiome': self.collected['oral_microbiome'].copy(),
            'gut_microbiome': self.collected['gut_microbiome'].copy(),
            'blood_biomarkers': self.collected['blood_biomarkers'].copy(),
            'modality_mask': modality_mask,
            'meta_features': meta_features
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute action in environment."""
        reward = 0
        done = False
        info = {}
        
        # Test ordering actions (0-3)
        if action == 0 and not self.modality_collected['clinical']:
            self.collected['clinical'] = self.true_features['clinical'].copy()
            self.modality_collected['clinical'] = True
            reward = -self.test_costs['clinical']
            self.total_cost += self.test_costs['clinical']
            
        elif action == 1 and not self.modality_collected['oral_microbiome']:
            self.collected['oral_microbiome'] = self.true_features['oral_microbiome'].copy()
            self.modality_collected['oral_microbiome'] = True
            reward = -self.test_costs['oral_microbiome']
            self.total_cost += self.test_costs['oral_microbiome']
            
        elif action == 2 and not self.modality_collected['gut_microbiome']:
            self.collected['gut_microbiome'] = self.true_features['gut_microbiome'].copy()
            self.modality_collected['gut_microbiome'] = True
            reward = -self.test_costs['gut_microbiome']
            self.total_cost += self.test_costs['gut_microbiome']
            
        elif action == 3 and not self.modality_collected['blood_biomarkers']:
            self.collected['blood_biomarkers'] = self.true_features['blood_biomarkers'].copy()
            self.modality_collected['blood_biomarkers'] = True
            reward = -self.test_costs['blood_biomarkers']
            self.total_cost += self.test_costs['blood_biomarkers']
        
        # Classification actions (4-6)
        elif action >= 4:
            predicted_class = action - 4
            
            if predicted_class == self.true_label:
                # Correct classification
                reward = 100.0 - self.total_cost * 0.2
                info['correct'] = True
            else:
                # Incorrect classification
                reward = -100.0 - self.total_cost * 0.2
                info['correct'] = False
            
            done = True
            info['predicted'] = predicted_class
            info['true'] = self.true_label
        
        self.steps += 1
        
        # Force termination after max steps
        if self.steps >= self.max_steps and not done:
            reward = -50.0  # Penalty for not deciding
            done = True
        
        next_state = self._get_state()
        
        return next_state, reward, done, info


class MultiModalAgent:
    """
    RL agent for Alzheimer's diagnosis with multi-modal data.
    """
    
    def __init__(
        self,
        clinical_dim: int,
        oral_microbiome_dim: int,
        gut_microbiome_dim: int,
        biomarker_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-network
        self.q_network = MultiModalQNetwork(
            clinical_dim,
            oral_microbiome_dim,
            gut_microbiome_dim,
            biomarker_dim,
            action_dim
        ).to(self.device)
        
        # Target network
        self.target_network = MultiModalQNetwork(
            clinical_dim,
            oral_microbiome_dim,
            gut_microbiome_dim,
            biomarker_dim,
            action_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_target_freq = 10
        self.steps = 0
    
    def _state_to_tensor(self, state: Dict) -> Dict[str, torch.Tensor]:
        """Convert state dict to tensor dict."""
        return {
            'clinical': torch.FloatTensor(state['clinical']).unsqueeze(0).to(self.device),
            'oral_microbiome': torch.FloatTensor(state['oral_microbiome']).unsqueeze(0).to(self.device),
            'gut_microbiome': torch.FloatTensor(state['gut_microbiome']).unsqueeze(0).to(self.device),
            'blood_biomarkers': torch.FloatTensor(state['blood_biomarkers']).unsqueeze(0).to(self.device),
            'modality_mask': torch.FloatTensor(state['modality_mask']).unsqueeze(0).to(self.device),
            'meta_features': torch.FloatTensor(state['meta_features']).unsqueeze(0).to(self.device)
        }
    
    def select_action(self, state: Dict, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors (batch processing)
        state_batch = {
            'clinical': torch.FloatTensor([s['clinical'] for s in states]).to(self.device),
            'oral_microbiome': torch.FloatTensor([s['oral_microbiome'] for s in states]).to(self.device),
            'gut_microbiome': torch.FloatTensor([s['gut_microbiome'] for s in states]).to(self.device),
            'blood_biomarkers': torch.FloatTensor([s['blood_biomarkers'] for s in states]).to(self.device),
            'modality_mask': torch.FloatTensor([s['modality_mask'] for s in states]).to(self.device),
            'meta_features': torch.FloatTensor([s['meta_features'] for s in states]).to(self.device)
        }
        
        next_state_batch = {
            'clinical': torch.FloatTensor([s['clinical'] for s in next_states]).to(self.device),
            'oral_microbiome': torch.FloatTensor([s['oral_microbiome'] for s in next_states]).to(self.device),
            'gut_microbiome': torch.FloatTensor([s['gut_microbiome'] for s in next_states]).to(self.device),
            'blood_biomarkers': torch.FloatTensor([s['blood_biomarkers'] for s in next_states]).to(self.device),
            'modality_mask': torch.FloatTensor([s['modality_mask'] for s in next_states]).to(self.device),
            'meta_features': torch.FloatTensor([s['meta_features'] for s in next_states]).to(self.device)
        }
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(state_batch).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_state_batch).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, env: AlzheimerEnv, episodes: int = 10000, verbose: bool = True):
        """Train agent on environment."""
        episode_rewards = []
        episode_accuracy = []
        episode_costs = []
        
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
            
            episode_rewards.append(total_reward)
            episode_costs.append(env.total_cost)
            if correct is not None:
                episode_accuracy.append(1.0 if correct else 0.0)
            
            if verbose and (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                avg_accuracy = np.mean(episode_accuracy[-1000:]) if episode_accuracy else 0
                avg_cost = np.mean(episode_costs[-1000:])
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Accuracy: {avg_accuracy:.2%}")
                print(f"  Avg Cost: ${avg_cost:.2f}")
                print(f"  Epsilon: {self.epsilon:.3f}\n")
        
        return episode_rewards, episode_accuracy, episode_costs


if __name__ == "__main__":
    print("=" * 70)
    print("Training Multi-Modal RL Agent for Alzheimer's Disease Classification")
    print("=" * 70)
    print()
    
    # Create environment
    env = AlzheimerEnv(
        clinical_dim=10,
        oral_microbiome_dim=20,
        gut_microbiome_dim=30,
        biomarker_dim=15,
        num_classes=3
    )
    
    print(f"Environment Configuration:")
    print(f"  Clinical features: {env.clinical_dim}")
    print(f"  Oral microbiome features: {env.oral_dim}")
    print(f"  Gut microbiome features: {env.gut_dim}")
    print(f"  Blood biomarker features: {env.biomarker_dim}")
    print(f"  Action space: {env.action_dim}")
    print(f"  Classes: CN, MCI, AD")
    print()
    
    # Create agent
    agent = MultiModalAgent(
        clinical_dim=env.clinical_dim,
        oral_microbiome_dim=env.oral_dim,
        gut_microbiome_dim=env.gut_dim,
        biomarker_dim=env.biomarker_dim,
        action_dim=env.action_dim,
        learning_rate=0.0001,
        gamma=0.95
    )
    
    print("Training agent...")
    print()
    
    # Train agent
    rewards, accuracy, costs = agent.train(env, episodes=10000, verbose=True)
    
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Accuracy: {np.mean(accuracy[-1000:]):.2%}")
    print(f"Final Avg Cost: ${np.mean(costs[-1000:]):.2f}")
    print(f"Cost-Effectiveness: {np.mean(accuracy[-1000:]) / np.mean(costs[-1000:]):.4f}")
