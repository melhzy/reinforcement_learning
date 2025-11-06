"""
Policy Gradient Methods Implementation
REINFORCE and Actor-Critic algorithms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action logits."""
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Neural network for value function approximation."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value."""
        return self.network(state).squeeze()


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dims: List[int] = [128, 128]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select action from policy.
        
        Returns:
            action, log_prob
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy.get_action_probs(state_tensor)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns."""
        returns = []
        G = 0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns.tolist()
    
    def update(self, states: List[np.ndarray], actions: List[int], rewards: List[float]) -> float:
        """
        Update policy using REINFORCE algorithm.
        
        Returns:
            Loss value
        """
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Compute action log probabilities
        action_probs = self.policy.get_action_probs(states_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)
        
        # Compute policy gradient loss
        loss = -(log_probs * returns_tensor).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))


class ActorCriticAgent:
    """
    Actor-Critic agent with separate policy and value networks.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        hidden_dims: List[int] = [128, 128]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor (policy) network
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic (value) network
        self.critic = ValueNetwork(state_dim, hidden_dims).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor.get_action_probs(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[float, float]:
        """
        Update actor and critic networks.
        
        Returns:
            actor_loss, critic_loss
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Compute TD target
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
            td_target = reward_tensor + (1 - done_tensor) * self.gamma * next_value
        
        # Compute TD error (advantage)
        current_value = self.critic(state_tensor)
        td_error = td_target - current_value
        
        # Update critic
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor.get_action_probs(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action_tensor)
        
        actor_loss = -(log_prob * td_error.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class MedicalTreatmentEnv:
    """
    Simple medical treatment environment for demonstration.
    
    Agent decides treatment intensity for patients with different severity levels.
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim  # Low, Medium, High treatment intensity
        
        self.current_state = None
        self.true_severity = None
    
    def reset(self) -> np.ndarray:
        """Reset environment with new patient."""
        # Generate patient features
        self.true_severity = np.random.choice([0, 1, 2])  # Mild, Moderate, Severe
        
        # Generate features correlated with severity
        features = np.random.randn(self.state_dim)
        features[0] = self.true_severity + np.random.randn() * 0.5
        
        self.current_state = features
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action (select treatment intensity).
        
        Reward: Higher for matching treatment to severity
        """
        # Optimal action matches severity level
        if action == self.true_severity:
            reward = 1.0  # Correct treatment intensity
        else:
            reward = -0.5 * abs(action - self.true_severity)  # Penalty for mismatch
        
        done = True  # One-step episodes for simplicity
        
        return self.current_state, reward, done


def train_reinforce(env, agent: REINFORCEAgent, num_episodes: int = 1000) -> List[float]:
    """Train REINFORCE agent."""
    episode_rewards = []
    
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Update policy after episode
        loss = agent.update(states, actions, rewards)
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
    
    return episode_rewards


def train_actor_critic(env, agent: ActorCriticAgent, num_episodes: int = 1000) -> List[float]:
    """Train Actor-Critic agent."""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update networks
            actor_loss, critic_loss = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    print("=" * 60)
    print("Training REINFORCE Agent")
    print("=" * 60)
    
    env = MedicalTreatmentEnv(state_dim=5, action_dim=3)
    reinforce_agent = REINFORCEAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001
    )
    
    reinforce_rewards = train_reinforce(env, reinforce_agent, num_episodes=1000)
    
    print("\n" + "=" * 60)
    print("Training Actor-Critic Agent")
    print("=" * 60)
    
    env = MedicalTreatmentEnv(state_dim=5, action_dim=3)
    ac_agent = ActorCriticAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        actor_lr=0.001,
        critic_lr=0.001
    )
    
    ac_rewards = train_actor_critic(env, ac_agent, num_episodes=1000)
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"REINFORCE - Final Avg Reward: {np.mean(reinforce_rewards[-100:]):.2f}")
    print(f"Actor-Critic - Final Avg Reward: {np.mean(ac_rewards[-100:]):.2f}")
