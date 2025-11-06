"""
Q-Learning Implementation
Basic reinforcement learning using tabular Q-learning
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    Q-Learning agent for discrete state and action spaces.
    
    Attributes:
        state_size (int): Number of possible states
        action_size (int): Number of possible actions
        learning_rate (float): Learning rate (alpha)
        discount_factor (float): Discount factor (gamma)
        epsilon (float): Exploration rate
        epsilon_decay (float): Decay rate for epsilon
        epsilon_min (float): Minimum epsilon value
        q_table (np.ndarray): Q-value table
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_size)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, episodes: int = 1000, verbose: bool = True) -> List[float]:
        """
        Train the agent on an environment.
        
        Args:
            env: Environment to train on
            episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            List of episode rewards
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Select and perform action
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            # Decay exploration rate
            self.decay_epsilon()
            episode_rewards.append(total_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)


class GridWorld:
    """
    Simple GridWorld environment for demonstration.
    
    The agent navigates a grid to reach a goal while avoiding obstacles.
    """
    
    def __init__(self, size: int = 5, goal_reward: float = 100.0, step_penalty: float = -1.0):
        self.size = size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # State space: flattened grid positions
        self.state_size = size * size
        # Action space: up, down, left, right
        self.action_size = 4
        
        # Goal position (bottom-right corner)
        self.goal_pos = (size - 1, size - 1)
        
        # Obstacles (example positions)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        self.current_pos = None
        self.reset()
    
    def reset(self) -> int:
        """Reset environment to starting position."""
        self.current_pos = (0, 0)  # Start at top-left
        return self._pos_to_state(self.current_pos)
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert 2D position to state index."""
        return pos[0] * self.size + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to 2D position."""
        return (state // self.size, state % self.size)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            next_state, reward, done, info
        """
        row, col = self.current_pos
        
        # Apply action
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.size - 1, col + 1)
        
        new_pos = (row, col)
        
        # Check if hit obstacle (return to previous position)
        if new_pos in self.obstacles:
            new_pos = self.current_pos
            reward = self.step_penalty * 2  # Penalty for hitting obstacle
        elif new_pos == self.goal_pos:
            reward = self.goal_reward
        else:
            reward = self.step_penalty
        
        self.current_pos = new_pos
        done = (new_pos == self.goal_pos)
        
        next_state = self._pos_to_state(new_pos)
        
        return next_state, reward, done, {}
    
    def render(self):
        """Visualize the current state."""
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        # Mark goal
        grid[self.goal_pos] = 'G'
        
        # Mark agent
        grid[self.current_pos] = 'A'
        
        print('\n'.join([''.join(row) for row in grid]))


def plot_training_results(rewards: List[float], window: int = 100):
    """Plot training rewards over episodes."""
    plt.figure(figsize=(10, 5))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (window={window})')
    plt.title('Moving Average Rewards')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training results saved to 'training_results.png'")


if __name__ == "__main__":
    # Example usage
    print("Training Q-Learning Agent on GridWorld...")
    
    # Create environment
    env = GridWorld(size=5)
    
    # Create agent
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    # Train agent
    rewards = agent.train(env, episodes=1000, verbose=True)
    
    # Plot results
    plot_training_results(rewards)
    
    # Test trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    env.render()
    print()
    
    for step in range(20):
        action = agent.get_action(state, training=False)
        next_state, reward, done, _ = env.step(action)
        env.render()
        print(f"Step {step + 1}, Reward: {reward}")
        print()
        
        if done:
            print("Goal reached!")
            break
        
        state = next_state
