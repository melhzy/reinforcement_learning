# Tutorial 1: Introduction to Q-Learning

## Overview

This tutorial introduces the fundamental concepts of Reinforcement Learning through Q-Learning, a value-based method that learns optimal action-value functions.

## What is Q-Learning?

Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking a specific action in a given state. The "Q" stands for Quality, representing how good an action is in a particular state.

### Key Concepts

1. **State (s)**: Current situation of the agent
2. **Action (a)**: Possible moves the agent can take
3. **Reward (r)**: Feedback from environment
4. **Q-Value Q(s,a)**: Expected future reward for action a in state s

### The Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- α (alpha): Learning rate
- γ (gamma): Discount factor
- s': Next state
- a': Next action

## Implementation

See `q_learning.py` for a complete implementation.

## Example: GridWorld Navigation

The agent learns to navigate a grid to reach a goal while avoiding obstacles.

```python
from q_learning import QLearningAgent, GridWorld

# Create environment
env = GridWorld(size=5)

# Create agent
agent = QLearningAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    learning_rate=0.1,
    discount_factor=0.95
)

# Train
agent.train(env, episodes=1000)
```

## Medical Application Context

In medical settings, Q-Learning can be used for:
- **Treatment selection**: Learning optimal treatment sequences
- **Diagnostic pathways**: Determining which tests to order
- **Resource allocation**: Optimizing hospital resource usage

For Alzheimer's analysis:
- States: Patient biomarker profiles
- Actions: Diagnostic tests to perform
- Rewards: Diagnostic accuracy improvements

## Exercise

Implement a simple Q-Learning agent for feature selection in a medical classification task.

## Next Steps

Proceed to Tutorial 2: Deep Q-Networks to learn how to scale Q-Learning with neural networks.
