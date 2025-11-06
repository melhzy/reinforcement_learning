# Tutorial 2: Deep Q-Networks (DQN)

## Overview

Deep Q-Networks extend Q-Learning by using neural networks to approximate Q-values, enabling RL to work with continuous and high-dimensional state spaces.

## Why DQN?

Traditional Q-Learning uses a table to store Q-values, which becomes impractical when:
- State space is continuous (e.g., sensor readings, biomarker levels)
- State space is high-dimensional (e.g., medical images, genomic data)

DQN uses a neural network to approximate Q(s,a) for any state.

## Key Innovations

### 1. Experience Replay
Store past experiences and sample randomly for training:
- Breaks correlation between consecutive samples
- Improves sample efficiency
- Enables learning from rare events

### 2. Target Network
Use a separate network for computing target Q-values:
- Stabilizes training
- Reduces oscillations
- Updated periodically

## Architecture

```
State → Neural Network → Q-values for all actions
```

For medical applications:
```
Patient Features → DQN → Q-values for diagnostic/treatment actions
```

## Medical Application: Alzheimer's Biomarker Analysis

### State Space
Multi-modal patient features:
- Clinical: Age, MMSE score, education level
- Microbiome: Abundance of key bacterial species
- Biomarkers: Aβ42, tau, p-tau levels

### Action Space
- Order additional tests
- Classify as AD/MCI/Healthy
- Recommend intervention

### Reward
- Correct classification: +1
- Incorrect classification: -1
- Unnecessary test: -0.1

## Implementation

See `dqn.py` for a complete PyTorch implementation.

## Example Usage

```python
from dqn import DQNAgent
import numpy as np

# Define state and action dimensions
state_dim = 10  # e.g., 10 biomarkers
action_dim = 3   # e.g., 3 classification categories

# Create agent
agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=0.001
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        
        state = next_state
```

## Advanced Topics

- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage functions
- **Prioritized Experience Replay**: Sample important transitions more frequently

## Exercise

Implement a DQN agent for classifying Alzheimer's patients using synthetic biomarker data.

## Next Steps

Proceed to Tutorial 3: Policy Gradient Methods for an alternative approach to RL.
