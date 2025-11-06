# Tutorial 3: Policy Gradient Methods

## Overview

Policy Gradient methods directly optimize the policy (action selection strategy) rather than learning value functions. This approach is particularly useful for continuous action spaces and stochastic policies.

## Key Concepts

### Value-Based vs Policy-Based RL

**Value-Based (Q-Learning, DQN):**
- Learn Q(s,a) → derive policy
- Indirect: Policy is implicit
- Works well for discrete actions

**Policy-Based (Policy Gradient):**
- Learn π(a|s) directly
- Direct: Policy is explicit
- Works for continuous and stochastic policies

## REINFORCE Algorithm

The classic policy gradient algorithm:

```
∇J(θ) = E[∇log π_θ(a|s) * G_t]
```

Where:
- θ: Policy parameters
- G_t: Return (cumulative reward)
- π_θ(a|s): Policy probability

### Algorithm Steps

1. Initialize policy network π_θ
2. For each episode:
   - Generate trajectory using π_θ
   - Calculate returns G_t
   - Update θ using policy gradient

## Actor-Critic Methods

Combine policy gradients with value estimation:
- **Actor**: Policy network π_θ(a|s)
- **Critic**: Value network V_φ(s)

Advantages:
- Lower variance
- More stable training
- Better sample efficiency

## Medical Application: Treatment Optimization

### Scenario: Alzheimer's Disease Progression

**State Space:**
- Clinical scores (MMSE, CDR)
- Biomarker levels (Aβ42, tau, p-tau)
- Microbiome composition
- Patient demographics

**Action Space (Continuous):**
- Medication dosage
- Cognitive training intensity
- Lifestyle intervention parameters

**Reward:**
- Cognitive improvement: +reward
- Side effects: -penalty
- Disease progression: -penalty

## Implementation

See `policy_gradient.py` for REINFORCE and Actor-Critic implementations.

## Example Usage

```python
from policy_gradient import REINFORCEAgent

# Create agent
agent = REINFORCEAgent(
    state_dim=10,
    action_dim=3,
    learning_rate=0.001
)

# Training
for episode in range(num_episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Update policy after episode
    agent.update(states, actions, rewards)
```

## Advanced Topics

### Advantage Actor-Critic (A2C)
Use advantage function A(s,a) = Q(s,a) - V(s) to reduce variance.

### Proximal Policy Optimization (PPO)
Constrain policy updates to prevent large changes:
- More stable training
- Better performance
- Industry standard for many applications

### Trust Region Policy Optimization (TRPO)
Theoretical guarantees on policy improvement.

## Exercise

Implement a policy gradient agent for optimizing diagnostic test ordering in Alzheimer's screening.

## Next Steps

Proceed to Tutorial 4: Medical RL Applications to see domain-specific implementations.
