# Tutorial 4: RL for Medical Classification

## Overview

This tutorial demonstrates how to apply reinforcement learning to medical classification tasks, focusing on diagnostic decision-making and feature selection.

## Medical Classification as RL

### Traditional Approach
- Supervised learning: X → Y
- Fixed feature set
- Single-step classification

### RL Approach
- Sequential decision-making
- Adaptive feature selection
- Cost-aware diagnosis

## Key Advantages of RL in Medical Settings

### 1. Cost-Aware Decision Making
Not all medical tests have equal cost:
- Blood tests: Low cost
- MRI scans: High cost
- Genetic sequencing: Very high cost

RL can learn to:
- Order cheap tests first
- Only request expensive tests when necessary
- Balance diagnostic accuracy vs. cost

### 2. Sequential Diagnosis
Real-world diagnosis is iterative:
1. Initial assessment
2. Order tests based on results
3. Refine diagnosis
4. Request additional tests if needed

### 3. Uncertainty Handling
RL naturally handles uncertainty:
- Exploration: Try different diagnostic strategies
- Exploitation: Use proven effective approaches
- Learning from mistakes

## Application: Alzheimer's Disease Classification

### Problem Setup

**Goal**: Classify patients into categories:
- Cognitively Normal (CN)
- Mild Cognitive Impairment (MCI)
- Alzheimer's Disease (AD)

**Available Features:**
- **Clinical** (Low cost):
  - Age, education, family history
  - MMSE score, cognitive tests
  
- **Biomarkers** (Medium cost):
  - CSF: Aβ42, tau, p-tau
  - Blood: Various markers
  
- **Microbiome** (Medium-High cost):
  - Oral bacteria composition
  - Gut microbiome profile
  
- **Imaging** (High cost):
  - MRI: Brain volume, hippocampal atrophy
  - PET: Amyloid/tau imaging

### State Space
Current information about the patient:
- Features collected so far
- Uncertainty about diagnosis
- Cost spent

### Action Space
- **Test Actions**: Order specific tests
- **Classification Actions**: Make diagnosis (CN/MCI/AD)

### Reward Function
```python
def reward_function(action, outcome):
    if action is classification:
        if correct:
            return +10 - cost_spent * penalty
        else:
            return -10 - cost_spent * penalty
    else:  # test action
        return -test_cost
```

## Implementation

See `medical_classification_rl.py` for a complete implementation with:
- Feature selection using RL
- Cost-aware diagnostic strategies
- Multi-modal data integration

## Example Usage

```python
from medical_classification_rl import MedicalDiagnosisEnv, DiagnosticAgent

# Create environment with real medical data structure
env = MedicalDiagnosisEnv(
    num_clinical_features=10,
    num_biomarker_features=5,
    num_microbiome_features=20,
    num_classes=3
)

# Create agent
agent = DiagnosticAgent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    learning_rate=0.001
)

# Train
agent.train(env, episodes=5000)

# Evaluate
accuracy, avg_cost = evaluate_agent(agent, env)
print(f"Accuracy: {accuracy:.2%}, Avg Cost: ${avg_cost:.2f}")
```

## Evaluation Metrics

### Accuracy
- Classification correctness
- Per-class performance

### Efficiency
- Number of tests ordered
- Total diagnostic cost
- Time to diagnosis

### Cost-Effectiveness
```
Cost-Effectiveness = Accuracy / Average_Cost
```

## Advanced Topics

### Multi-Task RL
- Diagnose multiple conditions simultaneously
- Transfer learning across diseases

### Hierarchical RL
- High-level: Choose test category
- Low-level: Select specific test

### Offline RL
- Learn from historical medical records
- No patient interaction during training

## Exercise

Implement an RL agent that:
1. Starts with clinical features only
2. Selectively orders additional tests
3. Achieves >85% accuracy with minimal cost

## Next Steps

Proceed to Tutorial 5: Alzheimer's Analysis with Multi-Modal Data for a comprehensive application.
