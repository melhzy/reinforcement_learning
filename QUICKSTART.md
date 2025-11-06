# Quick Start Guide

Get started with the RL Framework for Alzheimer's Disease Research in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- No external dependencies required for basic functionality

## Installation

```bash
# Clone the repository
git clone https://github.com/melhzy/reinforcement_learning.git
cd reinforcement_learning

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Your First Training Run

### Step 1: Run the Basic Example

```bash
python examples/basic_example.py
```

This will:
- Train an agent for 100 episodes
- Save checkpoints every 10 episodes
- Evaluate the trained agent
- Export results to `training_results.json`

Expected output:
```
============================================================
Alzheimer's Research RL Framework - Basic Example
============================================================
✓ Configuration loaded
✓ Environment created
✓ LLM Agent created
✓ Medical reward function created
✓ Trainer created

Starting Training...
```

### Step 2: Understand the Results

After training completes, you'll see:
- **Training Summary**: Total episodes, steps, rewards
- **Evaluation Results**: Performance metrics on test episodes
- **Checkpoints**: Saved in `./checkpoints/` directory

### Step 3: Try Custom Configuration

Create a custom configuration file:

```python
from rl_framework import LLMAgent, AlzheimersResearchEnv, MedicalRewardFunction, RLTrainer
from rl_framework.configs import get_default_config, merge_configs

# Load default config
base_config = get_default_config()

# Override specific settings
custom_config = {
    'training': {
        'num_episodes': 50,  # Shorter training
        'save_frequency': 5   # More frequent saves
    },
    'agent': {
        'temperature': 0.5,      # More focused actions
        'learning_rate': 0.0005  # Slower learning
    }
}

# Merge configurations
config = merge_configs(base_config, custom_config)

# Create and train
env = AlzheimersResearchEnv(config['environment'])
agent = LLMAgent(config['agent'])
reward_fn = MedicalRewardFunction(config['reward_function'])

trainer = RLTrainer(agent, env, reward_fn, config['training'])
results = trainer.train()
```

## Common Tasks

### Generate Synthetic Patient Data

```python
from rl_framework.utils import DataProcessor

# Generate 100 synthetic patients
dataset = DataProcessor.generate_synthetic_dataset(
    num_patients=100,
    disease_distribution={
        "healthy": 0.4,
        "mild": 0.3,
        "moderate": 0.2,
        "severe": 0.1
    }
)

# Save dataset
DataProcessor.save_dataset(dataset, "my_patients.json")

# Calculate risk scores
for patient in dataset[:5]:  # First 5 patients
    risk = DataProcessor.calculate_risk_score(patient)
    print(f"{patient['patient_id']}: Risk = {risk:.2f}")
```

### Evaluate a Trained Agent

```python
# Load a checkpoint
trainer.load_checkpoint("ep50")  # Load episode 50 checkpoint

# Evaluate
eval_results = trainer.evaluate(num_episodes=10, render=True)

print(f"Average Reward: {eval_results['average_reward']:.2f}")
print(f"Std Deviation: {eval_results['std_reward']:.2f}")
```

### Analyze Reward Breakdown

```python
from rl_framework import MedicalRewardFunction

# Create reward function
reward_fn = MedicalRewardFunction({
    "safety_weight": 0.3,
    "validity_weight": 0.3,
    "impact_weight": 0.2,
    "ethics_weight": 0.2
})

# Simulate a state and action
state = {
    'patient_data': {
        'age': 70,
        'disease_stage': 'mild',
        'contraindications': []
    }
}

action = {
    'action_type': 'design_experiment',
    'confidence': 0.85,
    'reasoning': 'Well-designed protocol',
    'parameters': {
        'sample_size': 100,
        'duration_weeks': 12
    }
}

# Get detailed breakdown
breakdown = reward_fn.get_reward_breakdown(action, state, state)

print("Reward Breakdown:")
for component in ['safety', 'validity', 'impact', 'ethics']:
    print(f"  {component.capitalize()}: {breakdown[component]:.3f}")
print(f"  Total: {breakdown['total']:.3f}")
```

### Visualize Training Progress

```python
import json

# Load training history
with open('training_results.json', 'r') as f:
    results = json.load(f)

# Extract rewards per episode
episode_rewards = [ep['total_reward'] for ep in results['history']]

# Print progress
print("Training Progress:")
for i, reward in enumerate(episode_rewards[::10], 1):  # Every 10th episode
    print(f"  Episode {i*10}: {reward:.2f}")
```

## Understanding the Output

### Training Logs

```
Episode 1/100
  Steps: 50
  Total Reward: 41.55
  Average Reward: 0.83
  Actions: analyze_biomarkers, suggest_treatment, ...
  Moving Avg (10 eps): 41.79
```

- **Steps**: Number of actions taken in the episode
- **Total Reward**: Sum of all rewards in the episode
- **Average Reward**: Total reward / steps
- **Actions**: First few actions taken by the agent
- **Moving Avg**: Average reward over last 10 episodes

### Checkpoint Files

After training, you'll find:
- `checkpoints/agent_ep10.json`: Agent state at episode 10
- `checkpoints/history_ep10.json`: Training history up to episode 10
- `checkpoints/agent_final.json`: Final trained agent
- `checkpoints/history_final.json`: Complete training history

## Next Steps

1. **Experiment with Parameters**: Try different learning rates, temperatures, reward weights
2. **Create Custom Environments**: Extend `BaseEnvironment` for other medical domains
3. **Implement Custom Agents**: Extend `BaseAgent` with your own learning algorithms
4. **Integrate Real LLMs**: Add OpenAI or Anthropic API integration
5. **Advanced Analysis**: Implement visualization and analysis tools

## Troubleshooting

### Issue: Training is too slow
**Solution**: Reduce `num_episodes` or `max_steps_per_episode` in config

### Issue: Agent always selects the same action
**Solution**: Increase `temperature` parameter for more exploration

### Issue: Rewards are very low
**Solution**: Check reward function weights, adjust for your specific goals

### Issue: Memory usage is high
**Solution**: Reduce `memory_size` in agent config

## Getting Help

- Read the full documentation in `README.md`
- Check `ARCHITECTURE.md` for design details
- Review examples in `examples/` directory
- Run tests: `python -m unittest discover tests`
- Open an issue on GitHub

## Tips for Success

1. **Start Small**: Begin with short training runs (10-20 episodes) to test
2. **Monitor Progress**: Check the moving average to see if agent is learning
3. **Save Often**: Use frequent checkpointing during experimentation
4. **Validate Results**: Always evaluate on separate test episodes
5. **Document Changes**: Keep track of configuration changes and results

Happy researching! 🧠🔬
