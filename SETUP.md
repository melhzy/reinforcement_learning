# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for accelerated training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/melhzy/reinforcement_learning.git
cd reinforcement_learning
```

### 2. Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n rl_tutorials python=3.9
conda activate rl_tutorials
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import numpy; import pandas; print('Installation successful!')"
```

## Quick Start

### Running Tutorial 1: Basic Q-Learning

```bash
cd tutorials/01_basic_rl
python q_learning.py
```

Expected output:
- Training progress every 100 episodes
- Final trained agent demonstration
- Saved training plot

### Running Tutorial 2: Deep Q-Networks

```bash
cd tutorials/02_deep_qn
python dqn.py
```

### Running Tutorial 5: Alzheimer's Multi-Modal Analysis

```bash
cd tutorials/05_alzheimer_analysis
python alzheimer_multimodal_rl.py
```

## Tutorial Structure

Each tutorial contains:
- `README.md`: Theoretical background and explanations
- `*.py`: Implementation code with examples
- Runnable examples with synthetic data

## Working with Real Data

### Loading Your Own Data

```python
import pandas as pd

# Load your dataset
data = pd.read_csv('path/to/your/data.csv')

# Prepare features
features = data[['feature1', 'feature2', ...]].values
labels = data['label'].values

# Use with RL agents (example)
from tutorials.medical_rl import DiagnosticAgent
agent = DiagnosticAgent(state_dim=features.shape[1], ...)
```

### Data Format Requirements

For Alzheimer's classification:
- **Clinical features**: Continuous values (age, MMSE, CDR, etc.)
- **Biomarkers**: Continuous values (Aβ42, tau, p-tau, etc.)
- **Microbiome**: Compositional data (use CLR transformation)
- **Labels**: Integer encoding (0=CN, 1=MCI, 2=AD)

## GPU Support

The code automatically detects and uses GPU if available:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

To force CPU usage:
```python
device = torch.device('cpu')
model.to(device)
```

## Troubleshooting

### Common Issues

**Issue**: ImportError for torch
```bash
# Solution: Install PyTorch
pip install torch torchvision
```

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or use CPU
# In Python:
batch_size = 16  # Reduce from 32
```

**Issue**: Slow training
```bash
# Solution: Enable GPU or reduce model size
# Check GPU:
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

If you encounter issues:
1. Check the tutorial README for specific guidance
2. Review the example code in each tutorial
3. Open an issue on GitHub with:
   - Python version
   - Package versions (`pip list`)
   - Error message and traceback

## Development Setup

For contributing or modifying the code:

### Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black tutorials/ utils/
flake8 tutorials/ utils/
```

## Performance Optimization

### Training Speed

1. **Use GPU**: Significant speedup for neural networks
2. **Batch Processing**: Increase batch size for better GPU utilization
3. **Parallel Environments**: Run multiple environments in parallel

```python
# Example: Parallel training
from multiprocessing import Pool

def train_agent(seed):
    # Training code with different seed
    pass

with Pool(4) as p:
    results = p.map(train_agent, range(4))
```

### Memory Usage

1. **Replay Buffer Size**: Reduce if memory is limited
2. **Network Size**: Use smaller hidden dimensions
3. **Gradient Accumulation**: Split large batches

## Advanced Configuration

### Custom Hyperparameters

```python
# Example: Custom DQN configuration
agent = DQNAgent(
    state_dim=10,
    action_dim=3,
    learning_rate=0.0001,      # Lower for stability
    gamma=0.99,                # Discount factor
    epsilon=1.0,               # Initial exploration
    epsilon_decay=0.995,       # Exploration decay
    batch_size=64,             # Training batch size
    buffer_capacity=100000,    # Replay buffer size
    target_update_freq=100     # Target network update
)
```

### Logging and Monitoring

```python
# Enable TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for episode in range(num_episodes):
    # Training code...
    writer.add_scalar('Reward/train', reward, episode)
    writer.add_scalar('Loss/train', loss, episode)

writer.close()

# View in TensorBoard
# tensorboard --logdir=runs
```

## Jupyter Notebooks

To use tutorials in Jupyter:

```bash
pip install jupyter
jupyter notebook
```

Then create a new notebook and import:
```python
import sys
sys.path.append('..')

from tutorials.basic_rl.q_learning import QLearningAgent
from utils import plot_training_curve
```

## Docker Setup (Optional)

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "tutorials/01_basic_rl/q_learning.py"]
```

Build and run:
```bash
docker build -t rl-tutorials .
docker run rl-tutorials
```

## Next Steps

1. Start with Tutorial 1 for basics
2. Progress through tutorials sequentially
3. Experiment with hyperparameters
4. Apply to your own medical datasets
5. Explore CoT integration for LLM fine-tuning

## Resources

- [Reinforcement Learning Book](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Citation

```bibtex
@misc{reinforcement_learning_tutorials,
  author = {Huang, Ziyuan},
  title = {Reinforcement Learning Tutorials for Medical AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/melhzy/reinforcement_learning}
}
```
