# Reinforcement Learning Tutorials for Medical AI

A comprehensive repository of reinforcement learning tutorials designed to support Chain of Thought (CoT) fine-tuning for LLM-based Alzheimer's disease analysis.

## Overview

This repository provides hands-on tutorials covering fundamental to advanced reinforcement learning concepts, with a specific focus on medical applications, particularly Alzheimer's disease classification using:
- Clinical variables
- Oral and gut microbiome data
- Blood biomarkers

## Repository Structure

```
reinforcement_learning/
├── tutorials/
│   ├── 01_basic_rl/           # Introduction to Q-Learning
│   ├── 02_deep_qn/            # Deep Q-Networks (DQN)
│   ├── 03_policy_gradient/    # Policy Gradient Methods
│   ├── 04_medical_rl/         # RL for Medical Classification
│   ├── 05_alzheimer_analysis/ # RL with Medical Data
│   └── 06_cot_integration/    # CoT Integration for LLM
├── datasets/                   # Example datasets
├── utils/                      # Utility functions
└── requirements.txt           # Project dependencies
```

## Features

### Core RL Tutorials
1. **Basic Q-Learning**: Introduction to value-based methods
2. **Deep Q-Networks**: Neural network-based Q-learning
3. **Policy Gradient Methods**: REINFORCE and Actor-Critic algorithms

### Medical AI Applications
4. **Medical Classification with RL**: Applying RL to diagnostic tasks
5. **Alzheimer's Analysis**: Multi-modal feature integration
   - Clinical variables (age, MMSE scores, cognitive assessments)
   - Microbiome data (oral and gut bacterial composition)
   - Blood biomarkers (Aβ42, tau, p-tau levels)
6. **CoT Fine-tuning**: Integration with LLM training pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/melhzy/reinforcement_learning.git
cd reinforcement_learning

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Example: Basic Q-Learning
from tutorials.basic_rl.q_learning import QLearningAgent

agent = QLearningAgent(state_size=4, action_size=2)
# Train your agent...
```

## Use Cases

### For Alzheimer's Disease Research
- **Feature Selection**: Use RL to identify most informative biomarkers
- **Classification**: Train diagnostic models with multi-modal data
- **Treatment Optimization**: Model treatment response prediction

### For LLM Fine-tuning
- **CoT Generation**: Generate reasoning chains for medical decisions
- **Reward Modeling**: Train reward models for medical text generation
- **RLHF Integration**: Reinforcement Learning from Human Feedback

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{reinforcement_learning_tutorials,
  author = {Huang, Ziyuan},
  title = {Reinforcement Learning Tutorials for Medical AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/melhzy/reinforcement_learning}
}
```

## Acknowledgments

This repository is developed to support research in Alzheimer's disease analysis using machine learning and large language models.