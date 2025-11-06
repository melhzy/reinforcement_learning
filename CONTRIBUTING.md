# Contributing Guide

Thank you for your interest in contributing to this Reinforcement Learning Tutorials repository! This guide will help you get started.

## Ways to Contribute

1. **Report Bugs**: Found an issue? Open a GitHub issue
2. **Suggest Enhancements**: Ideas for new tutorials or improvements
3. **Submit Code**: Fix bugs or add new features
4. **Improve Documentation**: Clarify explanations or add examples
5. **Share Results**: Contribute your training results or insights

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/reinforcement_learning.git
cd reinforcement_learning
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy jupyter
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Good: Clear function names and docstrings
def compute_reward(state: np.ndarray, action: int) -> float:
    """
    Compute reward for state-action pair.
    
    Args:
        state: Current state array
        action: Action taken
        
    Returns:
        Reward value
    """
    return reward_value

# Good: Type hints
def train_agent(
    agent: RLAgent,
    env: Environment,
    num_episodes: int = 1000
) -> List[float]:
    ...

# Good: Clear variable names
episode_rewards = []
total_steps = 0
learning_rate = 0.001

# Avoid: Unclear abbreviations
er = []
ts = 0
lr = 0.001
```

### Code Formatting

Run before committing:

```bash
# Format code
black tutorials/ utils/

# Check style
flake8 tutorials/ utils/

# Type checking (optional)
mypy tutorials/ utils/
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Can span multiple lines
    and include implementation details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = example_function(5, "test")
        >>> print(result)
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return True
```

## Adding New Tutorials

### Tutorial Structure

Each tutorial should include:

```
tutorials/XX_tutorial_name/
├── README.md           # Theory and explanations
├── algorithm.py        # Main implementation
├── example.py         # Usage examples (optional)
└── tests/             # Unit tests (optional)
    └── test_algorithm.py
```

### Tutorial README Template

```markdown
# Tutorial X: Title

## Overview
Brief description of what this tutorial covers.

## Theory
Explain the algorithm/concept with equations if needed.

## Implementation
Code walkthrough and key design decisions.

## Example Usage
```python
# Code example
```

## Exercises
Suggested exercises for learners.

## Next Steps
Link to next tutorial.
```

### Code Template

```python
"""
Module docstring explaining the algorithm.
"""

import numpy as np
from typing import List, Tuple


class YourAlgorithm:
    """
    Brief description of the algorithm.
    """
    
    def __init__(self, param1: int, param2: float = 0.1):
        """Initialize algorithm."""
        self.param1 = param1
        self.param2 = param2
    
    def train(self, data: np.ndarray, epochs: int = 100) -> List[float]:
        """
        Train the algorithm.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            
        Returns:
            List of training metrics
        """
        # Implementation
        pass


if __name__ == "__main__":
    # Example usage
    print("Running example...")
    # Example code
```

## Testing

### Writing Tests

```python
# tests/test_your_algorithm.py
import pytest
import numpy as np
from tutorials.your_tutorial.algorithm import YourAlgorithm


def test_initialization():
    """Test algorithm initialization."""
    algo = YourAlgorithm(param1=10)
    assert algo.param1 == 10


def test_training():
    """Test training process."""
    algo = YourAlgorithm(param1=10)
    data = np.random.randn(100, 10)
    metrics = algo.train(data, epochs=10)
    assert len(metrics) == 10


def test_invalid_input():
    """Test error handling."""
    algo = YourAlgorithm(param1=10)
    with pytest.raises(ValueError):
        algo.train(None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_algorithm.py

# Run with coverage
pytest --cov=tutorials --cov-report=html
```

## Documentation

### README Updates

When adding new content, update the main README.md:

```markdown
## Repository Structure

- `tutorials/XX_new_tutorial/` - Description of new tutorial
```

### Code Comments

```python
# Good: Explain why, not what
# Use epsilon-greedy to balance exploration and exploitation
if np.random.random() < epsilon:
    action = random_action()

# Avoid: Obvious comments
# Set i to 0
i = 0
```

## Pull Request Process

### 1. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of changes"

# Use prefixes:
# Add: New feature or tutorial
# Fix: Bug fix
# Docs: Documentation changes
# Refactor: Code refactoring
# Test: Adding tests
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub and create a Pull Request
2. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New tutorial
- [ ] Enhancement
- [ ] Documentation update

## Testing
- [ ] Code passes all tests
- [ ] Added new tests for new features
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for visual changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 4. Code Review

- Address reviewer feedback
- Make requested changes
- Push updates to the same branch

## Medical Data Guidelines

### Privacy and Ethics

When working with medical data:

1. **Never commit real patient data** to the repository
2. Use synthetic data for examples
3. Ensure HIPAA/GDPR compliance in tutorials
4. Include data privacy disclaimers

### Synthetic Data Generation

```python
def generate_synthetic_medical_data(num_samples: int):
    """
    Generate synthetic medical data for examples.
    
    Note: This is synthetic data for educational purposes only.
    Do not use for clinical decisions.
    """
    # Generation code
    pass
```

### Medical Accuracy

- Cite sources for medical facts
- Have medical terminology reviewed if possible
- Include disclaimers about educational use
- Don't make clinical recommendations

## Adding New Dependencies

### Process

1. Check if dependency is necessary
2. Use well-maintained packages
3. Update `requirements.txt`:

```bash
pip install new-package
pip freeze | grep new-package >> requirements.txt
```

4. Test on clean environment:

```bash
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
# Run tests
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

### Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Mentioned in documentation

## Questions?

- Open an issue with the `question` label
- Check existing issues and documentation
- Reach out to maintainers

## Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [NumPy Documentation Style](https://numpydoc.readthedocs.io/)
- [PyTorch Contributing Guide](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)
- [Scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html)

Thank you for contributing! 🎉
