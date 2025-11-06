"""
Base Environment interface for the RL framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseEnvironment(ABC):
    """Abstract base class for RL environments."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary for the environment
        """
        self.config = config
        self.state = None
        self.done = False
        
    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render the current state of the environment."""
        pass
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
