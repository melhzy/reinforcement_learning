"""
Base Agent interface for the RL framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config
        self.state = None
        
    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, experience: Tuple) -> Dict[str, float]:
        """
        Update agent based on experience.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent to disk.
        
        Args:
            path: Path to save the agent
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent from disk.
        
        Args:
            path: Path to load the agent from
        """
        pass
