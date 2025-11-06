"""
Reinforcement Learning Framework for LLM + Alzheimer's Disease Research

A modular framework for applying reinforcement learning to medical research,
specifically focusing on Alzheimer's disease using Large Language Models.
"""

__version__ = "0.1.0"

from .agents.llm_agent import LLMAgent
from .environments.alzheimers_env import AlzheimersResearchEnv
from .rewards.medical_reward import MedicalRewardFunction
from .utils.trainer import RLTrainer

__all__ = [
    "LLMAgent",
    "AlzheimersResearchEnv",
    "MedicalRewardFunction",
    "RLTrainer",
]
