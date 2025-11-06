"""
Basic example of using the RL framework for Alzheimer's research.

This example demonstrates:
1. Setting up the environment
2. Creating an LLM-based agent
3. Training the agent
4. Evaluating performance
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_framework import LLMAgent, AlzheimersResearchEnv, MedicalRewardFunction, RLTrainer
from rl_framework.configs import get_default_config


def main():
    """Run the basic example."""
    print("="*60)
    print("Alzheimer's Research RL Framework - Basic Example")
    print("="*60)
    
    # Load configuration
    config = get_default_config()
    print("\n✓ Configuration loaded")
    
    # Create environment
    env = AlzheimersResearchEnv(config['environment'])
    print("✓ Environment created")
    
    # Create agent
    agent = LLMAgent(config['agent'])
    print("✓ LLM Agent created")
    
    # Create reward function
    reward_fn = MedicalRewardFunction(config['reward_function'])
    print("✓ Medical reward function created")
    
    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        environment=env,
        reward_function=reward_fn,
        config=config['training']
    )
    print("✓ Trainer created")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    # Train the agent
    training_summary = trainer.train()
    
    # Print training summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Episodes: {training_summary['total_episodes']}")
    print(f"Total Steps: {training_summary['total_steps']}")
    print(f"Average Episode Reward: {training_summary['average_episode_reward']:.2f}")
    print(f"Best Episode Reward: {training_summary['best_episode_reward']:.2f}")
    print(f"Final 10 Episode Average: {training_summary['final_10_episode_average']:.2f}")
    
    # Evaluate the agent
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    eval_summary = trainer.evaluate(num_episodes=5, render=False)
    
    # Export results
    trainer.export_results("./training_results.json")
    
    # Show agent policy summary
    print("\n" + "="*60)
    print("Agent Policy Summary")
    print("="*60)
    policy_summary = agent.get_policy_summary()
    print(f"Total Experiences: {policy_summary['total_experiences']}")
    print(f"Unique States: {policy_summary['unique_states']}")
    print(f"Average Value: {policy_summary['average_value']:.3f}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
