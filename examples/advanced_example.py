"""
Advanced example demonstrating custom configurations and data processing.

This example shows:
1. Generating synthetic patient data
2. Custom configuration
3. Advanced training options
4. Reward breakdown analysis
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_framework import LLMAgent, AlzheimersResearchEnv, MedicalRewardFunction, RLTrainer
from rl_framework.utils import DataProcessor
from rl_framework.configs import get_default_config, merge_configs


def main():
    """Run the advanced example."""
    print("="*60)
    print("Alzheimer's Research RL Framework - Advanced Example")
    print("="*60)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic patient dataset...")
    dataset = DataProcessor.generate_synthetic_dataset(
        num_patients=50,
        disease_distribution={
            "healthy": 0.3,
            "mild": 0.4,
            "moderate": 0.2,
            "severe": 0.1
        }
    )
    print(f"✓ Generated {len(dataset)} synthetic patients")
    
    # Save dataset
    DataProcessor.save_dataset(dataset, "./synthetic_patients.json")
    
    # Calculate risk scores
    print("\nCalculating risk scores for patients...")
    high_risk_patients = []
    for patient in dataset:
        risk_score = DataProcessor.calculate_risk_score(patient)
        if risk_score > 0.7:
            high_risk_patients.append((patient['patient_id'], risk_score))
    
    # Note: Patient IDs are synthetic for demonstration purposes
    # In production, never log real patient identifiable information
    print(f"✓ Identified {len(high_risk_patients)} high-risk patients")
    if high_risk_patients:
        print("  Top 3 high-risk patients (synthetic IDs only):")
        for patient_id, risk in sorted(high_risk_patients, key=lambda x: x[1], reverse=True)[:3]:
            print(f"    - {patient_id}: {risk:.2f}")
    
    # Custom configuration
    custom_config = {
        'training': {
            'num_episodes': 50,
            'save_frequency': 5,
        },
        'agent': {
            'temperature': 0.5,
            'learning_rate': 0.0005
        }
    }
    
    base_config = get_default_config()
    config = merge_configs(base_config, custom_config)
    print("\n✓ Custom configuration applied")
    
    # Create components
    env = AlzheimersResearchEnv(config['environment'])
    agent = LLMAgent(config['agent'])
    reward_fn = MedicalRewardFunction(config['reward_function'])
    
    print("✓ Framework components initialized")
    
    # Test reward function breakdown
    print("\n" + "="*60)
    print("Testing Reward Function")
    print("="*60)
    
    # Simulate a state and action
    test_state = {
        'patient_data': dataset[0],
        'research_context': 'Initial assessment'
    }
    
    test_action = {
        'action_type': 'design_experiment',
        'confidence': 0.8,
        'reasoning': 'Designing a comprehensive study for high-risk patients',
        'parameters': {
            'sample_size': 100,
            'duration_weeks': 16,
            'safety_protocol': True
        }
    }
    
    reward_breakdown = reward_fn.get_reward_breakdown(test_action, test_state, test_state)
    
    print("\nReward Breakdown for 'design_experiment' action:")
    print(f"  Safety Score: {reward_breakdown['safety']:.3f}")
    print(f"  Validity Score: {reward_breakdown['validity']:.3f}")
    print(f"  Impact Score: {reward_breakdown['impact']:.3f}")
    print(f"  Ethics Score: {reward_breakdown['ethics']:.3f}")
    print(f"  Total Reward: {reward_breakdown['total']:.3f}")
    
    # Short training run
    print("\n" + "="*60)
    print("Running Training")
    print("="*60)
    
    trainer = RLTrainer(
        agent=agent,
        environment=env,
        reward_function=reward_fn,
        config=config['training']
    )
    
    training_summary = trainer.train()
    
    # Results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Average Reward: {training_summary['average_episode_reward']:.2f}")
    print(f"Final Performance: {training_summary['final_10_episode_average']:.2f}")
    
    # Export
    trainer.export_results("./advanced_results.json")
    
    print("\n✓ Advanced example completed!")


if __name__ == "__main__":
    main()
