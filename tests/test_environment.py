"""
Tests for the Alzheimer's Research Environment.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_framework.environments import AlzheimersResearchEnv


class TestAlzheimersEnvironment(unittest.TestCase):
    """Test cases for Alzheimer's Research Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "num_patients": 10,
            "disease_stages": ["healthy", "mild", "moderate", "severe"],
            "available_biomarkers": ["amyloid_beta", "tau_protein", "cognitive_score"],
            "max_steps": 20
        }
        self.env = AlzheimersResearchEnv(self.config)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.num_patients, 10)
        self.assertEqual(self.env.max_steps, 20)
        self.assertEqual(len(self.env.disease_stages), 4)
    
    def test_reset(self):
        """Test environment reset."""
        observation = self.env.reset()
        
        self.assertIsInstance(observation, dict)
        self.assertIn('patient_data', observation)
        self.assertIn('research_context', observation)
        self.assertIn('available_actions', observation)
        self.assertEqual(observation['step'], 0)
        self.assertEqual(self.env.current_step, 0)
        self.assertFalse(self.env.done)
    
    def test_patient_generation(self):
        """Test patient data generation."""
        observation = self.env.reset()
        patient = observation['patient_data']
        
        self.assertIn('patient_id', patient)
        self.assertIn('age', patient)
        self.assertIn('disease_stage', patient)
        self.assertIn('biomarkers', patient)
        self.assertIn('medical_history', patient)
        
        # Check age range
        self.assertGreaterEqual(patient['age'], 55)
        self.assertLessEqual(patient['age'], 90)
        
        # Check disease stage
        self.assertIn(patient['disease_stage'], self.config['disease_stages'])
    
    def test_step(self):
        """Test environment step."""
        observation = self.env.reset()
        
        action = {
            'action_type': 'analyze_biomarkers',
            'confidence': 0.8,
            'reasoning': 'Initial analysis',
            'parameters': {}
        }
        
        next_observation, reward, done, info = self.env.step(action)
        
        # Check return types
        self.assertIsInstance(next_observation, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check observation structure
        self.assertIn('patient_data', next_observation)
        self.assertIn('research_context', next_observation)
        self.assertEqual(next_observation['step'], 1)
        
        # Check info
        self.assertIn('action_type', info)
        self.assertIn('cumulative_reward', info)
    
    def test_episode_termination(self):
        """Test episode termination."""
        observation = self.env.reset()
        
        action = {
            'action_type': 'analyze_biomarkers',
            'confidence': 0.8
        }
        
        # Run until max steps
        for _ in range(self.config['max_steps']):
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        
        self.assertTrue(done or self.env.current_step >= self.config['max_steps'])
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        observation = self.env.reset()
        
        # Test different action types
        actions = [
            {'action_type': 'analyze_biomarkers', 'confidence': 0.9},
            {'action_type': 'suggest_treatment', 'confidence': 0.8},
            {'action_type': 'design_experiment', 'confidence': 0.85}
        ]
        
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            self.assertIsInstance(reward, float)
            self.assertGreater(reward, 0)  # Rewards should be positive
    
    def test_episode_summary(self):
        """Test episode summary."""
        observation = self.env.reset()
        
        # Take some actions
        action = {'action_type': 'analyze_biomarkers', 'confidence': 0.8}
        for _ in range(5):
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        
        summary = self.env.get_episode_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_steps', summary)
        self.assertIn('total_reward', summary)
        self.assertIn('actions_taken', summary)
        self.assertEqual(summary['total_steps'], self.env.current_step)


if __name__ == '__main__':
    unittest.main()
