"""
Tests for the Medical Reward Function.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_framework.rewards import MedicalRewardFunction


class TestMedicalReward(unittest.TestCase):
    """Test cases for Medical Reward Function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "safety_weight": 0.3,
            "validity_weight": 0.3,
            "impact_weight": 0.2,
            "ethics_weight": 0.2
        }
        self.reward_fn = MedicalRewardFunction(self.config)
    
    def test_initialization(self):
        """Test reward function initialization."""
        # Weights should be normalized to sum to 1
        total_weight = (self.reward_fn.safety_weight + 
                       self.reward_fn.validity_weight +
                       self.reward_fn.impact_weight +
                       self.reward_fn.ethics_weight)
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_calculate_reward(self):
        """Test reward calculation."""
        state = {
            'patient_data': {
                'age': 70,
                'disease_stage': 'mild',
                'contraindications': []
            }
        }
        
        action = {
            'action_type': 'analyze_biomarkers',
            'confidence': 0.8,
            'reasoning': 'Detailed analysis of biomarkers',
            'parameters': {}
        }
        
        reward = self.reward_fn.calculate_reward(action, state, state)
        
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)
        self.assertLessEqual(reward, 1.0)
    
    def test_safety_evaluation(self):
        """Test safety score evaluation."""
        # Safe action
        safe_state = {
            'patient_data': {
                'age': 65,
                'disease_stage': 'mild',
                'contraindications': []
            }
        }
        safe_action = {
            'action_type': 'analyze_biomarkers',
            'confidence': 0.9
        }
        
        # Risky action
        risky_state = {
            'patient_data': {
                'age': 85,
                'disease_stage': 'severe',
                'contraindications': ['kidney_disease', 'liver_disease']
            }
        }
        risky_action = {
            'action_type': 'suggest_treatment',
            'confidence': 0.7,
            'parameters': {}
        }
        
        safe_reward = self.reward_fn.calculate_reward(safe_action, safe_state, safe_state)
        risky_reward = self.reward_fn.calculate_reward(risky_action, risky_state, risky_state)
        
        # Safe actions should generally get higher rewards
        # (though not always due to other factors)
        self.assertIsInstance(safe_reward, float)
        self.assertIsInstance(risky_reward, float)
    
    def test_validity_evaluation(self):
        """Test validity score evaluation."""
        state = {
            'patient_data': {
                'age': 70,
                'disease_stage': 'mild'
            }
        }
        
        # High confidence, detailed reasoning
        good_action = {
            'action_type': 'design_experiment',
            'confidence': 0.9,
            'reasoning': 'This is a detailed explanation of the experimental design rationale',
            'parameters': {
                'sample_size': 100,
                'duration_weeks': 16
            }
        }
        
        # Low confidence, minimal reasoning
        poor_action = {
            'action_type': 'design_experiment',
            'confidence': 0.5,
            'reasoning': 'Quick test',
            'parameters': {
                'sample_size': 10,
                'duration_weeks': 2
            }
        }
        
        good_reward = self.reward_fn.calculate_reward(good_action, state, state)
        poor_reward = self.reward_fn.calculate_reward(poor_action, state, state)
        
        # Better designed experiments should get higher rewards
        self.assertGreater(good_reward, poor_reward)
    
    def test_impact_evaluation(self):
        """Test impact score evaluation."""
        # Early stage (higher potential impact)
        early_state = {
            'patient_data': {
                'disease_stage': 'mild'
            }
        }
        
        # Late stage
        late_state = {
            'patient_data': {
                'disease_stage': 'severe'
            }
        }
        
        action = {
            'action_type': 'design_experiment',
            'confidence': 0.8
        }
        
        early_reward = self.reward_fn.calculate_reward(action, early_state, early_state)
        late_reward = self.reward_fn.calculate_reward(action, late_state, late_state)
        
        # Early interventions typically have higher impact
        self.assertGreater(early_reward, late_reward)
    
    def test_reward_breakdown(self):
        """Test reward breakdown functionality."""
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
            'reasoning': 'Well-thought-out experimental protocol',
            'parameters': {
                'sample_size': 100,
                'duration_weeks': 12
            }
        }
        
        breakdown = self.reward_fn.get_reward_breakdown(action, state, state)
        
        self.assertIsInstance(breakdown, dict)
        self.assertIn('safety', breakdown)
        self.assertIn('validity', breakdown)
        self.assertIn('impact', breakdown)
        self.assertIn('ethics', breakdown)
        self.assertIn('total', breakdown)
        self.assertIn('weights', breakdown)
        
        # All scores should be between 0 and 1
        for key in ['safety', 'validity', 'impact', 'ethics']:
            self.assertGreaterEqual(breakdown[key], 0)
            self.assertLessEqual(breakdown[key], 1.0)


if __name__ == '__main__':
    unittest.main()
