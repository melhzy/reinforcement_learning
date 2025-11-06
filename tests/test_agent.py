"""
Tests for the LLM Agent.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_framework.agents import LLMAgent


class TestLLMAgent(unittest.TestCase):
    """Test cases for LLM Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model_name": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
            "learning_rate": 0.001,
            "memory_size": 1000
        }
        self.agent = LLMAgent(self.config)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.model_name, "test-model")
        self.assertEqual(self.agent.temperature, 0.7)
        self.assertEqual(self.agent.learning_rate, 0.001)
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_select_action(self):
        """Test action selection."""
        observation = {
            'patient_data': {
                'age': 70,
                'disease_stage': 'mild',
                'biomarkers': {'amyloid_beta': 0.5}
            },
            'research_context': 'Test context',
            'available_actions': ['analyze_biomarkers', 'suggest_treatment']
        }
        
        action = self.agent.select_action(observation)
        
        self.assertIsInstance(action, dict)
        self.assertIn('action_type', action)
        self.assertIn('reasoning', action)
        self.assertIn('confidence', action)
    
    def test_update(self):
        """Test agent update."""
        state = {'patient_data': {'age': 70, 'disease_stage': 'mild'}}
        action = {'action_type': 'analyze', 'confidence': 0.8}
        reward = 1.5
        next_state = {'patient_data': {'age': 70, 'disease_stage': 'mild'}}
        done = False
        
        experience = (state, action, reward, next_state, done)
        metrics = self.agent.update(experience)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('td_error', metrics)
        self.assertIn('value_estimate', metrics)
        self.assertIn('reward', metrics)
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_memory_limit(self):
        """Test memory buffer size limit."""
        state = {'test': 'state'}
        action = {'action_type': 'test'}
        
        # Add more experiences than memory size
        for i in range(self.config['memory_size'] + 100):
            experience = (state, action, 1.0, state, False)
            self.agent.update(experience)
        
        # Memory should not exceed limit
        self.assertEqual(len(self.agent.memory), self.config['memory_size'])
    
    def test_save_load(self):
        """Test saving and loading agent."""
        import tempfile
        
        # Add some experiences
        state = {'patient_data': {'age': 70}}
        action = {'action_type': 'analyze'}
        experience = (state, action, 1.0, state, False)
        self.agent.update(experience)
        
        # Save agent
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.agent.save(temp_path)
            
            # Create new agent and load
            new_agent = LLMAgent(self.config)
            new_agent.load(temp_path)
            
            # Check loaded state
            self.assertEqual(len(new_agent.value_estimates), len(self.agent.value_estimates))
            self.assertEqual(new_agent.model_name, self.agent.model_name)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_policy_summary(self):
        """Test policy summary generation."""
        # Add some experiences
        state = {'patient_data': {'age': 70, 'disease_stage': 'mild'}}
        action = {'action_type': 'analyze'}
        
        for i in range(5):
            experience = (state, action, 1.0, state, False)
            self.agent.update(experience)
        
        summary = self.agent.get_policy_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_experiences', summary)
        self.assertIn('unique_states', summary)
        self.assertIn('average_value', summary)
        self.assertEqual(summary['total_experiences'], 5)


if __name__ == '__main__':
    unittest.main()
