"""
Alzheimer's Research Environment for RL.
"""

import random
from typing import Any, Dict, List, Tuple
from .base_environment import BaseEnvironment


class AlzheimersResearchEnv(BaseEnvironment):
    """
    Environment for Alzheimer's disease research tasks.
    
    This environment simulates research scenarios where an agent must:
    - Analyze patient data
    - Suggest treatments or interventions
    - Design experiments
    - Evaluate research outcomes
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Alzheimer's research environment.
        
        Args:
            config: Configuration dictionary containing:
                - num_patients: Number of simulated patients
                - disease_stages: List of disease stages to consider
                - available_biomarkers: List of biomarkers to track
                - max_steps: Maximum steps per episode
        """
        super().__init__(config)
        
        self.num_patients = config.get("num_patients", 100)
        self.disease_stages = config.get("disease_stages", 
                                        ["healthy", "mild", "moderate", "severe"])
        self.available_biomarkers = config.get("available_biomarkers",
                                              ["amyloid_beta", "tau_protein", "apoe4", 
                                               "brain_volume", "cognitive_score"])
        self.max_steps = config.get("max_steps", 50)
        
        # Current episode state
        self.current_step = 0
        self.current_patient = None
        self.research_history: List[Dict] = []
        
        # Available actions
        self.action_space = [
            "analyze_biomarkers",
            "suggest_treatment",
            "design_experiment",
            "evaluate_data",
            "collect_samples"
        ]
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to a new research scenario.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.done = False
        self.research_history = []
        
        # Generate a new patient case
        self.current_patient = self._generate_patient()
        
        # Create initial observation
        observation = {
            "patient_data": self.current_patient,
            "research_context": "Initial patient assessment for Alzheimer's research",
            "available_actions": self.action_space,
            "step": self.current_step,
            "research_history": []
        }
        
        self.state = observation
        return observation
    
    def _generate_patient(self) -> Dict[str, Any]:
        """Generate a simulated patient with medical data."""
        age = random.randint(55, 90)
        disease_stage = random.choice(self.disease_stages)
        
        # Biomarker values (simulated)
        biomarkers = {}
        for marker in self.available_biomarkers:
            if marker == "amyloid_beta":
                # Higher values indicate more pathology
                biomarkers[marker] = random.uniform(0.3, 1.2)
            elif marker == "tau_protein":
                biomarkers[marker] = random.uniform(0.2, 1.0)
            elif marker == "apoe4":
                # 0, 1, or 2 alleles
                biomarkers[marker] = random.choice([0, 1, 2])
            elif marker == "brain_volume":
                # Normalized volume (lower is worse)
                biomarkers[marker] = random.uniform(0.7, 1.0)
            elif marker == "cognitive_score":
                # Score from 0-30 (MMSE-like)
                biomarkers[marker] = random.randint(10, 30)
        
        patient = {
            "patient_id": f"P{random.randint(1000, 9999)}",
            "age": age,
            "disease_stage": disease_stage,
            "biomarkers": biomarkers,
            "medical_history": self._generate_medical_history(),
            "contraindications": self._generate_contraindications()
        }
        
        return patient
    
    def _generate_medical_history(self) -> List[str]:
        """Generate simulated medical history."""
        possible_conditions = [
            "hypertension",
            "diabetes",
            "cardiovascular_disease",
            "depression",
            "vitamin_deficiency"
        ]
        num_conditions = random.randint(0, 3)
        return random.sample(possible_conditions, num_conditions)
    
    def _generate_contraindications(self) -> List[str]:
        """Generate contraindications for treatments."""
        possible_contraindications = [
            "kidney_disease",
            "liver_disease",
            "bleeding_disorder",
            "drug_allergies"
        ]
        num_contraindications = random.randint(0, 2)
        return random.sample(possible_contraindications, num_contraindications)
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action dictionary containing:
                - action_type: Type of action to perform
                - parameters: Action-specific parameters
                - reasoning: Agent's reasoning for the action
                
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Extract action type
        action_type = action.get("action_type", "analyze")
        
        # Calculate reward based on action quality
        reward = self._calculate_reward(action)
        
        # Update research history
        self.research_history.append({
            "step": self.current_step,
            "action": action_type,
            "reward": reward
        })
        
        # Check if episode is done
        self.done = (self.current_step >= self.max_steps) or self._check_termination_condition(action)
        
        # Generate next observation
        observation = {
            "patient_data": self.current_patient,
            "research_context": self._get_updated_context(action),
            "available_actions": self.action_space,
            "step": self.current_step,
            "research_history": self.research_history[-5:]  # Last 5 actions
        }
        
        # Additional info
        info = {
            "action_type": action_type,
            "cumulative_reward": sum(h["reward"] for h in self.research_history),
            "episode_length": self.current_step
        }
        
        self.state = observation
        return observation, reward, self.done, info
    
    def _calculate_reward(self, action: Dict[str, Any]) -> float:
        """
        Calculate reward for an action.
        
        Rewards are based on:
        - Action appropriateness for disease stage
        - Safety considerations
        - Scientific rigor
        - Potential research impact
        """
        action_type = action.get("action_type", "analyze")
        confidence = action.get("confidence", 0.5)
        
        base_reward = 0.0
        
        # Reward based on action type and context
        disease_stage = self.current_patient.get("disease_stage", "mild")
        
        if action_type == "analyze_biomarkers":
            # Always good to analyze data
            base_reward = 1.0
            
        elif action_type == "suggest_treatment":
            # Higher reward for appropriate stage treatment
            if disease_stage in ["mild", "moderate"]:
                base_reward = 2.0
            else:
                base_reward = 1.5
                
            # Penalty for contraindications not considered
            contraindications = self.current_patient.get("contraindications", [])
            if contraindications:
                base_reward *= 0.8
                
        elif action_type == "design_experiment":
            # High reward for experimental design
            base_reward = 2.5
            
        elif action_type == "evaluate_data":
            # Good reward for data evaluation
            base_reward = 1.5
            
        elif action_type == "collect_samples":
            # Moderate reward for sample collection
            base_reward = 1.2
        
        # Modify by confidence
        final_reward = base_reward * confidence
        
        # Add small noise
        final_reward += random.uniform(-0.1, 0.1)
        
        return final_reward
    
    def _check_termination_condition(self, action: Dict[str, Any]) -> bool:
        """Check if episode should terminate early."""
        # Terminate if a comprehensive experiment is designed
        if action.get("action_type") == "design_experiment":
            if self.current_step >= 10:  # At least 10 steps of preparation
                return True
        return False
    
    def _get_updated_context(self, action: Dict[str, Any]) -> str:
        """Generate updated research context after an action."""
        action_type = action.get("action_type", "analyze")
        
        contexts = {
            "analyze_biomarkers": f"Biomarker analysis completed. Current disease stage: {self.current_patient['disease_stage']}",
            "suggest_treatment": "Treatment suggestions have been recorded for review",
            "design_experiment": "Experimental protocol has been drafted and awaits approval",
            "evaluate_data": "Data quality assessment complete, ready for further analysis",
            "collect_samples": "Sample collection scheduled, awaiting laboratory processing"
        }
        
        return contexts.get(action_type, "Research action completed")
    
    def render(self) -> None:
        """Render the current state of the environment."""
        print("\n" + "="*60)
        print(f"Alzheimer's Research Environment - Step {self.current_step}")
        print("="*60)
        
        if self.current_patient:
            print(f"\nPatient ID: {self.current_patient['patient_id']}")
            print(f"Age: {self.current_patient['age']}")
            print(f"Disease Stage: {self.current_patient['disease_stage']}")
            print(f"\nBiomarkers:")
            for marker, value in self.current_patient['biomarkers'].items():
                print(f"  - {marker}: {value:.2f}" if isinstance(value, float) else f"  - {marker}: {value}")
        
        if self.research_history:
            print(f"\nRecent Actions:")
            for entry in self.research_history[-3:]:
                print(f"  Step {entry['step']}: {entry['action']} (reward: {entry['reward']:.2f})")
        
        print("="*60 + "\n")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current episode."""
        return {
            "total_steps": self.current_step,
            "total_reward": sum(h["reward"] for h in self.research_history),
            "actions_taken": [h["action"] for h in self.research_history],
            "patient_stage": self.current_patient.get("disease_stage") if self.current_patient else None
        }
