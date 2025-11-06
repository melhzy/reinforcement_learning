"""
Medical Reward Function for Alzheimer's research.
"""

from typing import Any, Dict, List
import math


class MedicalRewardFunction:
    """
    Reward function specifically designed for medical research tasks.
    
    This reward function considers:
    - Scientific validity
    - Patient safety
    - Research ethics
    - Potential impact
    - Cost-effectiveness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward function.
        
        Args:
            config: Configuration dictionary containing:
                - safety_weight: Weight for safety considerations (default: 0.3)
                - validity_weight: Weight for scientific validity (default: 0.3)
                - impact_weight: Weight for research impact (default: 0.2)
                - ethics_weight: Weight for ethical considerations (default: 0.2)
        """
        self.config = config
        self.safety_weight = config.get("safety_weight", 0.3)
        self.validity_weight = config.get("validity_weight", 0.3)
        self.impact_weight = config.get("impact_weight", 0.2)
        self.ethics_weight = config.get("ethics_weight", 0.2)
        
        # Normalize weights
        total_weight = (self.safety_weight + self.validity_weight + 
                       self.impact_weight + self.ethics_weight)
        self.safety_weight /= total_weight
        self.validity_weight /= total_weight
        self.impact_weight /= total_weight
        self.ethics_weight /= total_weight
    
    def calculate_reward(
        self,
        action: Dict[str, Any],
        state: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a given state-action-next_state transition.
        
        Args:
            action: Action taken
            state: Previous state
            next_state: Resulting state
            
        Returns:
            Computed reward value
        """
        # Calculate individual components
        safety_score = self._evaluate_safety(action, state)
        validity_score = self._evaluate_validity(action, state)
        impact_score = self._evaluate_impact(action, state)
        ethics_score = self._evaluate_ethics(action, state)
        
        # Weighted combination
        total_reward = (
            self.safety_weight * safety_score +
            self.validity_weight * validity_score +
            self.impact_weight * impact_score +
            self.ethics_weight * ethics_score
        )
        
        return total_reward
    
    def _evaluate_safety(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Evaluate safety of an action.
        
        Returns score between 0 and 1.
        """
        action_type = action.get("action_type", "analyze")
        patient_data = state.get("patient_data", {})
        contraindications = patient_data.get("contraindications", [])
        
        # Base safety score
        safety_score = 1.0
        
        # Treatment suggestions with contraindications are risky
        if action_type == "suggest_treatment":
            if contraindications:
                safety_score *= (1.0 - 0.1 * len(contraindications))
            
            # Check age considerations
            age = patient_data.get("age", 65)
            if age > 80:
                safety_score *= 0.9  # Extra caution for elderly
        
        # Experimental designs need safety protocols
        elif action_type == "design_experiment":
            params = action.get("parameters", {})
            if "safety_protocol" not in params:
                safety_score *= 0.8
        
        # Analysis and evaluation are generally safe
        elif action_type in ["analyze_biomarkers", "evaluate_data"]:
            safety_score = 1.0
        
        return max(0.0, min(1.0, safety_score))
    
    def _evaluate_validity(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Evaluate scientific validity of an action.
        
        Returns score between 0 and 1.
        """
        action_type = action.get("action_type", "analyze")
        confidence = action.get("confidence", 0.5)
        reasoning = action.get("reasoning", "")
        
        # Base validity score from confidence
        validity_score = confidence
        
        # Experimental design requires good methodology
        if action_type == "design_experiment":
            params = action.get("parameters", {})
            
            # Check for sample size
            sample_size = params.get("sample_size", 0)
            if sample_size >= 50:
                validity_score *= 1.2
            elif sample_size < 20:
                validity_score *= 0.7
            
            # Check for duration
            duration = params.get("duration_weeks", 0)
            if duration >= 12:
                validity_score *= 1.1
        
        # Data evaluation should be thorough
        elif action_type == "evaluate_data":
            params = action.get("parameters", {})
            quality_threshold = params.get("data_quality_threshold", 0.5)
            if quality_threshold >= 0.8:
                validity_score *= 1.2
        
        # Reward detailed reasoning
        if len(reasoning) > 50:
            validity_score *= 1.1
        
        return max(0.0, min(1.0, validity_score))
    
    def _evaluate_impact(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Evaluate potential research impact of an action.
        
        Returns score between 0 and 1.
        """
        action_type = action.get("action_type", "analyze")
        patient_data = state.get("patient_data", {})
        disease_stage = patient_data.get("disease_stage", "mild")
        
        # Impact varies by action type and disease stage
        impact_scores = {
            "analyze_biomarkers": 0.6,
            "suggest_treatment": 0.8,
            "design_experiment": 0.9,  # Highest impact
            "evaluate_data": 0.7,
            "collect_samples": 0.5
        }
        
        base_impact = impact_scores.get(action_type, 0.5)
        
        # Early intervention has higher potential impact
        if disease_stage in ["healthy", "mild"]:
            base_impact *= 1.2
        
        return max(0.0, min(1.0, base_impact))
    
    def _evaluate_ethics(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """
        Evaluate ethical considerations of an action.
        
        Returns score between 0 and 1.
        """
        action_type = action.get("action_type", "analyze")
        
        # Base ethics score (most actions are ethical)
        ethics_score = 0.9
        
        # Check for patient consent considerations
        if action_type in ["design_experiment", "collect_samples"]:
            params = action.get("parameters", {})
            if "informed_consent" not in params:
                ethics_score *= 0.8
        
        # Treatment suggestions should consider patient autonomy
        if action_type == "suggest_treatment":
            if "patient_preferences" not in action.get("parameters", {}):
                ethics_score *= 0.9
        
        return max(0.0, min(1.0, ethics_score))
    
    def get_reward_breakdown(
        self,
        action: Dict[str, Any],
        state: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Args:
            action: Action taken
            state: Previous state
            next_state: Resulting state
            
        Returns:
            Dictionary with reward component breakdowns
        """
        safety = self._evaluate_safety(action, state)
        validity = self._evaluate_validity(action, state)
        impact = self._evaluate_impact(action, state)
        ethics = self._evaluate_ethics(action, state)
        
        total = (
            self.safety_weight * safety +
            self.validity_weight * validity +
            self.impact_weight * impact +
            self.ethics_weight * ethics
        )
        
        return {
            "safety": safety,
            "validity": validity,
            "impact": impact,
            "ethics": ethics,
            "total": total,
            "weights": {
                "safety": self.safety_weight,
                "validity": self.validity_weight,
                "impact": self.impact_weight,
                "ethics": self.ethics_weight
            }
        }
