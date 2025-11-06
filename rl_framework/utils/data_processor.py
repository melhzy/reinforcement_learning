"""
Data processing utilities for Alzheimer's research.
"""

import json
from typing import Any, Dict, List, Optional
import random


class DataProcessor:
    """Utility class for processing medical research data."""
    
    @staticmethod
    def normalize_biomarkers(biomarkers: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize biomarker values to [0, 1] range.
        
        Args:
            biomarkers: Dictionary of biomarker values
            
        Returns:
            Normalized biomarker values
        """
        # Standard ranges for normalization
        ranges = {
            "amyloid_beta": (0.0, 1.5),
            "tau_protein": (0.0, 1.2),
            "brain_volume": (0.5, 1.0),
            "cognitive_score": (0, 30)
        }
        
        normalized = {}
        for marker, value in biomarkers.items():
            if marker in ranges:
                min_val, max_val = ranges[marker]
                normalized[marker] = (value - min_val) / (max_val - min_val)
                normalized[marker] = max(0.0, min(1.0, normalized[marker]))
            else:
                normalized[marker] = value
        
        return normalized
    
    @staticmethod
    def calculate_risk_score(patient_data: Dict[str, Any]) -> float:
        """
        Calculate an overall risk score for Alzheimer's disease.
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Risk score between 0 and 1
        """
        biomarkers = patient_data.get("biomarkers", {})
        age = patient_data.get("age", 65)
        
        # Normalize biomarkers
        norm_biomarkers = DataProcessor.normalize_biomarkers(biomarkers)
        
        # Weighted risk calculation
        risk_score = 0.0
        
        # Age contribution (increases with age)
        age_risk = max(0, (age - 55) / 35)  # Scale from 55-90
        risk_score += 0.2 * age_risk
        
        # Biomarker contributions
        if "amyloid_beta" in norm_biomarkers:
            risk_score += 0.25 * norm_biomarkers["amyloid_beta"]
        
        if "tau_protein" in norm_biomarkers:
            risk_score += 0.25 * norm_biomarkers["tau_protein"]
        
        if "apoe4" in biomarkers:
            # 0, 1, or 2 alleles
            apoe4_count = biomarkers["apoe4"]
            risk_score += 0.15 * (apoe4_count / 2)
        
        if "brain_volume" in norm_biomarkers:
            # Lower volume = higher risk
            risk_score += 0.15 * (1 - norm_biomarkers["brain_volume"])
        
        return min(1.0, risk_score)
    
    @staticmethod
    def generate_synthetic_dataset(
        num_patients: int,
        disease_distribution: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic patient dataset for testing.
        
        Args:
            num_patients: Number of patients to generate
            disease_distribution: Distribution of disease stages
            
        Returns:
            List of patient data dictionaries
        """
        if disease_distribution is None:
            disease_distribution = {
                "healthy": 0.4,
                "mild": 0.3,
                "moderate": 0.2,
                "severe": 0.1
            }
        
        # Calculate number of patients per stage
        stages = []
        for stage, proportion in disease_distribution.items():
            count = int(num_patients * proportion)
            stages.extend([stage] * count)
        
        # Fill remaining to reach exact num_patients
        while len(stages) < num_patients:
            stages.append("mild")
        
        random.shuffle(stages)
        
        dataset = []
        for i, stage in enumerate(stages):
            patient = {
                "patient_id": f"P{1000 + i}",
                "age": random.randint(55, 90),
                "disease_stage": stage,
                "biomarkers": DataProcessor._generate_biomarkers_for_stage(stage),
                "medical_history": DataProcessor._generate_medical_history(),
                "contraindications": random.sample(
                    ["kidney_disease", "liver_disease", "bleeding_disorder", "drug_allergies"],
                    random.randint(0, 2)
                )
            }
            dataset.append(patient)
        
        return dataset
    
    @staticmethod
    def _generate_biomarkers_for_stage(stage: str) -> Dict[str, Any]:
        """Generate biomarkers consistent with disease stage."""
        # Base values depend on stage
        stage_params = {
            "healthy": {
                "amyloid_beta": (0.1, 0.4),
                "tau_protein": (0.1, 0.3),
                "brain_volume": (0.9, 1.0),
                "cognitive_score": (27, 30)
            },
            "mild": {
                "amyloid_beta": (0.4, 0.7),
                "tau_protein": (0.3, 0.6),
                "brain_volume": (0.8, 0.9),
                "cognitive_score": (21, 27)
            },
            "moderate": {
                "amyloid_beta": (0.7, 1.0),
                "tau_protein": (0.6, 0.8),
                "brain_volume": (0.7, 0.8),
                "cognitive_score": (15, 21)
            },
            "severe": {
                "amyloid_beta": (1.0, 1.3),
                "tau_protein": (0.8, 1.0),
                "brain_volume": (0.6, 0.7),
                "cognitive_score": (0, 15)
            }
        }
        
        params = stage_params.get(stage, stage_params["mild"])
        
        return {
            "amyloid_beta": random.uniform(*params["amyloid_beta"]),
            "tau_protein": random.uniform(*params["tau_protein"]),
            "apoe4": random.choice([0, 1, 2]),
            "brain_volume": random.uniform(*params["brain_volume"]),
            "cognitive_score": random.randint(*params["cognitive_score"])
        }
    
    @staticmethod
    def _generate_medical_history() -> List[str]:
        """Generate medical history."""
        conditions = ["hypertension", "diabetes", "cardiovascular_disease", "depression"]
        return random.sample(conditions, random.randint(0, 3))
    
    @staticmethod
    def save_dataset(dataset: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            dataset: List of patient data
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath}")
    
    @staticmethod
    def load_dataset(filepath: str) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            List of patient data
        """
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        print(f"Dataset loaded from {filepath}: {len(dataset)} patients")
        return dataset
