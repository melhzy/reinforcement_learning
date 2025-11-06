"""
Chain of Thought Integration for LLM Fine-tuning
Demonstrates RL-based optimization of medical reasoning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import re


class SimpleReasoningModel(nn.Module):
    """
    Simplified reasoning model for demonstration.
    In practice, use transformer-based LLMs.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_steps: int = 5):
        super(SimpleReasoningModel, self).__init__()
        
        self.max_steps = max_steps
        
        # Encoder for patient data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Step generator (simplified)
        self.step_generator = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, 3)  # CN, MCI, AD
        
        # Reasoning quality predictor
        self.quality_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, patient_features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate reasoning chain and classification.
        
        Returns:
            classification_logits, reasoning_steps
        """
        batch_size = patient_features.size(0)
        
        # Encode patient data
        encoded = self.encoder(patient_features)
        
        # Generate reasoning steps
        h = encoded.unsqueeze(1)
        reasoning_steps = []
        
        for step in range(self.max_steps):
            h, _ = self.step_generator(h)
            reasoning_steps.append(h.squeeze(1))
        
        # Final classification
        final_representation = reasoning_steps[-1]
        classification = self.classifier(final_representation)
        
        return classification, reasoning_steps


class MedicalRewardModel:
    """
    Reward model for evaluating reasoning quality.
    """
    
    def __init__(self):
        self.clinical_threshold = {
            'mmse': {'CN': 27, 'MCI': 23, 'AD': 18},
            'cdr': {'CN': 0, 'MCI': 0.5, 'AD': 1.0}
        }
        
        self.biomarker_threshold = {
            'abeta42': 500,  # Below indicates pathology
            'abeta_ratio': 0.08,  # Below indicates pathology
            'ptau181': 50  # Above indicates pathology
        }
    
    def evaluate_clinical_reasoning(
        self,
        reasoning_features: torch.Tensor,
        patient_data: Dict,
        predicted_class: int
    ) -> float:
        """
        Evaluate if clinical reasoning is sound.
        """
        score = 0.0
        
        # Check MMSE interpretation
        mmse = patient_data.get('mmse', 25)
        if predicted_class == 2 and mmse < 20:  # AD
            score += 1.0
        elif predicted_class == 1 and 20 <= mmse < 24:  # MCI
            score += 1.0
        elif predicted_class == 0 and mmse >= 24:  # CN
            score += 1.0
        
        # Check CDR interpretation
        cdr = patient_data.get('cdr', 0)
        if predicted_class == 2 and cdr >= 1.0:
            score += 1.0
        elif predicted_class == 1 and 0.5 <= cdr < 1.0:
            score += 1.0
        elif predicted_class == 0 and cdr == 0:
            score += 1.0
        
        return score
    
    def evaluate_biomarker_reasoning(
        self,
        reasoning_features: torch.Tensor,
        patient_data: Dict,
        predicted_class: int
    ) -> float:
        """
        Evaluate biomarker interpretation.
        """
        score = 0.0
        
        abeta42 = patient_data.get('abeta42', 600)
        abeta40 = patient_data.get('abeta40', 6000)
        abeta_ratio = abeta42 / abeta40 if abeta40 > 0 else 1.0
        ptau = patient_data.get('ptau181', 40)
        
        # AD should have low Aβ42/40 and high p-tau
        if predicted_class == 2:  # AD
            if abeta_ratio < 0.08:
                score += 1.5
            if ptau > 50:
                score += 1.5
        
        # CN should have normal biomarkers
        elif predicted_class == 0:  # CN
            if abeta_ratio >= 0.08:
                score += 1.5
            if ptau <= 50:
                score += 1.5
        
        # MCI intermediate
        elif predicted_class == 1:  # MCI
            score += 1.0  # Partial credit for middle ground
        
        return score
    
    def evaluate_microbiome_reasoning(
        self,
        reasoning_features: torch.Tensor,
        patient_data: Dict,
        predicted_class: int
    ) -> float:
        """
        Evaluate microbiome consideration.
        """
        score = 0.0
        
        p_gingivalis = patient_data.get('p_gingivalis', 1.0)
        fb_ratio = patient_data.get('fb_ratio', 2.0)
        
        # High P. gingivalis associated with AD
        if predicted_class == 2 and p_gingivalis > 2.0:
            score += 0.5
        
        # Dysbiosis (abnormal F/B ratio) in AD
        if predicted_class == 2 and (fb_ratio < 1.0 or fb_ratio > 3.0):
            score += 0.5
        
        return score
    
    def compute_reward(
        self,
        reasoning_features: List[torch.Tensor],
        patient_data: Dict,
        predicted_class: int,
        true_class: int
    ) -> Dict[str, float]:
        """
        Compute comprehensive reward for reasoning quality.
        """
        # Correctness (most important)
        correctness = 10.0 if predicted_class == true_class else -10.0
        
        # Clinical reasoning quality
        clinical_score = self.evaluate_clinical_reasoning(
            reasoning_features[-1],
            patient_data,
            predicted_class
        )
        
        # Biomarker reasoning quality
        biomarker_score = self.evaluate_biomarker_reasoning(
            reasoning_features[-1],
            patient_data,
            predicted_class
        )
        
        # Microbiome reasoning quality
        microbiome_score = self.evaluate_microbiome_reasoning(
            reasoning_features[-1],
            patient_data,
            predicted_class
        )
        
        # Reasoning coherence (check if steps build on each other)
        coherence_score = self.evaluate_coherence(reasoning_features)
        
        total_reward = (
            correctness +
            clinical_score * 2.0 +
            biomarker_score * 2.0 +
            microbiome_score * 1.0 +
            coherence_score * 1.0
        )
        
        return {
            'total': total_reward,
            'correctness': correctness,
            'clinical': clinical_score,
            'biomarker': biomarker_score,
            'microbiome': microbiome_score,
            'coherence': coherence_score
        }
    
    def evaluate_coherence(self, reasoning_features: List[torch.Tensor]) -> float:
        """
        Evaluate if reasoning steps are coherent.
        """
        if len(reasoning_features) < 2:
            return 0.0
        
        # Measure similarity between consecutive steps
        coherence = 0.0
        for i in range(len(reasoning_features) - 1):
            similarity = torch.cosine_similarity(
                reasoning_features[i],
                reasoning_features[i + 1],
                dim=1
            ).mean().item()
            coherence += max(0, similarity - 0.3)  # Should be related but not identical
        
        return min(coherence, 2.0)  # Cap at 2.0


class CoTAgent:
    """
    RL agent for Chain of Thought reasoning in medical diagnosis.
    """
    
    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network (reasoning model)
        self.policy = SimpleReasoningModel(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Reward model
        self.reward_model = MedicalRewardModel()
        
        self.gamma = gamma
    
    def generate_reasoning(self, patient_features: np.ndarray) -> Tuple[int, List]:
        """
        Generate reasoning chain and classification.
        """
        patient_tensor = torch.FloatTensor(patient_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            classification, reasoning_steps = self.policy(patient_tensor)
            predicted_class = classification.argmax(dim=1).item()
        
        return predicted_class, reasoning_steps
    
    def train_step(
        self,
        patient_features: np.ndarray,
        patient_data: Dict,
        true_class: int
    ) -> Dict[str, float]:
        """
        Perform one training step with policy gradient.
        """
        patient_tensor = torch.FloatTensor(patient_features).unsqueeze(0).to(self.device)
        
        # Forward pass
        classification, reasoning_steps = self.policy(patient_tensor)
        predicted_class = classification.argmax(dim=1).item()
        
        # Compute reward
        reward_dict = self.reward_model.compute_reward(
            reasoning_steps,
            patient_data,
            predicted_class,
            true_class
        )
        
        # Policy gradient loss
        log_probs = torch.log_softmax(classification, dim=1)
        selected_log_prob = log_probs[0, true_class]
        
        # REINFORCE-style update
        loss = -selected_log_prob * reward_dict['total']
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'reward': reward_dict['total'],
            'predicted': predicted_class,
            'correct': predicted_class == true_class,
            'reward_breakdown': reward_dict
        }
    
    def train(
        self,
        dataset: List[Tuple],
        num_epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train agent on dataset of patient cases.
        """
        epoch_rewards = []
        epoch_accuracy = []
        
        for epoch in range(num_epochs):
            epoch_reward = 0
            correct = 0
            
            # Shuffle dataset
            np.random.shuffle(dataset)
            
            for patient_features, patient_data, true_class in dataset:
                result = self.train_step(patient_features, patient_data, true_class)
                
                epoch_reward += result['reward']
                if result['correct']:
                    correct += 1
            
            avg_reward = epoch_reward / len(dataset)
            accuracy = correct / len(dataset)
            
            epoch_rewards.append(avg_reward)
            epoch_accuracy.append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Accuracy: {accuracy:.2%}\n")
        
        return epoch_rewards, epoch_accuracy


def generate_synthetic_dataset(num_samples: int = 1000) -> List[Tuple]:
    """
    Generate synthetic Alzheimer's dataset for CoT training.
    """
    dataset = []
    
    for _ in range(num_samples):
        # True class
        true_class = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
        
        # Generate features
        features = np.random.randn(75)  # 75 total features
        
        # Clinical features (influenced by class)
        features[0] += true_class * 0.8  # Age
        features[1] -= true_class * 1.2  # MMSE
        features[2] += true_class * 0.5  # CDR
        
        # Biomarker features
        features[10] -= true_class * 1.0  # Aβ42
        features[11] += true_class * 0.5  # Aβ40
        features[12] += true_class * 1.2  # tau
        features[13] += true_class * 1.5  # p-tau
        
        # Microbiome features
        features[25] += true_class * 0.5  # P. gingivalis
        features[26] += (true_class - 1) * 0.3  # F/B ratio deviation
        
        # Patient data dictionary (for reward computation)
        patient_data = {
            'age': 65 + features[0] * 5,
            'mmse': 28 - true_class * 5 + np.random.randn(),
            'cdr': true_class * 0.5 + np.random.randn() * 0.2,
            'abeta42': 600 - true_class * 150 + np.random.randn() * 50,
            'abeta40': 6000 + np.random.randn() * 500,
            'tau': 300 + true_class * 100 + np.random.randn() * 50,
            'ptau181': 35 + true_class * 20 + np.random.randn() * 10,
            'p_gingivalis': 1.0 + true_class * 0.8 + np.random.randn() * 0.3,
            'fb_ratio': 2.0 + (true_class - 1) * 0.5 + np.random.randn() * 0.3
        }
        
        dataset.append((features, patient_data, true_class))
    
    return dataset


def create_reasoning_prompt(patient_data: Dict) -> str:
    """
    Create human-readable prompt for reasoning visualization.
    """
    prompt = f"""
Chain of Thought Analysis for Alzheimer's Diagnosis:

Step 1 - Clinical Assessment:
  Age: {patient_data['age']:.1f} years
  MMSE Score: {patient_data['mmse']:.1f}/30
  CDR: {patient_data['cdr']:.1f}
  
Step 2 - Biomarker Analysis:
  Aβ42: {patient_data['abeta42']:.1f} pg/mL
  Aβ40: {patient_data['abeta40']:.1f} pg/mL
  Aβ42/40 Ratio: {patient_data['abeta42']/patient_data['abeta40']:.3f}
  Tau: {patient_data['tau']:.1f} pg/mL
  p-tau181: {patient_data['ptau181']:.1f} pg/mL

Step 3 - Microbiome Indicators:
  Oral P. gingivalis: {patient_data['p_gingivalis']:.2f}
  Gut F/B Ratio: {patient_data['fb_ratio']:.2f}

Step 4 - Integration and Diagnosis:
"""
    return prompt


if __name__ == "__main__":
    print("=" * 70)
    print("Chain of Thought RL Training for Alzheimer's Diagnosis")
    print("=" * 70)
    print()
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(num_samples=1000)
    print(f"Created {len(dataset)} patient cases\n")
    
    # Create agent
    print("Initializing CoT agent...")
    agent = CoTAgent(input_dim=75, learning_rate=0.0001)
    print()
    
    # Train agent
    print("Training agent with RL...")
    print()
    rewards, accuracy = agent.train(dataset, num_epochs=100, verbose=True)
    
    # Evaluate on test cases
    print("=" * 70)
    print("Evaluating on test cases...")
    test_dataset = generate_synthetic_dataset(num_samples=100)
    
    correct = 0
    total_reward = 0
    
    for features, patient_data, true_class in test_dataset:
        predicted_class, reasoning_steps = agent.generate_reasoning(features)
        
        if predicted_class == true_class:
            correct += 1
        
        reward_dict = agent.reward_model.compute_reward(
            reasoning_steps,
            patient_data,
            predicted_class,
            true_class
        )
        total_reward += reward_dict['total']
    
    print(f"Test Accuracy: {correct / len(test_dataset):.2%}")
    print(f"Average Test Reward: {total_reward / len(test_dataset):.2f}")
    
    # Show example reasoning
    print("\n" + "=" * 70)
    print("Example Reasoning Chain:")
    print("=" * 70)
    features, patient_data, true_class = test_dataset[0]
    predicted_class, reasoning_steps = agent.generate_reasoning(features)
    
    class_names = ['Cognitively Normal', 'Mild Cognitive Impairment', 'Alzheimer\'s Disease']
    print(create_reasoning_prompt(patient_data))
    print(f"Predicted: {class_names[predicted_class]}")
    print(f"True Label: {class_names[true_class]}")
    print(f"Correct: {predicted_class == true_class}")
