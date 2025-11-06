# Tutorial 6: Chain of Thought (CoT) Integration for LLM Fine-tuning

## Overview

This tutorial demonstrates how to integrate reinforcement learning with Large Language Models (LLMs) for medical reasoning tasks, specifically focusing on Chain of Thought (CoT) fine-tuning for Alzheimer's disease analysis.

## What is Chain of Thought (CoT)?

Chain of Thought prompting enables LLMs to break down complex reasoning into intermediate steps, similar to human problem-solving:

**Standard Prompting:**
```
Q: Classify this patient's Alzheimer's status.
A: Alzheimer's Disease
```

**CoT Prompting:**
```
Q: Classify this patient's Alzheimer's status.
A: Let me analyze step by step:
1. The MMSE score is 18, indicating cognitive impairment
2. Aβ42 levels are significantly reduced (< 500 pg/mL)
3. p-tau181 is elevated (> 50 pg/mL)
4. The Aβ42/Aβ40 ratio is < 0.08
5. These biomarkers strongly suggest amyloid pathology
6. Combined with clinical presentation: Alzheimer's Disease
```

## RL for CoT Fine-tuning

### Why Use RL?

Traditional supervised learning:
- Requires explicit reasoning chains
- Limited to seen examples
- Difficult to optimize for correctness

RL enables:
- **Self-supervised learning**: Generate reasoning chains
- **Reward-based optimization**: Optimize for final answer correctness
- **Exploration**: Discover new reasoning patterns
- **Iterative improvement**: Learn from mistakes

### RLHF (Reinforcement Learning from Human Feedback)

1. **Pre-training**: Train base medical LLM
2. **Supervised Fine-tuning**: Train on medical Q&A with CoT
3. **Reward Modeling**: Train reward model from human preferences
4. **RL Optimization**: Use PPO to optimize policy

## Architecture

### 1. Policy (LLM)
The language model that generates reasoning chains:

```python
class MedicalCoTPolicy(nn.Module):
    """
    Policy network based on transformer LLM.
    """
    def __init__(self, base_model="biomedbert"):
        self.llm = AutoModelForCausalLM.from_pretrained(base_model)
    
    def generate_reasoning(self, patient_data):
        """Generate CoT reasoning for patient diagnosis."""
        prompt = self.format_patient_data(patient_data)
        reasoning = self.llm.generate(prompt, max_length=512)
        return reasoning
```

### 2. Reward Model
Evaluates quality of reasoning chains:

```python
class MedicalRewardModel:
    """
    Reward model for medical reasoning quality.
    """
    def compute_reward(self, reasoning, patient_data, true_diagnosis):
        rewards = {
            'correctness': self.check_diagnosis_correct(reasoning),
            'clinical_accuracy': self.verify_clinical_facts(reasoning),
            'biomarker_interpretation': self.assess_biomarker_logic(reasoning),
            'reasoning_coherence': self.evaluate_logic_flow(reasoning),
            'confidence': self.estimate_confidence(reasoning)
        }
        return sum(rewards.values())
```

### 3. Training Loop
RL fine-tuning with PPO:

```python
for episode in range(num_episodes):
    # Sample patient case
    patient_data = sample_patient()
    
    # Generate reasoning chain
    reasoning = policy.generate_reasoning(patient_data)
    
    # Extract diagnosis from reasoning
    diagnosis = extract_diagnosis(reasoning)
    
    # Compute reward
    reward = reward_model.compute_reward(
        reasoning, 
        patient_data, 
        true_diagnosis
    )
    
    # Update policy using PPO
    policy.update(reasoning, reward)
```

## Application: Alzheimer's Diagnosis CoT

### Example Training Data Format

```json
{
  "patient_id": "AD_001",
  "features": {
    "clinical": {
      "age": 72,
      "mmse": 18,
      "cdr": 1.0,
      "education_years": 16
    },
    "microbiome": {
      "oral": {
        "p_gingivalis": 2.3,
        "diversity": 3.1
      },
      "gut": {
        "firmicutes_bacteroidetes_ratio": 1.8,
        "lactobacillus": 0.5
      }
    },
    "biomarkers": {
      "abeta42": 420,
      "abeta40": 5200,
      "tau": 580,
      "ptau181": 65,
      "nfl": 45
    }
  },
  "ground_truth": "Alzheimer's Disease",
  "reasoning_chain": [
    "Step 1: Assess cognitive status...",
    "Step 2: Evaluate biomarker profile...",
    "Step 3: Consider microbiome indicators...",
    "Step 4: Integrate all evidence...",
    "Conclusion: Alzheimer's Disease"
  ]
}
```

### Reward Function Components

```python
def compute_comprehensive_reward(reasoning, patient, true_label):
    """
    Multi-component reward for medical reasoning.
    """
    # Correctness (most important)
    if extract_diagnosis(reasoning) == true_label:
        correctness_reward = 10.0
    else:
        correctness_reward = -10.0
    
    # Clinical reasoning quality
    clinical_score = evaluate_clinical_logic(reasoning, patient)
    
    # Biomarker interpretation
    biomarker_score = evaluate_biomarker_reasoning(reasoning, patient)
    
    # Microbiome analysis
    microbiome_score = evaluate_microbiome_integration(reasoning, patient)
    
    # Reasoning structure
    structure_score = evaluate_reasoning_structure(reasoning)
    
    # Confidence calibration
    confidence_score = evaluate_confidence_calibration(reasoning, correctness)
    
    # Medical accuracy (no hallucinations)
    factual_score = check_medical_facts(reasoning)
    
    return {
        'total': (correctness_reward + clinical_score + biomarker_score + 
                 microbiome_score + structure_score + confidence_score + 
                 factual_score),
        'breakdown': {
            'correctness': correctness_reward,
            'clinical': clinical_score,
            'biomarker': biomarker_score,
            'microbiome': microbiome_score,
            'structure': structure_score,
            'confidence': confidence_score,
            'factual': factual_score
        }
    }
```

## Implementation Details

### Prompt Engineering

```python
def create_alzheimer_prompt(patient_data):
    """
    Create structured prompt for Alzheimer's diagnosis.
    """
    prompt = f"""
You are an expert neurologist specializing in Alzheimer's disease.
Analyze the following patient data and provide a step-by-step diagnosis.

Patient Data:
- Age: {patient_data['age']} years
- MMSE Score: {patient_data['mmse']}/30
- CDR: {patient_data['cdr']}

Biomarkers:
- Aβ42: {patient_data['abeta42']} pg/mL
- Aβ40: {patient_data['abeta40']} pg/mL
- Aβ42/40 ratio: {patient_data['abeta42']/patient_data['abeta40']:.3f}
- Tau: {patient_data['tau']} pg/mL
- p-tau181: {patient_data['ptau181']} pg/mL

Microbiome:
- Oral P. gingivalis: {patient_data['p_gingivalis']}
- Gut F/B ratio: {patient_data['fb_ratio']}

Instructions:
1. Evaluate cognitive assessment results
2. Interpret biomarker profile
3. Consider microbiome indicators
4. Integrate all evidence
5. Provide final diagnosis (CN/MCI/AD) with confidence level

Provide your reasoning step by step:
"""
    return prompt
```

### Training with Transformer Reinforcement Learning (TRL)

```python
from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base")

# PPO configuration
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True
)

# Create PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer
)

# Training loop
for epoch in range(num_epochs):
    for patient_batch in dataset:
        # Generate responses
        prompts = [create_alzheimer_prompt(p) for p in patient_batch]
        responses = ppo_trainer.generate(prompts)
        
        # Compute rewards
        rewards = [compute_reward(r, p) for r, p in zip(responses, patient_batch)]
        
        # Update model
        stats = ppo_trainer.step(prompts, responses, rewards)
```

## Evaluation Metrics

### Reasoning Quality
- **Factual Accuracy**: No medical hallucinations
- **Logic Coherence**: Valid reasoning steps
- **Evidence Integration**: Uses all relevant data

### Diagnostic Performance
- **Accuracy**: Correct classifications
- **Sensitivity/Specificity**: Per-class performance
- **Calibration**: Confidence matches accuracy

### Reasoning Transparency
- **Interpretability**: Clear reasoning steps
- **Reproducibility**: Consistent logic
- **Explainability**: Justifies decisions

## Advanced Topics

### Multi-Modal CoT
Integrate different data types in reasoning:
```
Step 1 (Clinical): MMSE of 18 indicates moderate impairment
Step 2 (Biomarkers): Low Aβ42/40 suggests amyloid pathology
Step 3 (Microbiome): Elevated P. gingivalis supports AD diagnosis
Step 4 (Integration): All modalities converge on AD diagnosis
```

### Uncertainty Quantification
```python
def quantify_uncertainty(reasoning):
    """
    Extract confidence and uncertainty from reasoning.
    """
    confidence_indicators = extract_confidence_phrases(reasoning)
    evidence_strength = assess_evidence_quality(reasoning)
    reasoning_consistency = check_internal_consistency(reasoning)
    
    return {
        'confidence': aggregate_confidence(confidence_indicators),
        'epistemic_uncertainty': estimate_model_uncertainty(),
        'aleatoric_uncertainty': estimate_data_uncertainty()
    }
```

### Iterative Refinement
```python
def iterative_cot_refinement(patient_data, num_iterations=3):
    """
    Iteratively refine reasoning chain.
    """
    reasoning = initial_reasoning(patient_data)
    
    for i in range(num_iterations):
        # Identify weak points
        weak_steps = identify_weak_reasoning(reasoning)
        
        # Generate refinement
        refinement_prompt = create_refinement_prompt(reasoning, weak_steps)
        refined_reasoning = model.generate(refinement_prompt)
        
        # Evaluate improvement
        if reward(refined_reasoning) > reward(reasoning):
            reasoning = refined_reasoning
    
    return reasoning
```

## Implementation

See `cot_llm_integration.py` for complete implementation with:
- LLM-based CoT generation
- Reward modeling for medical reasoning
- PPO training for policy optimization
- Evaluation and analysis tools

## Exercise

Implement a CoT fine-tuning system that:
1. Generates step-by-step diagnostic reasoning
2. Incorporates multi-modal Alzheimer's data
3. Optimizes for both correctness and explanation quality
4. Achieves >85% accuracy with interpretable reasoning

## Practical Considerations

### Data Requirements
- Labeled patient cases with diagnoses
- Optional: Expert-annotated reasoning chains
- Large corpus for pre-training

### Computational Resources
- GPU for LLM inference and training
- Significant memory for large models
- Parallel processing for efficiency

### Safety and Ethics
- Validate medical accuracy
- Avoid harmful recommendations
- Ensure patient privacy
- Human oversight required

## Next Steps

- Explore multi-task learning across diseases
- Investigate few-shot learning for rare conditions
- Develop interactive diagnostic systems
- Integrate with clinical decision support systems

## References

- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback"
- Singhal et al. (2023). "Large Language Models Encode Clinical Knowledge"
