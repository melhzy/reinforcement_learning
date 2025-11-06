# Tutorial 5: Alzheimer's Analysis with Multi-Modal Data

## Overview

This tutorial provides a comprehensive implementation of RL for Alzheimer's disease classification using multi-modal data including clinical variables, microbiome data, and blood biomarkers.

## Multi-Modal Data Integration

### Data Modalities

#### 1. Clinical Variables
Essential baseline information:
- **Demographics**: Age, sex, education level
- **Cognitive Assessments**:
  - MMSE (Mini-Mental State Examination): 0-30 scale
  - CDR (Clinical Dementia Rating): 0-3 scale
  - ADAS-Cog (Alzheimer's Disease Assessment Scale)
- **Medical History**: Family history, comorbidities
- **Lifestyle**: Physical activity, diet, social engagement

#### 2. Oral and Gut Microbiome
Emerging biomarkers in AD research:

**Oral Microbiome**:
- Porphyromonas gingivalis (linked to AD)
- Treponema denticola
- Tannerella forsythia
- Total bacterial load

**Gut Microbiome**:
- Firmicutes/Bacteroidetes ratio
- Lactobacillus abundance
- Bifidobacterium levels
- Escherichia coli
- Diversity indices (Shannon, Simpson)

#### 3. Blood Biomarkers
Key proteins in AD:

**Core AD Biomarkers**:
- **Aβ42** (Amyloid-beta 42): Plaque formation
- **Aβ40** (Amyloid-beta 40): Reference marker
- **Aβ42/Aβ40 ratio**: Better diagnostic value
- **t-tau** (Total tau): Neuronal damage
- **p-tau181** (Phosphorylated tau): Tangle formation
- **p-tau217**: High specificity for AD

**Additional Markers**:
- Neurofilament light (NfL): Neurodegeneration
- GFAP (Glial fibrillary acidic protein): Astrocyte activation
- Inflammatory markers: IL-6, TNF-α, CRP

## RL Framework for AD Classification

### Problem Formulation

**State Space** (Multi-modal Patient Profile):
```python
state = {
    'clinical': [age, mmse, cdr, ...],          # 10 features
    'oral_microbiome': [bacteria_1, ...],       # 20 features
    'gut_microbiome': [bacteria_1, ...],        # 30 features  
    'blood_biomarkers': [Aβ42, tau, ...],       # 15 features
    'available_tests': [bool, bool, ...],       # Which tests done
    'confidence': float,                         # Current confidence
    'cost_spent': float                          # Cost so far
}
```

**Action Space**:
- Test ordering actions (6):
  - Order clinical assessment
  - Order oral microbiome test
  - Order gut microbiome test
  - Order blood biomarker panel
  - Order advanced imaging
  - Order genetic testing
  
- Classification actions (3):
  - Classify as Cognitively Normal (CN)
  - Classify as Mild Cognitive Impairment (MCI)
  - Classify as Alzheimer's Disease (AD)

**Reward Function**:
```python
def compute_reward(action, outcome):
    if action in [classify_CN, classify_MCI, classify_AD]:
        if correct_classification:
            base_reward = 100
            cost_penalty = total_cost * 0.5
            confidence_bonus = confidence_level * 10
            return base_reward - cost_penalty + confidence_bonus
        else:
            return -100 - total_cost * 0.5
    else:  # Test ordering
        test_costs = {
            'clinical': 10,
            'oral_microbiome': 50,
            'gut_microbiome': 75,
            'blood_biomarkers': 100,
            'imaging': 500,
            'genetic': 1000
        }
        return -test_costs[action]
```

## Implementation Details

### Feature Preprocessing

```python
def preprocess_features(patient_data):
    """Normalize and prepare multi-modal features."""
    
    # Clinical features
    clinical = normalize_clinical(patient_data['clinical'])
    
    # Microbiome features (compositional data)
    oral_micro = clr_transform(patient_data['oral_microbiome'])
    gut_micro = clr_transform(patient_data['gut_microbiome'])
    
    # Biomarker features (log-transform skewed distributions)
    biomarkers = log_normalize(patient_data['blood_biomarkers'])
    
    return concatenate([clinical, oral_micro, gut_micro, biomarkers])
```

### Model Architecture

```python
class MultiModalDQN(nn.Module):
    """
    DQN with separate encoders for each modality.
    """
    
    def __init__(self):
        super().__init__()
        
        # Modality-specific encoders
        self.clinical_encoder = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU()
        )
        
        self.oral_encoder = nn.Sequential(
            nn.Linear(20, 32), nn.ReLU()
        )
        
        self.gut_encoder = nn.Sequential(
            nn.Linear(30, 32), nn.ReLU()
        )
        
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(15, 32), nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
```

## Real-World Considerations

### 1. Missing Data
Not all patients have all tests:
- Handle with masking mechanisms
- Imputation strategies
- Uncertainty estimation

### 2. Class Imbalance
AD/MCI/CN distribution may be skewed:
- Weighted loss functions
- Oversampling/undersampling
- Focal loss

### 3. Temporal Dynamics
Disease progression over time:
- Longitudinal data integration
- Recurrent architectures (LSTM/GRU)
- Time-series RL

### 4. Interpretability
Medical decisions must be explainable:
- Attention mechanisms
- Feature importance
- Decision trees for policy visualization

## Example: Complete Workflow

```python
from alzheimer_multimodal_rl import AlzheimerEnv, MultiModalAgent

# Load real patient data
data = load_alzheimer_dataset()

# Create environment
env = AlzheimerEnv(
    patient_data=data,
    feature_config={
        'clinical': 10,
        'oral_microbiome': 20,
        'gut_microbiome': 30,
        'blood_biomarkers': 15
    }
)

# Create agent with multi-modal architecture
agent = MultiModalAgent(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    use_attention=True
)

# Train
agent.train(env, episodes=10000)

# Evaluate on test set
results = agent.evaluate(test_data)
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Sensitivity (AD): {results['sensitivity_AD']:.2%}")
print(f"Specificity: {results['specificity']:.2%}")
print(f"Average Cost: ${results['avg_cost']:.2f}")
```

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Sensitivity/Recall**: True positive rate per class
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Efficiency Metrics
- **Average # of Tests**: How many tests ordered
- **Average Cost**: Total diagnostic cost
- **Time to Diagnosis**: Number of steps

### Clinical Metrics
- **Early Detection Rate**: MCI → AD transitions caught
- **False Positive Rate**: Especially important for AD
- **Cost per Correct Diagnosis**

## Research Directions

### Transfer Learning
- Pre-train on large AD datasets
- Fine-tune on specific populations

### Federated Learning
- Train across multiple institutions
- Privacy-preserving collaboration

### Multi-Task Learning
- Predict AD risk + progression rate
- Classify AD + recommend treatment

## Implementation

See `alzheimer_multimodal_rl.py` for complete implementation.

## Exercise

Build an RL agent that:
1. Processes multi-modal Alzheimer's data
2. Learns optimal test ordering strategy
3. Achieves >80% accuracy with <$200 average cost
4. Provides interpretable decisions

## Next Steps

Proceed to Tutorial 6: CoT Integration for LLM Fine-tuning to learn how to use RL for training language models on medical reasoning tasks.
