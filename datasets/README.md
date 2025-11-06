# Example Datasets

This directory contains example datasets for use with the reinforcement learning tutorials.

## Alzheimer's Disease Dataset

**File**: `example_alzheimer_data.csv`

A synthetic dataset for Alzheimer's disease classification with multi-modal features.

### Features

#### Clinical Variables (4)
- `age`: Patient age in years
- `mmse`: Mini-Mental State Examination score (0-30)
- `cdr`: Clinical Dementia Rating (0, 0.5, 1.0, 1.5+)
- `education`: Years of education

#### Blood Biomarkers (4)
- `abeta42`: Amyloid-beta 42 levels (pg/mL)
- `abeta40`: Amyloid-beta 40 levels (pg/mL)
- `tau`: Total tau protein (pg/mL)
- `ptau181`: Phosphorylated tau 181 (pg/mL)

#### Microbiome Indicators (2)
- `p_gingivalis`: Porphyromonas gingivalis abundance (oral microbiome)
- `fb_ratio`: Firmicutes/Bacteroidetes ratio (gut microbiome)

### Target Variable
- `diagnosis`: Classification label
  - `CN`: Cognitively Normal
  - `MCI`: Mild Cognitive Impairment
  - `AD`: Alzheimer's Disease

### Biomarker Interpretations

#### Normal (CN) Ranges
- MMSE: 27-30
- CDR: 0
- Aβ42: > 600 pg/mL
- Aβ42/Aβ40 ratio: > 0.08
- p-tau181: < 40 pg/mL

#### MCI Ranges
- MMSE: 20-26
- CDR: 0.5
- Aβ42: 450-600 pg/mL
- Aβ42/Aβ40 ratio: 0.06-0.08
- p-tau181: 40-60 pg/mL

#### AD Ranges
- MMSE: < 20
- CDR: ≥ 1.0
- Aβ42: < 450 pg/mL
- Aβ42/Aβ40 ratio: < 0.06
- p-tau181: > 60 pg/mL

### Usage Example

```python
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('datasets/example_alzheimer_data.csv')

# Extract features and labels
features = data[['age', 'mmse', 'cdr', 'education', 
                 'abeta42', 'abeta40', 'tau', 'ptau181',
                 'p_gingivalis', 'fb_ratio']].values

# Encode labels
label_map = {'CN': 0, 'MCI': 1, 'AD': 2}
labels = data['diagnosis'].map(label_map).values

print(f"Dataset shape: {features.shape}")
print(f"Class distribution:")
print(data['diagnosis'].value_counts())
```

## Important Notes

### Data Source
This is **synthetic data** created for educational purposes. For research on real Alzheimer's disease data, please refer to:
- ADNI (Alzheimer's Disease Neuroimaging Initiative)
- NACC (National Alzheimer's Coordinating Center)
- UK Biobank

### Privacy
No real patient data is included in this repository. All examples are synthetically generated.

### Citation
If you use the methodology or code structure in your research:
```bibtex
@misc{reinforcement_learning_tutorials,
  author = {Huang, Ziyuan},
  title = {Reinforcement Learning Tutorials for Medical AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/melhzy/reinforcement_learning}
}
```

## Extending the Dataset

To create your own synthetic dataset with more samples:

```python
from utils.utils import create_synthetic_alzheimer_features
import pandas as pd

# Generate samples for each class
data = []
for label, label_name in enumerate(['CN', 'MCI', 'AD']):
    features = create_synthetic_alzheimer_features(100, label)
    for i, feat in enumerate(features):
        row = {
            'patient_id': f'{label_name}_{i:03d}',
            'age': feat[0],
            'mmse': feat[1],
            'cdr': feat[2],
            'education': feat[3],
            'abeta42': feat[4],
            'abeta40': feat[5],
            'tau': feat[6],
            'ptau181': feat[7],
            'p_gingivalis': feat[8],
            'fb_ratio': feat[9],
            'diagnosis': label_name
        }
        data.append(row)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('datasets/large_alzheimer_dataset.csv', index=False)
```
