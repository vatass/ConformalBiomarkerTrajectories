# Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal Bands

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

This repository contains the implementation and experimental code for the paper **"Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal Bands"** accepted at NeurIPS 2025.

## Authors

- **Vasiliki Tassopoulou**$^{1,*}$ - University of Pennsylvania
- **Charis Stamouli**$^{2,*}$ - University of Pennsylvania  
- **Haochang Shou**$^{3}$ - University of Pennsylvania
- **George J. Pappas**$^{2}$ - University of Pennsylvania
- **Christos Davatzikos**$^{1}$ - University of Pennsylvania

$^{1}$ Department of Radiology, University of Pennsylvania  
$^{2}$ Department of Electrical and Systems Engineering, University of Pennsylvania  
$^{3}$ Department of Biostatistics, Epidemiology and Informatics, University of Pennsylvania  
$^{*}$ Equal contribution

## Overview

This work presents a novel approach for predicting biomarker trajectories with uncertainty quantification using conformal prediction methods. The method addresses the challenge of predicting biomarker evolution over time when measurement times are irregular and random.

## Key Features

- **Conformal Prediction**: Provides statistically valid uncertainty bands for biomarker trajectory predictions
- **Random Timing Handling**: Robust to irregular and randomly-timed biomarker measurements
- **Uncertainty Calibration**: Ensures prediction intervals maintain proper coverage guarantees
- **Biomarker Trajectory Modeling**: Specialized for longitudinal biomarker data analysis

## Repository Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                     # Source code
│   ├── models/              # Model implementations
│   ├── conformal/           # Conformal prediction methods
│   ├── data/                # Data processing utilities
│   └── utils/               # Helper functions
├── experiments/             # Experimental scripts
├── notebooks/               # Jupyter notebooks for analysis
├── data/                    # Data directory (not included in repo)
└── results/                 # Experimental results
```

## Installation



1. Clone the repository:
```bash
git clone https://github.com/yourusername/ConformalBiomarkerTrajectories.git
cd ConformalBiomarkerTrajectories
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.models.trajectory_predictor import ConformalTrajectoryPredictor
from src.conformal.conformal_bands import ConformalBands

# Initialize the predictor
predictor = ConformalTrajectoryPredictor()

# Train the model
predictor.fit(training_data)

# Generate predictions with conformal bands
predictions, bands = predictor.predict_with_uncertainty(test_data)
```

### Running Experiments

```bash
# Run main experiments
python experiments/main_experiment.py

# Run ablation studies
python experiments/ablation_studies.py
```

## Data

The experiments in this paper use longitudinal biomarker data. Due to privacy and data sharing restrictions, the actual datasets are not included in this repository. Please refer to the paper for details on the datasets used.

## Results

Key results from the paper:

- Improved uncertainty calibration compared to baseline methods
- Valid coverage guarantees for prediction intervals
- Robust performance across different biomarker types and measurement schedules

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{tassopoulou2025uncertainty,
  title={Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal Bands},
  author={Tassopoulou, Vasiliki and Stamouli, Charis and Shou, Haochang and Pappas, George J and Davatzikos, Christos},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the reviewers and the NeurIPS community for their valuable feedback. This work was supported by [funding information].

## Contact

For questions about this work, please contact:
- Vasiliki Tassopoulou: [email]
- Charis Stamouli: [email]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*This repository is associated with the NeurIPS 2025 paper "Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal Bands".*

## Environment Setup

This project requires a conda environment with specific dependencies. Here are the setup instructions:

### Method 1: Using Conda Environment File (Recommended)

```bash
# Create a new conda environment using the exported requirements
conda create --name conformal-biomarker --file conda_requirements.txt

# Activate the environment
conda activate conformal-biomarker

# Install additional Python packages
pip install -r requirements.txt
```

### Method 2: Manual Environment Creation

If the above method doesn't work, create the environment manually:

```bash
# Create a new conda environment with Python 3.8
conda create -n conformal-biomarker python=3.8
conda activate conformal-biomarker

# Install core dependencies
conda install numpy=1.22.3 pandas=1.2.3 scipy=1.9.3 scikit-learn=1.3.0
conda install pytorch=1.12.1 cpuonly -c pytorch
conda install gpytorch -c gpytorch
conda install pyyaml=6.0.2 joblib=1.4.2

# Install additional packages
pip install -r requirements.txt
```

### Key Dependencies

- **Python**: 3.8.20
- **PyTorch**: 1.12.1 (CPU version)
- **GPyTorch**: 1.10.0
- **NumPy**: 1.22.3
- **Pandas**: 1.2.3
- **SciPy**: 1.9.3
- **Scikit-learn**: 1.3.0

### Verification

Test your environment setup:

```bash
conda activate conformal-biomarker
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gpytorch; print(f'GPyTorch: {gpytorch.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```
