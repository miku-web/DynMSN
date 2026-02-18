# DynMSN: Explainable Anomaly Pinpointing via Dynamic Multi-Scale Synthesizing Network

## Overview

This repository contains the implementation of **DynMSN (Dynamic Multi-Scale Synthesizing Network)**, a novel deep learning framework for multivariate time-series anomaly detection and explainable anomaly pinpointing. DynMSN addresses the critical challenge of not only detecting anomalies in complex multivariate time-series data but also providing interpretable explanations by identifying which specific features contribute to the detected anomalies.

## Model Architecture

DynMSN integrates multiple neural network components to capture temporal dependencies, inter-feature relationships, and multi-scale patterns in time-series data:

### 1. Temporal Convolution Module
- Applies 1-D convolution along the temporal dimension to smooth input data and reduce noise
- Kernel size: configurable (default: 7)
- Helps extract local temporal patterns and mitigate the impact of measurement noise

### 2. Dual Graph Attention Networks (GAT)
DynMSN employs two parallel GAT layers to model different aspects of the data:

#### Feature-Oriented GAT
- Models dependencies and correlations among different features
- Each node represents one feature across all timestamps in the sliding window
- Learns which features are related and how they influence each other
- Uses GATv2 by default for dynamic attention computation

#### Time-Oriented GAT
- Captures temporal dependencies across different time steps
- Each node represents all features at a specific timestamp
- Learns temporal evolution patterns and time-step relationships
- Complements the feature-oriented perspective

### 3. GRU-based Sequential Encoder
- Processes the concatenated outputs from convolution and GAT modules
- Captures long-term sequential dependencies
- Hidden dimension: 150 (configurable)
- Number of layers: 1 (configurable)

### 4. Dual-Task Learning
DynMSN performs two complementary tasks simultaneously:

#### Forecasting Branch
- Predicts future time-series values
- Helps capture normal temporal progression patterns
- Anomalies manifest as large forecasting errors

#### Reconstruction Branch
- Reconstructs the input sequence
- Learns normal data patterns and distributions
- Anomalies appear as reconstruction failures

### 5. Multi-Scale Synthesis
- Dynamically combines information from different scales (convolution, GAT, GRU)
- Enables the model to capture both fine-grained and coarse-grained patterns
- Provides richer representations for anomaly detection

## Key Features

### Explainability
- **Feature-level Attribution**: Identifies which specific features contribute to detected anomalies
- **Attention Visualization**: GAT attention weights reveal feature and temporal relationships
- **Multi-scale Analysis**: Shows anomaly patterns at different temporal scales

### Advanced Techniques
- **GATv2 Implementation**: Uses dynamic attention mechanism for more expressive graph learning
- **Multiple Thresholding Methods**: 
  - Peaks-Over-Threshold (POT)
  - Dynamic thresholding
  - Brute-force optimal threshold search
- **Anomaly Scoring**: Combines forecasting and reconstruction errors with configurable weights

## Model Components

### Core Modules (`modules.py`)
- `ConvLayer`: 1-D temporal convolution
- `FeatureAttentionLayer`: Feature-oriented graph attention
- `TemporalAttentionLayer`: Time-oriented graph attention
- `GRULayer`: Sequential pattern encoder
- `Forecasting_Model`: Future value prediction
- `ReconstructionModel`: Input reconstruction

### Main Model (`mtad_gat.py`)
- Integrates all components into end-to-end architecture
- Handles forward pass and loss computation
- Supports both training and inference modes

### Training Pipeline (`train.py`, `training.py`)
- Sliding window data preparation
- Mini-batch training with Adam optimizer
- Validation-based early stopping
- TensorBoard logging support

### Prediction Pipeline (`predict.py`, `prediction.py`)
- Anomaly score computation
- Multiple threshold determination methods
- Anomaly segment detection and evaluation

## Datasets Supported

The implementation supports multiple benchmark datasets:

1. **SMD (Server Machine Dataset)**
   - 28 machines with 38 features each
   - Real-world server monitoring data
   - Specify machine with `--group` argument

2. **MSL (Mars Science Laboratory)**
   - NASA spacecraft telemetry data
   - 55 features
   - Space mission monitoring scenarios

3. **SMAP (Soil Moisture Active Passive)**
   - NASA satellite telemetry data
   - 25 features
   - Satellite health monitoring

4. **SWaT (Secure Water Treatment)**
   - Industrial control system data
   - Critical infrastructure monitoring

## Usage

### Training
```bash
python train.py --dataset <dataset_name> --epochs 30 --lookback 100
```

### Prediction and Evaluation
```bash
python predict.py --dataset <dataset_name> --load_model <model_path>
```

### Visualization
```bash
jupyter notebook result_visualizer.ipynb
```

## Configuration Parameters

### Data Parameters
- `--dataset`: Dataset name (SMD, MSL, SMAP, SWaT)
- `--group`: Machine ID for SMD dataset
- `--lookback`: Sliding window size (default: 100)
- `--normalize`: Data normalization flag

### Model Hyperparameters
- `--kernel_size`: Convolution kernel size (default: 7)
- `--use_gatv2`: Use GATv2 instead of standard GAT (default: True)
- `--gru_hid_dim`: GRU hidden dimension (default: 150)
- `--fc_n_layers`: Number of fully connected layers (default: 3)
- `--alpha`: LeakyReLU negative slope for GAT (default: 0.2)

### Training Parameters
- `--epochs`: Number of training epochs (default: 30)
- `--bs`: Batch size (default: 256)
- `--init_lr`: Initial learning rate (default: 1e-3)
- `--dropout`: Dropout rate (default: 0.3)

### Anomaly Detection Parameters
- `--gamma`: Weight for combining forecast and reconstruction errors (default: 1)
- `--q`: Quantile for POT threshold (default: 1e-3)
- `--dynamic_pot`: Use dynamic POT threshold

## Output and Results

Each training run generates:
- `model.pt`: Trained model parameters
- `summary.txt`: Performance metrics (Precision, Recall, F1-score, AUC)
- `config.txt`: Complete configuration used
- `train_scores.npy` / `test_scores.npy`: Anomaly scores
- `train.pkl` / `test.pkl`: Predictions, reconstructions, and ground truth
- `train_losses.png` / `validation_losses.png`: Loss curves

## Evaluation Metrics

- **Precision**: Proportion of detected anomalies that are true anomalies
- **Recall**: Proportion of true anomalies that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Point-wise and Range-based evaluation**: Both point-level and segment-level metrics

## Explainability Analysis

DynMSN provides multiple levels of explainability:

1. **Attention Weights**: GAT attention coefficients show feature importance
2. **Error Attribution**: Decompose anomaly scores by feature
3. **Multi-scale Contribution**: Analyze contributions from different model components
4. **Temporal Patterns**: Visualize how anomalies evolve over time

## Citation

If you use this code in your research, please cite our paper:

```
[Your paper citation will be added here]
```

## Dependencies

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter (for visualization)

See `requirements.txt` for complete dependency list.

## Repository Structure

```
.
├── args.py                 # Command-line arguments and configuration
├── train.py               # Training script
├── training.py            # Training loop implementation
├── predict.py             # Prediction script
├── prediction.py          # Prediction and evaluation logic
├── mtad_gat.py           # Main model architecture
├── modules.py            # Neural network modules
├── preprocess.py         # Data preprocessing
├── eval_methods.py       # Evaluation metrics
├── plotting.py           # Visualization utilities
├── utils.py              # Helper functions
├── datasets/             # Dataset storage
├── output/               # Training outputs and results
└── result_visualizer.ipynb  # Interactive result visualization
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository or contact the authors.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation builds upon and is inspired by:
- MTAD-GAT by Zhao et al. (2020)
- GATv2 by Brody et al. (2021)
- OmniAnomaly for preprocessing methods
- TelemAnom for evaluation and visualization approaches