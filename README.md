# DynMSN: Explainable Anomaly Pinpointing via Dynamic Multi-Scale Synthesizing Network

## Overview

This repository contains the implementation of **DynMSN (Dynamic Multi-Scale Synthesizing Network)** for Multivariate Time Series Anomaly Detection (MTAD). DynMSN addresses the critical challenge of capturing dependencies across different temporal scales and feature correlations in Industrial Internet of Things (IIoT) systems. Unlike traditional methods that rely on fixed granularity levels and adjacent temporal distance modeling, DynMSN adaptively synthesizes representations across temporal scales through dynamic multi-scale partitioning and dual-attention mechanisms.

## Key Innovations

- **Dynamic Multi-Scale Synthesizing**: Adaptively synthesizes representations across temporal scales rather than relying on fixed granularity, alleviating missed or misidentified anomalies caused by pre-defined multi-scale partitions.

- **Dual-Attention Decomposition**: Explicitly models temporal distance and temporal resolution dependencies through intra-patch and inter-patch attention mechanisms, distinguishing it from conventional local/global self-attention splits.

- **Variable Attention (VAT)**: Estimates feature-wise contributions to detected anomalies, enabling more reliable anomaly pinpointing and visualization for industrial diagnosis.

- **Joint Optimization Strategy**: Leverages the complementary strengths of forecasting-based and reconstruction-based learning, where the former captures future temporal dependencies while the latter models current data distributions.

## Model Architecture

The DynMSN model consists of the following components:

### 1. Preprocessing
- **Data Normalization**: Min-max normalization to compress data into [0,1) range
- Formula: `x̃ = (T_j - min(D)) / (max(D) - min(D) + δ)`

### 2. Dilated Convolution
- Applies 1-D dilated convolution with kernel size 7 along the temporal dimension
- Expands receptive field to capture long-term dependencies without increasing computational complexity
- Extracts high-level feature information and smooths input data

### 3. Dynamic Multi-Scale Block

#### Dynamic Router
- Dynamically selects sliding window sizes and positions based on feature importance scores
- Calculates feature importance: `FS_t = MLP(X_(t-W):(t-1))`
- Optimizes window size W* and position Pos* to maximize performance

#### Multi-Scale Division
- Partitions sliding windows into patches of varying sizes
- Patch sizes P = {P_1, P_2, ..., P_n} start from 1 and increase exponentially
- Filters patch sizes based on loss scores (threshold: 0.1)
- Enables extraction of features at different temporal resolutions

#### Dual Attention Mechanism

**Intra-Patch Attention**: Captures local temporal dependencies within each patch
```
Attn_intra^i = Softmax(Q_intra^i * K_intra^i^T / √d_m) * V_intra^i
```

**Inter-Patch Attention**: Establishes global correlations between patches
```
Attn_inter = Softmax(Q_inter * K_inter^T / √d_m') * V_inter
```

**Variable Attention (VAT)**: Captures feature correlations across different temporal scales
- Uses deformable attention with learnable offsets: `Δp = tanh(Q_p)`
- Applies bilinear interpolation for sampling: `Z_d = ϖ(Z_p; p + Δp)`
- Computes attention output: `A_i = (Softmax(Q_p * K_d^T / √d) * V_d) * W_i`

#### Multi-Scale Aggregator
- Aligns time dimensions across different scales using transformation function T(·)
- Aggregates multi-scale outputs with learned pathway weights
- Formula: `X_out = (1/N) * Σ P̄(X_trans)_i * T(X_out^i)`

### 4. Feature and Temporal GAT Layers
- **Feature-Oriented GAT**: Models dependencies among different features
- **Time-Oriented GAT**: Captures temporal dependencies across time steps
- Both use GATv2 by default for dynamic attention computation

### 5. GRU Layer
- Processes concatenated outputs from convolution, GAT, and multi-scale blocks
- Captures long-term sequential patterns
- Hidden dimension: 150 (configurable)

### 6. Joint Optimization Modules

#### Forecasting Module
- Predicts future time-series values using MLP
- Loss function: `Loss_for = RMSE = √((1/n×k) * Σ(x_r,t - x̂_r,t)²)`

#### Reconstruction Module
- Reconstructs input sequence using Variational Autoencoder (VAE)
- Loss function: `Loss_rec = E[log p_θ(x|z)] + D_KL(q_φ(z|x) || p(z))`

#### Anomaly Score
- Combines forecasting and reconstruction errors
- Formula: `Score = Σ S_r = Σ ((x_r - x̂_r)² + γ×(1-p_r)) / (1+γ)`
- Hyperparameter γ ∈ {0.2, 0.4, 0.6, 0.8} is dynamically selected per dataset

#### Anomaly Inference
- Uses SPOT (Streaming Peaks-Over-Threshold) method for automatic threshold determination
- Label(Score) = 1 if Score ≥ SPOT(Score), else 0

## Datasets

The implementation supports six public industrial datasets:

| Dataset | Dimensions | Training | Testing | Data Volume/Rate |
|---------|-----------|----------|---------|------------------|
| ASD     | 19        | 102,331  | 51,840  | 154,171/4.61%   |
| SMD     | 38        | 708,405  | 708,420 | 1,416,825/4.2%  |
| SWaT    | 51        | 496,736  | 449,855 | 946,719/12.1%   |
| PSM     | 25        | 132,481  | 87,841  | 220,322/27.8%   |
| MSL     | 55        | 58,317   | 73,729  | 132,046/10.27%  |
| SMAP    | 25        | 135,183  | 427,617 | 562,800/13.13%  |

## Usage

### Training
```bash
python train.py --dataset <dataset_name> --epochs 30 --lookback 32 --patch_size 4
```

### Prediction
```bash
python predict.py --dataset <dataset_name>
```

### Preprocessing
```bash
python preprocess.py --dataset <dataset_name>
```

## Configuration Parameters

### Data Parameters
- `--dataset`: Dataset name (ASD, SMD, SWaT, PSM, MSL, SMAP)
- `--group`: Machine ID for SMD dataset (e.g., "1-1")
- `--lookback`: Sliding window size (default: 32)
- `--normalize`: Data normalization flag (default: True)

### Model Hyperparameters
- `--kernel_size`: Dilated convolution kernel size (default: 7)
- `--use_gatv2`: Use GATv2 instead of standard GAT (default: True)
- `--feat_gat_embed_dim`: Feature GAT embedding dimension
- `--time_gat_embed_dim`: Temporal GAT embedding dimension
- `--gru_n_layers`: Number of GRU layers (default: 1)
- `--gru_hid_dim`: GRU hidden dimension (default: 150)
- `--fc_n_layers`: Number of forecasting FC layers (default: 3)
- `--fc_hid_dim`: Forecasting hidden dimension (default: 150)
- `--recon_n_layers`: Number of reconstruction layers (default: 1)
- `--recon_hid_dim`: Reconstruction hidden dimension (default: 150)
- `--alpha`: LeakyReLU negative slope (default: 0.2)
- `--activation`: Activation function (default: 'PReLU')
- `--patch_size`: Multi-scale patch sizes (default: [4])

### Training Parameters
- `--epochs`: Number of training epochs (default: 1)
- `--val_split`: Validation split ratio (default: 0.1)
- `--bs`: Batch size (default: 128)
- `--init_lr`: Initial learning rate (default: 1e-3)
- `--shuffle_dataset`: Shuffle training data (default: True)
- `--dropout`: Dropout rate (default: 0.3)
- `--use_cuda`: Use GPU acceleration (default: True)
- `--print_every`: Print frequency (default: 1)
- `--log_tensorboard`: Enable TensorBoard logging (default: True)

### Anomaly Detection Parameters
- `--gamma`: Weight for combining forecast and reconstruction errors (default: 1)
- `--level`: Threshold level for anomaly detection
- `--q`: Quantile for dynamic threshold
- `--dynamic_pot`: Use dynamic POT threshold (default: False)
- `--scale_scores`: Scale anomaly scores (default: False)
- `--use_mov_av`: Use moving average (default: False)

## Evaluation Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × Precision × Recall / (Precision + Recall)
- **AUC**: Area Under the ROC Curve
- **HitRate@P%**: Proportion of actual abnormal points among top P% detected points
- **NDCG@100%**: Normalized Discounted Cumulative Gain for ranking capability assessment

## Repository Structure

```
code/
├── dynmsn.py              # Main DynMSN model implementation
├── dynamic_router.py      # Dynamic router and window selector
├── modules.py             # Core neural network modules (Conv, GAT, GRU, etc.)
├── args.py                # Command-line argument parser
├── train.py               # Training script
├── training.py            # Training loop implementation
├── predict.py             # Prediction script
├── prediction.py          # Prediction and evaluation logic
├── preprocess.py          # Data preprocessing utilities
├── eval_methods.py        # Evaluation metrics implementation
├── spot.py                # SPOT threshold determination
├── plotting.py            # Visualization utilities
├── utils.py               # Helper functions
├── models/                # Additional model components
└── requirements.txt       # Python dependencies
```

## Experimental Setup

- **Hardware**: AMD Ryzen 7 3700X 8-Core CPU, 32GB RAM, NVIDIA GeForce RTX 3050 GPU (8GB)
- **Software**: Python 3.9.11, PyTorch 1.10.1, CUDA 12.4
- **Activation Function**: LeakyReLU
- **Dropout Rate**: 0.2 during training
- **Batch Size**: 128

## Time Complexity

The overall time complexity of DynMSN is O(n·d·h + n·d·W + n·d·S + n·d²), where:
- n: number of time steps
- d: feature dimension
- h: hidden layer size
- W: window size
- S: sliding window size

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## Contact

For questions or issues, please contact the authors or open an issue in the repository.

## Acknowledgments

This work builds upon research in multivariate time series anomaly detection and incorporates techniques from:
- Graph Attention Networks (GAT and GATv2)
- Variational Autoencoders (VAE)
- SPOT algorithm for threshold determination