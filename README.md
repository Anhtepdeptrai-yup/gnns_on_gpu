# Getting started with GNNs on NPUs

This repository provides tools and scripts to train, convert, and run inference on Graph Neural Networks (GNNs) using the Cora, Citeseer, and PubMed datasets. Leveraging OpenVINO's Intermediate Representation (IR) models ensures efficient deployment across multiple devices.

---

## Installation

Follow the steps below to set up the environment:

```bash
# Create and activate a virtual environment
conda create -n gnn
conda activate gnn

# Install PyTorch and Torch Geometric
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install torch_geometric

# Install additional dependencies
pip install openvino onnx scipy
```

---

## Workflow Steps

1. **Train the Model**
   - **Script**: `train_gcn.py`
   - **Description**: Trains GCN models using the Cora, Citeseer, or PubMed dataset and saves the trained PyTorch model.
   - **Output**: Saved PyTorch models in the `torch_models/` directory.

2. **Convert PyTorch Model to OpenVINO IR**
   - **Script**: `convert.py`
   - **Description**: Converts the trained PyTorch models into OpenVINO IR format for optimized inference.
   - **Output**: Converted IR models in the `ov_model/` directory.

3. **Benchmarking**
   - **Script**: `benchmark.py`
   - **Description**: Benchmarks the performance of OpenVINO IR models. Reports detailed latency and throughput metrics.

---

## Directory Structure

```plaintext
├── torch_models/       # Trained PyTorch models
├── ov_model/           # OpenVINO IR models
├── train_gcn.py        # Script to train the model
├── convert.py          # Script to convert PyTorch models to OpenVINO IR
├── benchmark.py        # Script for performance benchmarking
└── README.md           # Documentation
```

---

## Example Usage

1. **Train a GCN Model**:
   ```bash
   python train_gcn.py
   ```

2. **Convert the Model**:
   ```bash
   python convert.py
   ```

3. **Benchmark Performance**:
   ```bash
   python benchmark.py
   ```