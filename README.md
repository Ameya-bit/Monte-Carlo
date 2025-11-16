# Monte Carlo Photon Transport & Neural Network Emulator

This repository contains two complementary components for studying photon transport through scattering media:
1. **Monte Carlo Simulation** (`simulations.py`) - Physics-based photon transport simulator
2. **Neural Network Emulator** (`mc_nn.ipynb`) - PyTorch model that learns to predict simulation outcomes

## Overview

The Monte Carlo simulation models photon propagation through a 1D slab (representing fog, dust, or interstellar clouds), tracking escape probabilities and scattering statistics. The neural network learns these relationships from simulation data, enabling fast predictions without expensive Monte Carlo runs.

---

## Part 1: Monte Carlo Simulation (`simulations.py`)

### Physics Model

The code simulates how photons travel through a dusty or foggy material in 1D:

- **Optical Depth (`tau_tot`)**: Measures material opacity (0.1 = clear, 10 = thick fog)
- **Albedo (`omega`)**: Scattering probability (0 = all absorbed, 0.99 = mostly scattered)
- **Slab Geometry**: Photons enter at bottom (z=0) and may escape through top (z=1) or bottom

### How the Simulation Works

```python
def run_mc(tau_tot, omega, N=200000):
```

For each of N photon packets:

1. **Initialization**: Photon starts at z=0 with random upward direction (`mu`)

2. **Free Flight**: Travels a random distance drawn from exponential distribution:
   ```python
   d_tau = -np.log(np.random.rand())  # Optical path length
   dz = d_tau / kappa  # Physical distance
   z += mu * dz  # Update position
   ```

3. **Boundary Check**: 
   - If `z > 1.0`: Escapes through top (counted as success)
   - If `z < 0.0`: Escapes through bottom (lost)

4. **Interaction**: If photon hits a particle inside the slab:
   - **Absorbed** with probability `(1 - omega)`: Photon terminates
   - **Scattered** with probability `omega`: New random direction assigned, continues

5. **Statistics**: Tracks total escapes and scattering events

### Dataset Generation

The main script generates a comprehensive dataset:

```python
for tau in np.logspace(-1, 1, 30):      # 30 optical depths (0.1 to 10)
    for omega in np.linspace(0, 0.99, 30):  # 30 albedos (0 to 0.99)
        r = run_mc(tau, omega, N=50000)     # 50,000 photons per config
```

Creates **900 simulation runs** covering the parameter space, saved to `monte_carlo_results.csv`.

### Output Metrics

- `escape_fraction`: Fraction of photons escaping through the top
- `mean_scatterings`: Average number of scattering events per photon

### Requirements

```bash
pip install numpy pandas tqdm
```

### Usage

Generate the dataset:
```bash
python simulations.py
```

Run a single simulation:
```python
result = run_mc(tau_tot=1.0, omega=0.9, N=100000)
print(f"Escape fraction: {result['escape_fraction']:.3f}")
print(f"Mean scatterings: {result['mean_scatterings']:.2f}")
```

---

## Part 2: Neural Network Emulator (`mc_nn.ipynb`)

### Purpose

Train a neural network to instantly predict Monte Carlo outcomes (which normally take minutes to compute), enabling real-time parameter exploration.

### Architecture

**Feedforward Neural Network:**
- Input layer: 2 neurons (`tau_tot`, `omega`)
- Hidden layer 1: 64 neurons + ReLU activation
- Hidden layer 2: 64 neurons + ReLU activation  
- Output layer: 2 neurons (`escape_fraction`, `mean_scatterings`)

```python
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
```

### Training Pipeline

1. **Data Loading & Preprocessing**
   ```python
   data = pd.read_csv('monte_carlo_results.csv')
   X = data[['tau_tot','omega']].values
   Y = data[['escape_fraction','mean_scatterings']].values
   ```

2. **Train/Test Split**: 80% training, 20% testing

3. **Normalization**: Standardizes inputs to zero mean and unit variance
   ```python
   scaler = StandardScaler().fit(X_train)
   X_train_s = scaler.transform(X_train)
   ```

4. **Validation Split**: 10% of training data held out for validation

5. **Training Loop** (200 epochs, batch size 32):
   - Forward pass through network
   - Compute MSE loss
   - Backpropagation to calculate gradients
   - Adam optimizer updates weights
   - Validation metrics computed each epoch

6. **Evaluation**:
   - Mean Absolute Error (MAE) on test set
   - Median percent errors for each output

### Key Implementation Details

**PyTorch Components:**
- `nn.MSELoss()`: Mean Squared Error for regression
- `optim.Adam()`: Adaptive learning rate optimizer
- `DataLoader`: Mini-batch processing with shuffling
- `model.train()` / `model.eval()`: Switch between training and evaluation modes
- `torch.no_grad()`: Disable gradient tracking during validation/testing

**Training Features:**
- Batch processing (32 samples per batch)
- Validation monitoring to prevent overfitting
- Progress tracking every 20 epochs

### Performance Metrics

The model reports:
- **Test MAE**: Average absolute prediction error
- **Median percent errors**: Relative accuracy for each output variable

### Requirements

```bash
pip install numpy pandas scikit-learn torch tqdm
```

### Usage

1. Generate training data (if not already done):
   ```bash
   python simulations.py
   ```

2. Open and run `mc_nn.ipynb` in Jupyter/VS Code

3. The trained model can make instant predictions:
   ```python
   # New input: tau_tot=2.5, omega=0.85
   new_input = scaler.transform([[2.5, 0.85]])
   prediction = model(torch.FloatTensor(new_input))
   ```

---

## Workflow Summary

1. **Generate Data**: Run `simulations.py` to create `monte_carlo_results.csv` (900 parameter combinations)
2. **Train Model**: Execute `mc_nn.ipynb` to train the neural network emulator
3. **Fast Predictions**: Use trained model for real-time photon transport predictions

## Applications

- Astrophysics: Modeling light through interstellar dust clouds
- Atmospheric science: Photon transport in clouds and fog
- Medical imaging: Light propagation in tissue
- Computer graphics: Rendering subsurface scattering

---

## Repository Structure

```
Monte-Carlo/
├── simulations.py              # Monte Carlo photon transport simulator
├── mc_nn.ipynb                 # Neural network training notebook
├── monte_carlo_results.csv     # Generated dataset (900 rows)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
