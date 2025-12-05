# Monte Carlo Photon Transport & Neural Network Emulator

This repository contains a comprehensive suite for simulating and emulating photon transport through scattering media. It includes both **1D slab** and **3D volumetric** simulations, along with neural network models trained to predict simulation outcomes instantly.

## Project Structure

```
Monte-Carlo/
├── 1D simulations/             # Original 1D slab model
│   ├── simulations.py          # 1D Monte Carlo simulator
│   ├── mc_nn.ipynb             # Neural network emulator for 1D
│   └── monte_carlo_results.csv # 1D dataset
├── 3D simulations/             # Advanced 3D model
│   ├── simulations_3d.py       # 3D Monte Carlo simulator
│   ├── verify_3d.py            # Verification script for 3D logic
│   ├── mc_nn_3d.ipynb          # Neural network emulator for 3D
│   └── monte_carlo_results_3d.csv # 3D dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Part 1: 1D Simulation (`1D simulations/`)

### Physics Model
The 1D model simulates photon propagation through a semi-infinite slab (e.g., a layer of fog or dust).
- **Geometry**: Photons enter at $z=0$ and travel in a 1D vertical space ($z \in [0, 1]$).
- **Scattering**: Isotropic scattering based on albedo ($\omega$).
- **Outcomes**: Photons either escape through the top ($z>1$), reflect back bottom ($z<0$), or are absorbed.

### Usage
Run the simulation to generate data:
```bash
cd "1D simulations"
python simulations.py
```

Train the neural network:
Open `mc_nn.ipynb` and run the cells to train a model that predicts `escape_fraction` and `mean_scatterings` from `tau` and `omega`.

---

## Part 2: 3D Simulation (`3D simulations/`)

### Physics Model
The 3D model extends the simulation to full 3D space with more complex scattering physics.
- **Geometry**: Photons move in $(x, y, z)$ space. The medium is a slab between $z=0$ and $z=1$, infinite in $x, y$.
- **Scattering Phase Function**: Uses the **Henyey-Greenstein** phase function, controlled by the asymmetry parameter $g$:
    - $g = 0$: Isotropic scattering (like 1D).
    - $g > 0$: Forward scattering (e.g., biological tissue, clouds).
    - $g < 0$: Backward scattering.
- **Metrics**: Tracks top escape, bottom escape, and absorption fractions.

### Key Files
- `simulations_3d.py`: The core engine. Defines a `Photon` class with 3D vector math for movement and scattering.
- `verify_3d.py`: Runs automated checks (energy conservation, asymmetry trends) to validate the physics.

### Usage
Run the simulation:
```bash
cd "3D simulations"
python simulations_3d.py
```
This will generate `monte_carlo_results_3d.csv` with results for various combinations of optical depth ($\tau$), albedo ($\omega$), and asymmetry ($g$).

Verify the physics:
```bash
python verify_3d.py
```

Train the 3D emulator:
Open `mc_nn_3d.ipynb` to train a network that predicts 3D transport outcomes.

---

## Neural Network Emulators

Both directories contain Jupyter notebooks (`mc_nn.ipynb` and `mc_nn_3d.ipynb`) that demonstrate how to:
1.  Load the Monte Carlo datasets.
2.  Train a PyTorch Feedforward Neural Network.
3.  Use the trained model to replace expensive Monte Carlo runs with millisecond-fast predictions.

### 3D Model Architecture
- **Inputs**: Optical Depth ($\tau$), Albedo ($\omega$), Asymmetry ($g$).
- **Outputs**: Top Escape Fraction, Bottom Escape Fraction, Absorbed Fraction.

---

## Requirements

Install the necessary Python packages:
```bash
pip install numpy pandas tqdm torch scikit-learn jupyter
```
