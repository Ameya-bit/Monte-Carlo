# Monte Carlo Simulation of Photon Transport

This repository contains a Python script for simulating photon transport through a 1D plane-parallel slab using a Monte Carlo algorithm. The simulation models photon escape, absorption, and scattering in a simple computational physics setting.

## Features

- Simulates photon migration in a slab with user-defined optical depth and scattering probability.

- Tracks escape fraction (upward) and average number of scatterings per photon.

- Uses vectorized Python and tqdm for progress indication.

## Requirements

- Python 3 (tested with 3.9+)

- numpy

- tqdm

You can install dependencies using:

```
bash
pip install numpy tqdm
```

## How It Works

- Photons are released at the bottom of the slab and travel upwards.

- At each step, a random optical path length to the next interaction is drawn.

- If a photon escapes at the top or bottom, it's counted and the simulation moves on.

- At interactions, photons are either scattered (direction randomized) or absorbed (terminated) based on the specified albedo omega.

- Statistics are collected on photon escape and scattering events.

## Usage

To run a simulation with default parameters:

    bash
    python your_script.py

Or modify the call to run_mc() for different simulation setups:

    python
    out = run_mc(tau_tot=0.5, omega=0.9, N=50000)
    print(out)

Where:

- ```tau_tot``` = total optical depth of the slab (e.g., 0.5)

- ```omega``` = scattering probability (0.0 to 1.0)

- ```N``` = total number of photon trials (e.g., 50000)


## Output

Returns a dictionary with:

- ```escape_fraction```: fraction of photons escaping from the top

- ```mean_scatterings```: average number of scatterings per photon
