import numpy as np
import pandas as pd
from tqdm import trange

# how light travels through a foggy or dusty material (1D)
# clouds, smoke, interstellar dust
# tau_tot - opacity of the material (clear sky to thick fog)
# omega - probability of absorption
# N - number of photon packets to simulate
def run_mc(tau_tot, omega, N=200000):
    # material dimensions
    slab_thickness = 1

    kappa = tau_tot / slab_thickness
    escapes_top = 0
    total_scatterings = 0
    
    for _ in trange(N):
        # photons enter through the bottom of slab
        z = 0.0 
  
        # initialize initial direction randomly
        mu = np.random.rand()
        scat_count = 0
        alive = True
        while alive:
            # probability of traveling a distance without hitting
            # anything in the material 
            d_tau = -np.log(np.random.rand())
            dz = d_tau / kappa
            dz = d_tau / kappa

            # move the particle that distance
            # after this, the photon either escapes or hits a particle
            z += mu * dz

            # photon escapes through the bottom
            if z < 0.0:
                alive = False
                break
            # photon escapes through the top
            if z > 1.0:
                escapes_top += 1
                alive = False
                break

            # photon hits a particle
            # photon is absorbed
            if np.random.rand() > omega:
                alive = False
                break
            # photon scatters
            else:
                # new direction assigned (random scatter)
                mu = 2.0 * np.random.rand() - 1.0
                scat_count += 1
        total_scatterings += scat_count

    escape_fraction = escapes_top / N
    mean_scatterings = total_scatterings / N
    return {
    'tau_tot': tau_tot,
    'omega': omega,
    'N': N,
    'escape_fraction': escape_fraction,
    'mean_scatterings': mean_scatterings
    }


if __name__ == '__main__':  
    params = []
    results = []

    # iterates through tau values logarithmicly
    for tau in np.logspace(-1, 1, 30):
        # iterates through omega values linearly
        for omega in np.linspace(0, 0.99, 30):
            r = run_mc(tau, omega, N=50000)
            params.append([tau, omega])
            results.append([r['escape_fraction'], r['mean_scatterings']])
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'tau_tot': [p[0] for p in params],
        'omega': [p[1] for p in params],
        'escape_fraction': [r[0] for r in results],
        'mean_scatterings': [r[1] for r in results]
    })
    
    # Save to CSV
    df.to_csv('monte_carlo_results.csv', index=False)
    print(f"Results saved to monte_carlo_results.csv ({len(df)} rows)")