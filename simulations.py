import numpy as np
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
    out = run_mc(0.5, 0.9, N=50000)
    print(out)