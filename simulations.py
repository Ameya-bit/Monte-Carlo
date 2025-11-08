import numpy as np
from tqdm import trange

def run_mc(tau_tot, omega, N=200000):
    # slab thickness = 1. kappa maps tau to length: kappa = tau_tot / 1 =
    tau_tot
    kappa = tau_tot
    escapes_top = 0
    total_scatterings = 0
    
    for _ in trange(N):
        z = 0.0 # start at bottom
        # initial upward direction mu in (0,1)
        mu = np.random.rand()
        scat_count = 0
        alive = True
        while alive:
            # draw optical depth to next interaction
            d_tau = -np.log(np.random.rand())
            # convert to physical distance (since kappa = tau/L and L=1) =>
            dz = d_tau / kappa
            dz = d_tau / kappa
            z += mu * dz
            if z < 0.0:
                # escaped downward (ignore or count)
                alive = False
                break
            if z > 1.0:
                # escaped upward
                escapes_top += 1
                alive = False
                break
            # interaction occurs inside slab
            if np.random.rand() > omega:
                # absorbed
                alive = False
                break
            else:
                # scattered: choose new isotropic direction
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
# quick example
if __name__ == '__main__':
    out = run_mc(0.5, 0.9, N=50000)
    print(out)