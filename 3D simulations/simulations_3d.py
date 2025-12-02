import numpy as np
import pandas as pd
from tqdm import tqdm


class Photon:
    def __init__(self):
        # position
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        # direction
        phi = 2 * np.pi * np.random.rand()
        mu = np.random.rand()
        
        # determine what angle for initial direction
        # isotropic emission
        sin_theta = np.sqrt(1 - mu**2)
        self.ux = sin_theta * np.cos(phi)
        self.uy = sin_theta * np.sin(phi)
        self.uz = mu
        
        self.alive = True
        self.trajectory = [(self.x, self.y, self.z)]

    def move(self, distance):
        # move a specified distance
        # use current direction
        self.x += self.ux * distance
        self.y += self.uy * distance
        self.z += self.uz * distance
        
        # Append new position to trajectory
        self.trajectory.append((self.x, self.y, self.z))

    def check_boundaries(self):
        # check if photon escaped
        if self.z < 0.0 or self.z > 1.0:
            self.alive = False

    def scatter(self, g):
        """
        Update direction based on Henyey-Greenstein phase function.
        g: Asymmetry parameter (-1 to 1). g=0 is isotropic.
        """
        # 1. Sample scattering angle theta (represented by mu_s = cos(theta))
        if g == 0:
            mu_s = 2 * np.random.rand() - 1
        else:
            # Inverse CDF of Henyey-Greenstein
            xi = np.random.rand()
            term = (1 - g**2) / (1 - g + 2 * g * xi)
            mu_s = (1 / (2 * g)) * (1 + g**2 - term**2)
            
        # Clamp to handle numerical precision issues
        mu_s = max(-1.0, min(1.0, mu_s))
        
        # 2. Sample azimuthal angle phi_s
        phi_s = 2 * np.pi * np.random.rand()
        
        sin_theta_s = np.sqrt(1 - mu_s**2)
        cos_phi_s = np.cos(phi_s)
        sin_phi_s = np.sin(phi_s)

        # 3. Rotate current direction (ux, uy, uz) by (theta_s, phi_s)
        # Using standard rotation formulas for MC transport
        
        # Check if current direction is close to vertical to avoid division by zero
        if abs(self.uz) > 0.99999:
            # Special case: vertical incidence
            self.ux = sin_theta_s * cos_phi_s
            self.uy = sin_theta_s * sin_phi_s
            self.uz = np.sign(self.uz) * mu_s
        else:
            sqrt_term = np.sqrt(1 - self.uz**2)
            
            # Temporary variables for new direction
            new_ux = (sin_theta_s / sqrt_term) * (self.ux * self.uz * cos_phi_s - self.uy * sin_phi_s) + self.ux * mu_s
            new_uy = (sin_theta_s / sqrt_term) * (self.uy * self.uz * cos_phi_s + self.ux * sin_phi_s) + self.uy * mu_s
            new_uz = -sin_theta_s * cos_phi_s * sqrt_term + self.uz * mu_s
            
            self.ux = new_ux
            self.uy = new_uy
            self.uz = new_uz

        # Re-normalize to ensure unit vector (corrects drift over many scatters)
        norm = np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
        self.ux /= norm
        self.uy /= norm
        self.uz /= norm

def run_mc_3d(tau_tot, omega, g, N=10000):
    """
    Run 3D Monte Carlo simulation.
    tau_tot: Optical depth of slab
    omega: Single scattering albedo (prob of scatter vs absorption)
    g: Asymmetry parameter
    N: Number of photons
    """
    slab_thickness = 1.0
    kappa = tau_tot / slab_thickness
    
    escapes_top = 0
    escapes_bottom = 0
    absorbed_count = 0
    
    # Store final positions of escaping photons for analysis
    exit_positions_top = []
    
    for _ in range(N):
        p = Photon()
        
        while p.alive:
            # 1. Sample distance to next interaction
            # d_tau = -ln(rand)
            d_tau = -np.log(np.random.rand())
            distance = d_tau / kappa
            
            # 2. Move photon
            p.move(distance)
            
            # 3. Check boundaries
            p.check_boundaries()
            if not p.alive:
                if p.z > 1.0:
                    escapes_top += 1
                    exit_positions_top.append((p.x, p.y))
                else:
                    escapes_bottom += 1
                break
                
            # 4. Interaction (Scatter or Absorb)
            if np.random.rand() > omega:
                # Absorbed
                p.alive = False
                absorbed_count += 1
            else:
                # Scattered
                p.scatter(g)
                
    return {
        'tau_tot': tau_tot,
        'omega': omega,
        'g': g,
        'N': N,
        'escape_fraction_top': escapes_top / N,
        'escape_fraction_bottom': escapes_bottom / N,
        'absorbed_fraction': absorbed_count / N,
        'exit_positions_top': exit_positions_top
    }

if __name__ == '__main__':
    params = []
    results = []
    
    # Define parameter ranges
    # tau: 0.1 to 10 (logarithmic)
    taus = np.logspace(-1, 1, 10)
    # omega: 0.0 to 0.99
    omegas = np.linspace(0, 0.99, 10)
    # g: 0 (isotropic), 0.5, 0.9 (forward scattering)
    gs = [0, 0.5, 0.9]
    
    total_sims = len(taus) * len(omegas) * len(gs)
    print(f"Starting {total_sims} simulations...")
    
    with tqdm(total=total_sims) as pbar:
        for g in gs:
            for tau in taus:
                for omega in omegas:
                    # Run simulation with fewer photons for speed during sweep
                    r = run_mc_3d(tau, omega, g, N=5000)
                    
                    params.append([tau, omega, g])
                    results.append([
                        r['escape_fraction_top'], 
                        r['escape_fraction_bottom'], 
                        r['absorbed_fraction']
                    ])
                    pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'tau_tot': [p[0] for p in params],
        'omega': [p[1] for p in params],
        'g': [p[2] for p in params],
        'escape_fraction_top': [r[0] for r in results],
        'escape_fraction_bottom': [r[1] for r in results],
        'absorbed_fraction': [r[2] for r in results]
    })
    
    # Save to CSV
    output_file = 'monte_carlo_results_3d.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file} ({len(df)} rows)")
