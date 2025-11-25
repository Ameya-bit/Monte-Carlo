import numpy as np

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
