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
        """Check if photon has escaped the slab (z < 0 or z > 1)."""
        if self.z < 0.0 or self.z > 1.0:
            self.alive = False
