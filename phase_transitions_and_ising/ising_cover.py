import numpy as np
import numpy.random as random

from joblib import delayed, Parallel
from numba import jit


@jit(nopython=True)
def _calculate_energy(spins, i, j, J, size):
    """Calculate energy of spin at position (i,j)"""
    left  = spins[i, (j-1) % size[0]]
    right = spins[i, (j+1) % size[0]]

    up   = spins[(i-1) % size[1], j]
    down = spins[(i+1) % size[1], j]

    energy = -J * spins[i, j] * (left + right + up + down)

    return energy


@jit(nopython=True)
def _metropolis_step(spins, size, temperature, J):
    """Perform one Metropolis step"""
    n = size[0] * size[1]

    # Choose random spins and values
    for step in range(n):
        i = random.randint(0, size[0])
        j = random.randint(0, size[1])
        r = random.random()
        
        # Calculate energy change if we flip this spin
        curr_energy = _calculate_energy(spins, i, j, J, size)
        spins[i, j] *= -1
        flip_energy = _calculate_energy(spins, i, j, J, size)
        
        # Accept or reject the flip
        delta_E = flip_energy - curr_energy
        if (delta_E <= 0) or (r < np.exp(-delta_E / temperature)):
            continue
        else:
            spins[i, j] *= -1  # Reject the flip

    return spins


class IsingModel:
    def __init__(self, size = (50, 50), temperature = 2.0):
        self.size = size
        self.temperature = temperature

        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size = size)
        self.J = 1.0  # Coupling constant
        
    def calculate_energy(self, i, j):
        """Calculate energy of spin at position (i,j)"""
        return _calculate_energy(self.spins, i, j, self.J, self.size)
    
    def metropolis_step(self):
        """Perform one Metropolis step"""
        self.spins = _metropolis_step(self.spins, self.size, self.temperature, self.J)



def save_image(path):
    from PIL import Image
    image_uint8 = (image * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path, quality = 1000)

def _pcall(size, t):
    model = IsingModel(size = 50, temperature = t)
    for i in range(10_000):
        model.metropolis_step()
    m = model.spins.mean()
    return m


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #model = IsingModel(size = (800, 1000), temperature = 0.5)
    model = IsingModel(size = (800, 1000), temperature = 0.5)
    #model = IsingModel(size = (800, 1000), temperature = 2.2)
    model = IsingModel(size = (800, 1000), temperature = 5.0)
    for i in range(100):
        model.metropolis_step()

    cmap = plt.get_cmap("jet")

    image = (model.spins + 1) / 2
    image = image.clip(0.1, 0.9)
    image = cmap(image)[:, :, :3] # Convert to RBG

    #save_image("ising_example_05.png")
    #save_image("ising_example_22.png")
    save_image("ising_example_50.png")

    plt.imshow(image)
    plt.show()

