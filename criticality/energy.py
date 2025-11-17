from tqdm.auto import tqdm
import numpy as np

from branching_model import BranchingModel


if __name__ == "__main__":
    x = np.linspace(0.0, 2.7, 10_000)

    final_energies = []
    for j in tqdm(x):
        energies = []
        model = BranchingModel(size = (20, 25), j = j)
        n_init = 10
        for i in range(25):
            model.run(n_init = n_init)
            image = model.grid.copy()

            sig = np.sum(image.astype(np.float32), axis = 0)
            sig = sig[~np.isnan(sig)]
            energy = sig[-1]
            energies.append(energy)

        final_energies.append(energies)

    np.save("data/energy.npy", np.array(final_energies))


