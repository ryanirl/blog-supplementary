from tqdm.auto import tqdm
import numpy as np

from branching_model import BranchingModel


if __name__ == "__main__":
    n_init = 10
    x = np.linspace(0.0, 2.7, 10_000)

    rms = []
    final_sigs = []
    for j in tqdm(x):
        sigs = []
        model = BranchingModel(size = (20, 25), j = j)
        for i in range(25):
            model.run(n_init = n_init)
            image = model.grid.copy()
            sig = np.sum(image.astype(np.float32), axis = 0)
            sigs.append(sig)

        sigs = np.array(sigs)
        sigs = np.nanmean(sigs, axis = 0)
        sigs[np.isnan(sigs)] = 0
        final_sigs.append(sigs)
        rms.append(n_init - np.sqrt(np.mean((sigs - n_init) ** 2)))

    np.save("data/correlation_values.npy", np.array(final_sigs))


