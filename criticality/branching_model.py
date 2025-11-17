import numpy as np

from typing import Tuple


class BranchingModel:
    def __init__(self, size: Tuple[int, int], j: float = 1.0) -> None:
        self.size = size
        self.j = j  # Order parameter

        # Grid that we update during a simulation.
        self.grid = np.zeros(size, dtype = np.float32)

        # Adjacency probability weight matrix for connections. The index w[t, i, j] 
        # is the connection from i to j at time point t. This is normalized so
        # that the sum probability of activation out from degree i is p.
        self.init_w()

    def update_j(self, j: float) -> None:
        self.j = j
        self.init_w()

    def init_w(self) -> None:
        cells, time = self.size
        self.w = np.random.random(size = (time, cells, cells))
        self.w = (self.w / self.w.sum(axis = 2, keepdims = True)) * self.j

    def reset_grid(self) -> None:
        self.grid = np.zeros(self.size, dtype = np.float32)

    def run(self, n_init: int = 1) -> None:
        if (n_init < 1) or (n_init > self.size[0]):
            raise ValueError("`n_init` must be in the range [1, size[0]).")

        # Reset the grid and w for a new experimental run.
        self.reset_grid()
        #self.init_w()

        # Set the initial values
        init_inds = np.random.choice(self.size[0], size = (n_init,), replace = False)
        self.grid[init_inds, 0] = 1

        for t in range(self.size[1]-1):
            # At time `t` we need to update the cells in layer t+1 according to the 
            # activations in layer t.

            active_cells = np.where(self.grid[:, t] > 0)[0]
            if len(active_cells) < 1:
                self.grid[:, t+1:] = np.nan
                break

            integral = self.w[t, active_cells].sum(axis = 0)
            p = np.random.random(size = (self.size[0],))
            inds = p < integral
            self.grid[inds, t + 1] = 1


