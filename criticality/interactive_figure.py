import numpy as np

from typing import Tuple

from bokeh.layouts import column, row
from bokeh.models import Button, Slider, ColumnDataSource
from bokeh.plotting import figure, curdoc

WIDTH = 75
HEIGHT = 25

# Global variables for the simulation.
current_t = 0
callback_id = None


class BranchingModel:
    def __init__(self, size: Tuple[int, int], p: float = 1.0) -> None:
        self.size = size
        self.p = p
        self.grid = np.zeros(size, dtype = np.float32)
        self.prev_active = np.zeros(size, dtype = np.float32)
        self.init_w()
        
    def update_p(self, p: float) -> None:
        self.p = p
        self.init_w()
        
    def init_w(self) -> None:
        cells, time = self.size
        self.w = np.random.random(size = (time, cells, cells))
        self.w = (self.w / self.w.sum(axis = 2, keepdims = True)) * self.p
        
    def reset_grid(self) -> None:
        self.grid = np.zeros(self.size, dtype=np.float32)
        self.prev_active = np.zeros(self.size, dtype=np.float32)
        
    def step(self, t: int, n_init: int = 1) -> bool:
        if t == 0:
            init_inds = np.random.choice(self.size[0], size = (n_init,), replace = False)
            self.grid[init_inds, 0] = 1
            return True
            
        active_cells = np.where(self.grid[:, t-1] > 0)[0]
        if len(active_cells) < 1:
            self.grid[:, t:] = np.nan
            return False
        
        # Store previously active cells
        self.prev_active[:, t] = self.grid[:, t-1]

        integral = self.w[t, active_cells].sum(axis = 0)
        p = np.random.random(size = (self.size[0],))
        inds = p < integral

        self.grid[inds, t] = 1
        self.grid[:, t-1] = 0

        return True

    
def interactive_simulation() -> None:
    sim = BranchingModel(size = (HEIGHT, WIDTH), p = 1.0)

    # Create separate sources for each type of neuron
    y, x = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing = "ij")
    background_source = ColumnDataSource(data = dict(x = x.flatten(), y = y.flatten()))  # Background neurons (all possible positions)
    active_source = ColumnDataSource(data = dict(x = [], y = []))                  # Active neurons
    prev_source = ColumnDataSource(data = dict(x = [], y = []))                    # Previously active neurons

    p = figure(
        width = 1600, height = 600,
        title = "Branching Model",
        background_fill_color = "white",
        border_fill_color = "white",
        x_range = (-1, WIDTH+1),
        y_range = (-1, HEIGHT+1)
    )

    p.circle("x", "y", size = 15, color = "grey", alpha = 0.2, source = background_source)  # Add background (inactive) neurons
    p.circle("x", "y", size = 15, color = "grey", alpha = 0.6, source = prev_source)        # Add previously active neurons
    p.circle("x", "y", size = 15, color = "black", alpha = 1.0, source = active_source)     # Add currently active neurons

    p.grid.grid_line_color = "lightgray"
    p.grid.grid_line_alpha = 0.3
    p.axis.axis_line_color = "black"

    # Remove x-axis and y-axis grid lines
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p_slider = Slider(start = 0.0, end = 2.0, value = 1.0, step = 0.01, title = "p")
    n_slider = Slider(start = 1, end = 10, value = 1, step = 1, title = "Initial Sites")
    start_button = Button(label = "Start", button_type = "success")

    def update_sources():
        # Update active neurons
        active_y, active_x = np.where(sim.grid == 1)
        active_source.data = dict(x = active_x, y = active_y)
        
        # Update previously active neurons
        prev_y, prev_x = np.where(sim.prev_active == 1)
        prev_source.data = dict(x = prev_x-1, y = prev_y)

    def step_callback():
        global current_t, callback_id
        if current_t < WIDTH:
            if sim.step(current_t, int(n_slider.value)):
                update_sources()
                current_t += 1
            else:
                if callback_id is not None:
                    try:
                        curdoc().remove_periodic_callback(callback_id)
                    except ValueError:
                        pass
                    callback_id = None
        else:
            if callback_id is not None:
                try:
                    curdoc().remove_periodic_callback(callback_id)
                except ValueError:
                    pass
                callback_id = None

    def update():
        global current_t, callback_id
        if callback_id is not None:
            try:
                curdoc().remove_periodic_callback(callback_id)
            except ValueError:
                pass

        sim.reset_grid()
        sim.update_p(p_slider.value)

        current_t = 0
        active_source.data = dict(x = [], y = [])  # Clear active neurons
        prev_source.data = dict(x = [], y = [])  # Clear previously active neurons
        callback_id = curdoc().add_periodic_callback(step_callback, 200)

    start_button.on_click(update)
    layout = column(row(p_slider, n_slider), start_button, p)
    curdoc().add_root(layout)


interactive_simulation()

