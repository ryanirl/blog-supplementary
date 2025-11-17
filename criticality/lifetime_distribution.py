from tqdm.auto import tqdm
import numpy as np
import argparse

from branching_model import BranchingModel


def estimate_critical_exponent(x, y):
    """
    Estimate critical exponent β from power law: y ~ x^(-beta)
    
    Args:
        x: Array of x values (e.g., avalanche lengths)
        y: Array of y values (e.g., probabilities or counts)
    
    Returns:
        beta: critical exponent
    """
    # Remove zeros/negatives (can't take log)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    
    # Linear fit in log-log space: log(y) = -beta * log(x) + c
    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    beta = -coeffs[0]  # Slope is -beta
    
    return beta


def main(j: float = 1.0, n_runs: int = 250_000) -> None:
    model = BranchingModel(size = (40, 50), j = j)
    
    sigs = []
    for i in tqdm(range(n_runs)):
        model.run(n_init = 1) # Must be 1 for the lifetime distribution
        image = model.grid.copy()

        time = image.sum(axis = 0)
        time = time[~np.isnan(time)]

        size = len(time)
        if size == image.shape[1]:
            continue  # Skip truncated avalanches

        sigs.append(size)

    sigs = np.array(sigs)
    x, y = np.unique(sigs, return_counts = True)

    print(estimate_critical_exponent(x, y))

    np.save(f"data/power_x_{j:.1f}.npy", x)
    np.save(f"data/power_y_{j:.1f}.npy", y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Estimate critical exponent for the lifetime distribution of the branching model",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--j", type = float, default = 1.0, help = "Order parameter")
    parser.add_argument(
        "--n-runs", type = int, default = 250_000, help = "Number of runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.j, args.n_runs)


