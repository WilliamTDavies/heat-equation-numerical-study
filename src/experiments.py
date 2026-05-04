import numpy as np
import torch
from pathlib import Path

from src.core import make_grid, initial_condition, exact_solution_history
from src.solvers import solve_backward_euler, solve_forward_euler
from src.neural import TimeStepperNN, make_dataloader, train_model


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Error metric
def l2_error(u, ref):
    return np.sqrt(np.mean((u - ref) ** 2, axis=1))

# Solvers
def run_solvers(nx=101, nt=300, alpha=1.0, dt=1e-5):

    x, dx = make_grid(nx)
    u0 = initial_condition(x)

    exact = exact_solution_history(x, nt, alpha, dt)

    fe = solve_forward_euler(u0, nt, alpha, dt, dx)
    be = solve_backward_euler(u0, nt, alpha, dt, dx)

    return x, u0, exact, fe, be

# Training NN on BE data
def train_nn_on_be(be_hist, nx):

    loader = make_dataloader(be_hist, batch_size=32)

    model = TimeStepperNN(nx)

    model = train_model(model, loader, epochs=100, lr=1e-3)

    return model

# Rollout function for NN predictions
def rollout(model, u0, nt):

    u = u0.copy()
    hist = [u.copy()]

    model.eval()

    for _ in range(nt):

        x = torch.tensor(u, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            u = model(x).numpy()[0]

        u[0] = 0.0
        u[-1] = 0.0

        hist.append(u.copy())

    return np.array(hist)

# Stability sweep
def stability_sweep(nx, T, alpha, dt_list):

    results = []

    for dt in dt_list:

        nt = int(round(T / dt))

        x, dx = make_grid(nx)
        u0 = initial_condition(x)

        exact = exact_solution_history(x, nt, alpha, dt)

        fe = solve_forward_euler(u0, nt, alpha, dt, dx)
        be = solve_backward_euler(u0, nt, alpha, dt, dx)

        # full error over time
        fe_err = np.sqrt(np.mean((fe - exact) ** 2, axis=1))
        be_err = np.sqrt(np.mean((be - exact) ** 2, axis=1))

        # detect blow-up
        fe_blowup = (not np.all(np.isfinite(fe_err))) or (np.max(fe_err) > 1e6)

        # Record final errors and blow-up status
        fe_max_error = np.max(fe_err)
        be_max_error = np.max(be_err)

        results.append({
            "dt": dt,
            "fe_err": fe_err,
            "be_err": be_err,
            "fe_max_error": fe_max_error,
            "be_max_error": be_max_error,
            "fe_blowup": fe_blowup,
        })

    return results

# Growth analysis
def growth_rate(err):
    eps = 1e-12
    return np.log(err + eps)

# Save data utility
def save_data(name, **arrays):
    path = DATA_DIR / f"{name}.npz"
    np.savez(path, **arrays)
    print(f"Saved: {path}")

# Main experiments
def main():

    nx = 101
    nt = 300
    T = 0.01
    alpha = 1.0

    # Stability sweep
    dt_list = np.linspace(1e-5, 1e-4, 6)

    stab = stability_sweep(nx, T, alpha, dt_list)

    save_data("stability_sweep", results=np.array(stab, dtype=object))

    # Error over time (fixed dt)
    dt = 1e-5  # fixed reference timestep

    x, u0, exact, fe, be = run_solvers(nx, nt, alpha, dt)

    fe_err = l2_error(fe, exact)
    be_err = l2_error(be, exact)

    save_data(
        "pde_results",
        x=x,
        u0=u0,
        exact=exact,
        fe=fe,
        be=be,
        fe_err=fe_err,
        be_err=be_err,
    )

    # NN experiment (trained on BE)
    model = train_nn_on_be(be, nx)

    nn = rollout(model, u0, nt)

    nn_err = l2_error(nn, exact)

    # Compare NN dynamics directly to BE
    nn_be_diff = np.sqrt(np.mean((nn - be) ** 2, axis=1))

    save_data(
        "nn_results",
        nn=nn,
        nn_err=nn_err,
        nn_be_diff=nn_be_diff
)

    #Growth analysis (fixed dt only)
    fe_growth = growth_rate(fe_err)
    be_growth = growth_rate(be_err)
    nn_growth = growth_rate(nn_err)

    save_data(
        "growth_analysis",
        fe_growth=fe_growth,
        be_growth=be_growth,
        nn_growth=nn_growth,
    )

    #Print summary
    print(f"\nFinal errors (fixed dt={dt} run):")
    print("FE:", fe_err[-1])
    print("BE:", be_err[-1])
    print("NN:", nn_err[-1])

    print(f"\nGrowth trends (fixed dt={dt} run):")
    print("FE:", np.mean(np.diff(fe_growth)))
    print("BE:", np.mean(np.diff(be_growth)))
    print("NN:", np.mean(np.diff(nn_growth)))

if __name__ == "__main__":
    main()