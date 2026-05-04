import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "data"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load(name):
    return np.load(DATA_DIR / f"{name}.npz", allow_pickle=True)

# Error over time
def plot_error_vs_time():

    pde = load("pde_results")
    nn = load("nn_results")

    fe_err = pde["fe_err"]
    be_err = pde["be_err"]
    nn_err = nn["nn_err"]

    t = np.arange(1, len(fe_err))  # skip t=0

    fe_err = fe_err[1:]
    be_err = be_err[1:]
    nn_err = nn_err[1:]

    plt.figure()
    plt.plot(t, fe_err, label="Forward Euler")
    plt.plot(t, be_err, label="Backward Euler")
    plt.plot(t, nn_err, label="Neural Net")

    plt.yscale("log")
    plt.xlabel("Time step")
    plt.ylabel("L2 error (log scale)")
    plt.title("Error vs Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(FIG_DIR / "error_vs_time.png", dpi=300)

# Stability v dt
def plot_stability_vs_dt():

    data = load("stability_sweep")
    results = data["results"]

    dt_vals = []
    fe_errs = []
    be_errs = []

    for r in results:
        dt_vals.append(r["dt"])

        fe = r["fe_max_error"]
        be = r["be_max_error"]

        fe_errs.append(fe)
        be_errs.append(be)

    plt.figure()
    plt.plot(dt_vals, fe_errs, "o-", label="Forward Euler")
    plt.plot(dt_vals, be_errs, "o-", label="Backward Euler")

    plt.yscale("log")

    plt.xlabel("dt")
    plt.ylabel("Max L2 error (log scale)")
    plt.title("Stability vs Time Step")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(FIG_DIR / "stability_vs_dt.png", dpi=300)


# NN Dynamics
def plot_nn_dynamics():

    nn = load("nn_results")

    nn_be_diff = nn["nn_be_diff"][1:]
    nn_err = nn["nn_err"][1:]

    t = np.arange(len(nn_err))

    # Operator fidelity
    plt.figure()
    plt.plot(t, nn_be_diff, label="NN vs BE")

    plt.xlabel("Time step")
    plt.ylabel("Deviation from BE")
    plt.title("NN Learned Operator vs BE")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(FIG_DIR / "nn_operator_match.png", dpi=300)


# Main plotting function to call all diagnostics
def main():

    plot_error_vs_time()
    plot_stability_vs_dt()
    plot_nn_dynamics()


if __name__ == "__main__":
    main()