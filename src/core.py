import numpy as np

def make_grid(nx: int, x0: float = 0.0, x1: float = 1.0):
    """
    Create a 1D uniform grid on [x0, x1].

    Inputs:
        nx: number of spatial grid points (must be >= 2)
        x0, x1: domain endpoints

    Returns:
        x: numpy array of shape (nx,)
        dx: grid spacing
    """
    if nx < 2:
        raise ValueError("nx must be >= 2")

    if x1 <= x0:
        raise ValueError("Require x1 > x0")

    x = np.linspace(x0, x1, nx)
    dx = (x1 - x0) / (nx - 1)

    return x, dx


def initial_condition(x):
    """
    Initial condition u(x,0) = sin(pi * x)

    Inputs:
        x: numpy array of grid points

    Returns:
        u0: numpy array of same shape as x
    """
    u0 = np.sin(np.pi * x)

    # Enforce boundary conditions explicitly (robustness)
    u0[0] = 0.0
    u0[-1] = 0.0

    return u0


def exact_solution(x, t: float, alpha: float):
    """
    Exact solution:
        u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)

    Inputs:
        x: numpy array of grid points
        t: scalar time
        alpha: diffusion coefficient

    Returns:
        u_exact: numpy array of same shape as x
    """
    decay = np.exp(-alpha * np.pi * np.pi * t)
    u_exact = decay * np.sin(np.pi * x)

    # Enforce boundary conditions explicitly
    u_exact[0] = 0.0
    u_exact[-1] = 0.0

    return u_exact