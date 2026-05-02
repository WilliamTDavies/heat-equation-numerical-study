import numpy as np
from scipy.linalg import solve

def forward_euler_step(u, alpha: float, dt: float, dx: float):
    """
    Inputs:
        u: current solution array, shape (nx,)
        alpha: diffusion coefficient
        dt: timestep
        dx: spatial spacing

    Returns:
        u_next: updated solution after one Forward Euler step
    """
    u_next = np.copy(u)
    r = alpha * dt / (dx * dx)

    # Discrete Laplacian
    u_next[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
    
    u_next[0] = 0.0
    u_next[-1] = 0.0
    return u_next

def backward_euler_matrix(nx: int, alpha: float, dt: float, dx: float):
    """
    Inputs:
        nx: number of grid points
        alpha, dt, dx: PDE parameters

    Returns:
        A: tridiagonal system matrix (dense form)
    """
    r = alpha * dt / dx**2
    n = nx - 2  # interior points only

    main = (1.0 + 2.0 * r) * np.ones(n)
    off = (-r) * np.ones(n - 1)

    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    return A

def backward_euler_step(u, A):
    """
    Inputs:
        u: current solution array, shape (nx,)
        A: backward Euler matrix from backward_euler_matrix

    Returns:
        u_next: updated solution after one Backward Euler step
    """
    u_inner = u[1:-1]

    # Solve linear system
    u_next_inner = solve(A, u_inner)

    u_next = np.zeros_like(u)
    u_next[1:-1] = u_next_inner

    u_next[0] = 0.0
    u_next[-1] = 0.0
    return u_next

def solve_forward_euler(u0, nt: int, alpha: float, dt: float, dx: float):
    """
    Full time evolution using Forward Euler.
    Inputs:
        u0: initial condition
        nt: number of time steps
        alpha, dt, dx: PDE parameters

    Returns:
        u_hist: array of shape (nt+1, nx)
    """
    u = u0.copy()
    history = [u.copy()]

    for _ in range(nt):
        u = forward_euler_step(u, alpha, dt, dx)
        history.append(u.copy())

    return np.array(history)

def solve_backward_euler(u0, nt: int, alpha: float, dt: float, dx: float):
    """
    Full time evolution using Backward Euler.
    Inputs:
        u0: initial condition
        nt: number of time steps
        alpha, dt, dx: PDE parameters

    Returns:
        u_hist: array of shape (nt+1, nx)
    """
    u = u0.copy()
    history = [u.copy()]

    A = backward_euler_matrix(len(u0), alpha, dt, dx)

    for _ in range(nt):
        u = backward_euler_step(u, A)
        history.append(u.copy())

    return np.array(history)
