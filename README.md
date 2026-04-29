## Objectives

Consider the 1D heat equation: 

u_t = alpha * u_xx, for x in [0, 1]

with zero boundary conditions and initial condition:
u(x, 0) = sin(pi * x)

    Implement Forward Euler and Backward Euler using finite differences.
    Study stability as you vary the time step, and compare against the exact solution.
    Train a simple neural network that maps u^n -> u^(n+1) using data from a stable solver.
    Compare error over time and comment on stability and failure modes.

Deliverables:

    Code
    A few plots (e.g., error vs time, stability behavior)
    A short write-up (1–2 pages) describing observations