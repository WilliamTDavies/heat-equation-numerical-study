## Objectives

Consider the 1D heat equation: 

u_t = alpha * u_xx, for x in [0, 1]

with zero boundary conditions and initial condition:
u(x, 0) = sin(pi * x)

- Implement Forward Euler and Backward Euler using finite differences.
- Study stability as you vary the time step, and compare against the exact solution.
- Train a simple neural network that maps u^n -> u^(n+1) using data from a stable solver.
- Compare error over time and comment on stability and failure modes.

Deliverables:

- Code
- A few plots (e.g., error vs time, stability behavior)
- A short write-up (1–2 pages) describing observations



```
heat-equation-study/
│
├── src/
│   ├── core.py            # grid + initial condition + exact solution
│   ├── solvers.py         # Forward + Backward Euler
│   ├── experiments.py     # all simulation workflows (stability, error, dataset)
│   ├── neural.py          # NN + training + inference
│   ├── analysis.py        # metrics + comparisons
│   └── plotting.py        # all figures
│
├── results/
│   ├── figures/
│   └── data/
│
├── report/
│   ├── report.tex
│   └── report.pdf
│
├── run.py
└── requirements.txt
```