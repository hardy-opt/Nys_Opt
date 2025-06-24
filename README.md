````bash

nys-newton-optimizer/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
├── nys_newton/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── utils.py
│   └── nystrom_approximation.py
├── experiments/
│   ├── __init__.py
│   ├── mnist_experiment.py
│   ├── cifar10_experiment.py
│   ├── synthetic_experiment.py
│   └── benchmark_comparison.py
├── tests/
│   ├── __init__.py
│   ├── test_optimizer.py
│   ├── test_nystrom.py
│   └── test_utils.py
├── notebooks/
│   ├── demo_basic_usage.ipynb
│   ├── performance_analysis.ipynb
│   └── convergence_visualization.ipynb
├── results/
│   ├── figures/
│   ├── logs/
│   └── saved_models/
├── docs/
│   ├── api_reference.md
│   ├── theory_background.md
│   └── usage_examples.md
└── configs/
    ├── mnist_config.yaml
    ├── cifar10_config.yaml
    └── default_config.yaml

# Nys-Newton: Nyström-Approximated Curvature for Stochastic Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview
A PyTorch implementation of the Nys-Newton optimizer that efficiently approximates second-order curvature information using the Nyström method for improved convergence in deep learning optimization.

## Features
- Efficient second-order optimization using Nyström approximation
- Memory-efficient implementation suitable for large-scale problems
- Compatible with PyTorch's optimizer interface
- Comprehensive benchmarking against standard optimizers
- Theoretical convergence guarantees

## Installation
```bash
pip install -r requirements.txt
python setup.py install


import torch
from nys_newton import NysNewtonOptimizer

# Initialize model and optimizer
model = YourModel()
optimizer = NysNewtonOptimizer(model.parameters(), lr=0.01, rank=50)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = loss_fn(model(batch))
    loss.backward()
    optimizer.step()


@article{nys_newton_2021,
  title={Nyström-Approximated Curvature for Stochastic Optimization},
  author={Authors},
  journal={arXiv preprint arXiv:2110.08577},
  year={2021}
}