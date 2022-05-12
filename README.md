[![DOI](https://zenodo.org/badge/330059679.svg)](https://zenodo.org/badge/latestdoi/330059679)

# ECONJAX

This repository contains JAX implementations of Dynamic Programming Techniques commonly used in economics:

* Brute-force Value Function Iteration.
* Standard (F.O.C. based) Value Function Iteration.

The goal of this repository is to provide optimized versions of the algorithms that can run on CPU, GPU or TPU efficiently, using the simplity allowed by jax to leverage automatic vectorization (jax.vmap), automatic differntiation (jax.grad), parallelization (jax.pmap) and compiler-based optimizations (jax.jit). The intention is for them to be simple and readable implementations to build research on top of. 
If you use econjax in your work, please cite this repository in publications:
```
@misc{jaxrl,
  author = {Covarrubias, Matias},
  doi = {},
  month = {},
  title = {{ECONJAX: Implementations of Dynamic Programming Algorithms using JAX}},
  url = {https://github.com/MatiasCovarrubias/econjax/edit/main/README.md},
  year = {2021}
}
```

# Changelog


## May 12, 2022
- Created the private repository and an implementation of the Brute-Force VFI algorithm.

# Installation

Explain the intended use.

# [Examples](examples/)

# Google Cloud support

# Troubleshooting

Explain Troubleshooting

# Tensorboard

Launch tensorboard to see training and evaluation logs

```bash
tensorboard --logdir=./tmp/
```

# Results

Explain results

## Test

Code for test scripts.

# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue. If you are not familiar with pull requests, please read [this documentation](https://opensource.com/article/19/7/create-pull-request-github).

# Acknowledgements 

I want to thank Agustin Covarrubias for his collaboration in setting things in Google Cloud. 
