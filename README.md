# Model for communicable disease outbreak dynamics

This repository contains the simulator code for the disease outbreak model by Lintusaari et al. 2018 [1]. 
The attached jupyter notebook contains the ABC inference using ELFI [2]. 
The observed data is the tuberculosis epidemic data from San Francisco Bay area [3].

The attached code is tested with Python 3.5 and ELFI v. 0.7.1.

## Files

- **code/simulator.py**
  Contains a general Python implementation of the simulator for the model
- **code/elfi_operations.py**
  Contains ELFI operations to run parameter inference with ELFI
- **inference.ipynb**
  Jupyter notebook defining the ELFI model and running the inference

## References

- [1] Jarno Lintusaari, Paul Blomstedt, Tuomas Sivula, Michael U. Gutmann, Samuel Kaski, Jukka Corander 2018. 
  Resolving outbreak dynamics using Approximate Bayesian Computation for stochastic birth-death models.
  bioRxiv 215533; doi: https://doi.org/10.1101/215533.

- [2] Jarno Lintusaari, Henri Vuollekoski, Antti Kangasrääsiö, Kusti Skytén,
  Marko Järvenpää, Michael Gutmann, Aki Vehtari, Jukka Corander,
  Samuel Kaski 2018. ELFI: Engine for Likelihood Free Inference.
  ArXiv e-prints, https://arxiv.org/abs/1708.00707. Code
  https://github.com/elfi-dev/elfi.

- [3] Small, P. M., P. C. Hopewell, S. P. Singh, A. Paz, J. Parsonnet,
  et al., 1994 The epidemiology of tuberculosis in san francisco –
  a population-based study using conventional and molecular
  methods. New England Journal of Medicine 330: 1703–1709.

