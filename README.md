# Model for communicable disease outbreak dynamics

This repository contains the code for the simulator, ELFI model and
inference using Engine For Likelihood-Free Inference (ELFI, Lintusaari et al.,
2017). The inference is applied to the tuberculosis epidemic data from
San Francisco Bay area (Small et al., 1994).

The attached Jupyter notebook was run with Python 3.5 and ELFI v. 0.6.3.

## Files

- **code/simulator.py**
  Contains a general Python implementation of the simulator for the model
- **code/elfi_operations.py**
  Contains ELFI operations to run parameter inference with ELFI
- **inference.ipynb**
  Jupyter notebook defining the ELFI model and running the inference

## References

- Jarno Lintusaari, Henri Vuollekoski, Antti Kangasrääsiö, Kusti Skytén,
  Marko Järvenpää, Michael Gutmann, Aki Vehtari, Jukka Corander,
  Samuel Kaski 2017. ELFI: Engine for Likelihood Free Inference.
  ArXiv e-prints, https://arxiv.org/abs/1708.00707. Code
  https://github.com/elfi-dev/elfi.

- Small, P. M., P. C. Hopewell, S. P. Singh, A. Paz, J. Parsonnet,
  et al., 1994 The epidemiology of tuberculosis in san francisco –
  a population-based study using conventional and molecular
  methods. New England Journal of Medicine 330: 1703–1709.

