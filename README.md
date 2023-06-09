# teneva_jax


## Description

This python package, named **teneva_jax** (**ten**sor **eva**luation with **jax**), provides a very compact implementation of basic operations in the low rank tensor-train (TT) format with jax framework for approximation, optimization and sampling with multidimensional arrays and multivariate functions. The program code is organized within a functional paradigm and it is very easy to learn and use. Each function has detailed documentation and various usage demos.

> Please, see also our github repository [teneva](https://github.com/AndreiChertkov/teneva), which contains the basic ("numpy") version of the code.


## Installation

> Current version "0.1.1".

The package can be installed via pip: `pip install teneva_jax` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [teneva_jax](https://github.com/AndreiChertkov/teneva_jax) and installed by `python setup.py install` command from the root folder of the project.

> Required python package ["jax[cpu]"](https://github.com/google/jax) (0.4.6+) will be automatically installed during the installation of the main software product. However, it is recommended that you manually install it first.


## Documentation, examples and tests

- See detailed [online documentation](https://teneva-jax.readthedocs.io) for a description of each function and various numerical examples for each function.
- See the jupyter notebooks in the `demo` folder with brief description and demonstration of the capabilities of each function from the `teneva_jax` package. Note that all examples from this folder are also presented in the online documentation.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)

> ✭__🚂  The stars that you give to **teneva_jax**, motivate us to develop faster and add new interesting features to the code 😃


## Citation

If you find our approach and/or code useful in your research, please consider citing:

```bibtex
@article{chertkov2023black,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Oseledets, Ivan},
    year      = {2023},
    title     = {Black box approximation in the tensor train format initialized by ANOVA decomposition},
    journal   = {arXiv preprint arXiv:2208.03380 (accepted into the SIAM Journal on Scientific Computing)},
    doi       = {10.48550/ARXIV.2208.03380},
    url       = {https://arxiv.org/abs/2208.03380}
}
```

```bibtex
@article{chertkov2022optimization,
    author    = {Chertkov, Andrei and Ryzhakov, Gleb and Novikov, Georgii and Oseledets, Ivan},
    year      = {2022},
    title     = {Optimization of functions given in the tensor train format},
    journal   = {arXiv preprint arXiv:2209.14808},
    doi       = {10.48550/ARXIV.2209.14808},
    url       = {https://arxiv.org/abs/2209.14808}
}
```
