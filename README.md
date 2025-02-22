[![image](https://img.shields.io/badge/GitHub-quantifai-brightgreen.svg?style=flat)](https://github.com/astro-informatics/quantifai) [![image](https://img.shields.io/badge/License-GPL-blue.svg?style=flat)](https://github.com/astro-informatics/quantifai/blob/main/LICENSE.txt) [![image](https://img.shields.io/badge/arXiv-2312.00125-red.svg?style=flat)]( https://arxiv.org/abs/2312.00125)  


# QuantifAI

`quantifai` is a PyTorch-based open-source radio interferometric imaging reconstruction package with scalable Bayesian uncertainty quantification relying on data-driven (learned) priors. This package was used to produce the results of [Liaudat et al. 2023](https://arxiv.org/abs/2312.00125). The `quantifai` model relies on the data-driven convex regulariser from [Goujon et al. 2022](https://arxiv.org/abs/2211.12461).

In this code, we bypass the need to perform Markov chain Monte Carlo (MCMC) sampling for Bayesian uncertainty quantification, and we rely on convex accelerated optimisation algorithms. The `quantifai` package also includes MCMC algorithms for posterior sampling as they were used to validate our approach.

> [!NOTE]  
> This Python package is built on top of PyTorch, so a GPU can considerably accelerate all computations. 


## Installation

The `quantifai` package relies on the convex ridge regulariser CRR from [Goujon et al. 2022](https://arxiv.org/abs/2211.12461). The version used to generate the results from the Liaudat et al. paper is the release `v0.1` from the fork [github.com/tobias-liaudat/convex_ridge_regularizers](https://github.com/tobias-liaudat/convex_ridge_regularizers). The PyTorch wavelet support relies on the release `v0.1` from the fork [github.com/tobias-liaudat/PyTorch-Wavelet-Toolbox](https://github.com/tobias-liaudat/PyTorch-Wavelet-Toolbox).

We have not yet pushed the Python package to PyPi; therefore, the easiest way to install `quantifai` is to start by cloning the repo

```bash
git clone https://github.com/astro-informatics/QuantifAI
cd QuantifAI
```

Continue by creating a conda environment with all the requirements already specified in `environment.yml` as follows

```bash
conda env create -f environment.yml
conda activate quantifai_env
```

Finally, the `quantifai` package can be installed by running

```bash
pip install -e .
```


> [!NOTE]  
> If the user does not want to create a conda environment, they can install the dependencies in the `environment.yml` file. The specific version of the convex ridge regulariser, complex PyTorch support and the PyTorch wavelets used can be manually installed by running (in the following order)
> ```bash
> pip install git+https://github.com/tobias-liaudat/convex_ridge_regularizers@v0.1
> pip install git+https://github.com/tobias-liaudat/complexPyTorch@v0.1
> pip install git+https://github.com/tobias-liaudat/PyTorch-Wavelet-Toolbox@v0.1
> ```

The paper's numerical results were obtained using PyTorch version `1.13.1`.


## Examples & usage

The easiest way to get into using `quantifai` is to check the different notebooks in the `example/` directory, which includes:

- Compute the MAP estimation with the `QuantifAI` model ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_QuantifAI_MAP_estimation.ipynb)).
- Compute the MAP estimation with the wavelet-based model ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_wavelets_MAP_estimation.ipynb)).

- Compute the MAP-based LCIs with the `QuantifAI` model ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_QuantifAI_LCIs.ipynb)).
- Compute the MAP-based LCIs with the wavelet-based model ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_wavelets_MAP_estimation.ipynb)).
- Compute the MAP-based fast pixel uncertainty quantification method with `QuantifAI` ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_QuantifAI_fast_pixel_UQ.ipynb)).
- Compute a hypothesis test on an inpainted surrogate image with `QuantifAI` ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_QuantifAI_hypothesis_test)).
- Sample from the posterior distribution of the `QuantifAI` model using the SK-ROCK algorithm ([Pereyra et al. 2020](https://doi.org/10.1137/19M1283719)) and compare the results with sample-based LCIs ([Notebook](https://github.com/astro-informatics/QuantifAI/blob/main/examples/RI_imaging_QuantifAI_sampling)).


## Reproducibility

All the scripts and notebooks used to generate the plots of [Liaudat et al. 2023](https://arxiv.org/abs/2312.00125) can be found in the `paper/Liaudat2023/` directory and the data in the `data/` directory.

The most computationally intensive results of the paper can be obtained by running the two scripts in `paper/Liaudat2023/scripts/`, where `UQ_SKROCK_CRR.py` corresponds to the `QuantifAI` model and `UQ_SKROCK_wavelets.py` to the wavelet-based model. The rest of the results and plots can be generated by running the different notebooks in `paper/Liaudat2023/notebooks/`.


## Attribution

Should this code be used in any way, we kindly request that the following article be referenced. A BibTeX entry for this reference may look like:

```
@article{liaudat2023:quantifai, 
    author = {Tobías~I.~Liaudat and Matthijs~Mars and Matthew~A.~Price and Marcelo~Pereyra and Marta~M.~Betcke and Jason~D.~McEwen},
    title = {Scalable Bayesian uncertainty quantification with data-driven priors for radio interferometric imaging},
    journal = "RAS Techniques and Instruments (RASTI), submitted",
    eprint = "arXiv:2312.00125",
    year = "2023",
}
```

## License

`quantifai` is released under the GPL-3 license (see [LICENSE.txt](https://github.com/astro-informatics/QuantifAI/blob/main/LICENSE.txt)).

