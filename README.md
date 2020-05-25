[logo]: https://raw.githubusercontent.com/makgyver/rectorch/master/docsrc/img/logo_150w.svg
![logo]

[travis-img]: https://travis-ci.org/makgyver/rectorch.svg?branch=master
[travis-url]: https://travis-ci.org/makgyver/rectorch
[language]: https://img.shields.io/github/languages/top/makgyver/rectorch
[issues]: https://img.shields.io/github/issues/makgyver/rectorch
[license]: https://img.shields.io/github/license/makgyver/rectorch
[version]: https://img.shields.io/badge/python-3.6|3.7|3.8-blue
[pypi-image]: https://img.shields.io/pypi/v/rectorch.svg
[pypi]: https://pypi.python.org/pypi/rectorch
[pytorch]: https://pytorch.org/

[![Build Status][travis-img]][travis-url]
[![PyPi][pypi-image]][pypi]
[![DOI](https://zenodo.org/badge/241092441.svg)](https://zenodo.org/badge/latestdoi/241092441)
[![Coverage Status](https://coveralls.io/repos/github/makgyver/rectorch/badge.svg?branch=master)](https://coveralls.io/github/makgyver/rectorch?branch=master)
[![docs](https://img.shields.io/badge/docs-github.io-blue)](https://makgyver.github.io/rectorch/)
![version] ![issues] ![license]

**rectorch** is a pytorch-based framework for top-N recommendation.
It includes several state-of-the-art top-N recommendation approaches implemented in [pytorch](https://pytorch.org/).

### Included methods

The latest PyPi release contains the following methods.

| Name      | Description                                                                | Ref.      |
|-----------|----------------------------------------------------------------------------|-----------|
| MultiDAE  | Denoising Autoencoder for Collaborative filtering with Multinomial prior   | [[1]](#1) |
| MultiVAE  | Variational Autoencoder for Collaborative filtering with Multinomial prior | [[1]](#1) |
| CMultiVAE | Conditioned Variational Autoencoder                                        | [[2]](#2) |
| CFGAN     | Collaborative Filtering with Generative Adversarial Networks               | [[3]](#3) |
| EASE      | Embarrassingly shallow autoencoder for sparse data                         | [[4]](#4) |

New methods are coming and are currently available in the *dev* branch (see [Dev branch](#dev-branch)).

| Name      | Description                                                     | Ref.      |
|-----------|-----------------------------------------------------------------|-----------|
| ADMM_Slim | ADMM SLIM: Sparse Recommendations for Many Users                | [[5]](#5) |
| SVAE      | Sequential Variational Autoencoders for Collaborative Filtering | [[6]](#6) |


# Getting started
## Installation

**rectorch** is available on PyPi and it can be installed using *pip*

```
pip3 install rectorch
```

## Requirements

If you install **rectorch** by cloning this repository make sure to install all the requirements.
```
pip3 install -r requirements.txt
```

## Architecture
**rectorch** is composed of 7 main modules summarized in the following.

| Name          | Scope                                                                                        |
|---------------|----------------------------------------------------------------------------------------------|
| configuration | Contains useful classes to manage the configuration files.                                   |
| data          | Manages the reading, writing and loading of the data sets                                    |
| evaluation    | Contains utility functions to evaluate recommendation engines.                               |
| metrics       | Contains the definition of the evaluation metrics.                                           |
| models        | Includes the training algorithm for the implemented recommender systems.                     |
| nets          | Contains definitions of the neural newtork architectures used by the implemented approaches. |
| samplers      | Contains definitions of sampler classes useful when training neural network-based models.    |

## Tutorials

*(To be released soon)* 

We will soon release a series of python notebooks with examples on how to train and evaluate
recommendation methods using **rectorch**.

## Documentation
The full documentation of the **rectorch** APIs is available at https://makgyver.github.io/rectorch/.

### Known issues
The documentation has rendering issues on 4K display. To "fix" the problem zoom in ([Ctrl][+], [Cmd][+]) the page.
Thanks for your patience, it will be fixed soon.

## Testing
The easiest way to test **rectorch** is using [pytest](https://docs.pytest.org/en/latest/).

```
git clone https://github.com/makgyver/rectorch.git
cd rectorch/tests
pytest
```

You can also check the coverage using [coverage](https://pypi.org/project/coverage/).
From the `tests` folder:
```
coverage run -m pytest  
coverage report -m
```

# Dev branch

**rectorch** is developed using a test-driven approach. The *master* branch (i.e., the pypi release) is the up-to-date
version of the framework where each module has been fully tested. However, new untested
or under development features are available in the *dev* branch. The *dev* version of **rectorch**
can be used by cloning the branch.

```
git clone -b dev https://github.com/makgyver/rectorch.git
cd rectorch
pip3 install -r requirements.txt
```

# Work in progress

The following features/changes will be soon released:
* Inclusion of `ADMM_Slim` and `SVAE` in the official release;
* Splitting of the `models` module in more sub-modules on the basis of the models' characteristics;
* Introduction of a "global" setting/configuration for the framework.

# Suggestions

This framework is constantly growing and the implemented methods are chosen on the basis of the need
of our research activity. We plan to include as many state-of-the-art methods as soon as we can, but
if you have any specific requests feel free to contact us by opening an issue.

# Citing this repo

If you are using **rectorch** in your work, please consider citing this repository.

```
@misc{rectorch,
    author = {Mirko Polato},
    title = {{rectorch: pytorch-based framework for top-N recommendation}},
    year = {2020},
    month = {may},
    doi = {10.5281/zenodo.3841898},
    version = {1.0},
    publisher = {Zenodo},
    url = {https://doi.org/10.5281/zenodo.153841898991}
}
```

## References
<a id="1">[1]</a>
Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
   Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
   World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
   Committee, Republic and Canton of Geneva, CHE, 689–698.
   DOI: https://doi.org/10.1145/3178876.3186150

<a id="2">[2]</a>
Tommaso Carraro, Mirko Polato and Fabio Aiolli. Conditioned Variational
   Autoencoder for top-N item recommendation, 2020. arXiv pre-print:
   https://arxiv.org/abs/2004.11141

<a id="3">[3]</a>
Dong-Kyu Chae, Jin-Soo Kang, Sang-Wook Kim, and Jung-Tae Lee. 2018.
   CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks.
   In Proceedings of the 27th ACM International Conference on Information and Knowledge
   Management (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 137–146.
   DOI: https://doi.org/10.1145/3269206.3271743

<a id="4">[4]</a>
Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data.
   In The World Wide Web Conference (WWW ’19). Association for Computing Machinery,
   New York, NY, USA, 3251–3257. DOI: https://doi.org/10.1145/3308558.3313710

<a id="5">[5]</a>
Harald Steck, Maria Dimakopoulou, Nickolai Riabov, and Tony Jebara. 2020.
   ADMM SLIM: Sparse Recommendations for Many Users. In Proceedings of the 13th International
   Conference on Web Search and Data Mining (WSDM ’20). Association for Computing Machinery,
   New York, NY, USA, 555–563. DOI: https://doi.org/10.1145/3336191.3371774

<a id="6">[6]</a>
Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi. 2019.
   Sequential Variational Autoencoders for Collaborative Filtering. In Proceedings of the Twelfth
   ACM International Conference on Web Search and Data Mining (WSDM ’19). Association for Computing
   Machinery, New York, NY, USA, 600–608. DOI: https://doi.org/10.1145/3289600.3291007
