# CyTOF Dimension Reduction Framework

> A framework of dimension reduction and its evaluation for both CyTOF and general-purpose usages.

![Logo](/assets/logo.png)

## About

This is a work in progress for CyTOF DR analyses and evaluation.

## Installation

Note: All installation instructions are PENDING!

**Python (3.6, 3.7, 3.8) and pip (or conda)** are required. This is not yet a python package. So, ``git pull`` or manual downloading is required to get this working. But before you do that, make sure that you have all the dependencies installed.

### Required Dependencies

    - numpy
    - scikit-learn
    - openTSNE
    - umap-learn
    - scipy
    - phate

### Optional Dependencies
See below for notes on how to get these installed.

    - fit-SNE
    - BH t-SNE
    - SAUCIE
    - GrandPrix
    - ZIFA

### Conda Installation

I personally recommend using ``conda`` to install everything because virtual environment is very important for different parts of this project. If you need help on how to get ``conda`` installed in the first place, take a look [here](https://docs.anaconda.com/anaconda/install/).

To install all the required dependencies, run the following commands:


```shell
    conda create --name cytof
    conda activate cytof

    conda install python=3.8 numpy scikit-learn
    conda install -c conda-forge openTSNE umap-learn

```

## Usage
This project supports dimension reduction (DR), DR evaluation, and clustering. All these components are separate at this time. See examples for tutorials. 


## Optional Installations

A few methods cannot be easily installed as conda or pip packages. Therefore, more care is needed if usgae is required. Given that not all users are expected to have these packages installed, it is therefore safe to use ``-m all`` as it checks for import errors. 

### ZIFA

[Zero-inflated Factor Analysis](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-015-0805-z) is implemented by the authors [here](https://github.com/epierson9/ZIFA). Its installation is relatively easy given most of its major dependencies should already be satisfied. To install, 

```shell
    git clone https://github.com/epierson9/ZIFA
    cd ZIFA
    python setup.py install
```

### FIt-SNE

The original implementation of FIt-SNE can be found [here](https://github.com/KlugerLab/FIt-SNE), which is written in C++ with a python wrapper. For ease of use, I personally recommend openTSNE, which is a Python implementation of FIt-SNE, unless every bit performance is critical. From internal benchmark and benchmark published [here](https://opentsne.readthedocs.io/en/latest/benchmarks.html), openTSNE is only slighly slower thaFIt-SNE while vastly outperforming other traditional implementations.

To install FIt-SNE, ``git clone`` the FIt-SNE repository, and place it as a subfolder called "fitsne" inside this project directory. Compile the program according to instructions of FIt-SNE.  ``FFTW`` is a required dependency for FIt-SNE; therefore, it needs to be installed for the respective operating system. For Linux users, see [this issue](https://github.com/KlugerLab/FIt-SNE/issues/35) if you do not have root access.

### SAUCIE

SAUCIE does not have an official python package. Thus, we will need to pull the [GitHub repo](https://github.com/KrishnaswamyLab/SAUCIE). Place the pulled repo in a sub-directory called "saucie" inside this project directory.

Note: Only tensoflow 1.x is supported. This may cause issues with other dependencies in the future.

### BH t-SNE

The project already supports two implementations of BH t-SNE: sklearn and openTSNE. Both the former can be called with the flag ``-m sklearn_tsne_bh`` and the latter can be used with the combination of ``-m open_tsne --open_tsne_method bh``. Both, especially open_tsne, offer more flexibility and ease of use.

However, if you would like to use the original implementation from [here](https://github.com/lvdmaaten/bhtsne), pull the GitHub repositopry and place it as a subdirectory of this project and call it "bhtsne". Compuile the C++ file as described in the README.


## Future Directions
This is quite complex! More documentation will be added, inclusing docstrings, examples, etc. More methods will also be considered. Contributions are welcomed!