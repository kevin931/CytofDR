# CyTOF Dimension Reduction Framework

> A framework of dimension reduction and its evaluation for both CyTOF and general-purpose usages.

![Logo](/assets/logo.png)

| Branch | Release | CI/CD | Documentation | Code Coverage |
| --- | --- | --- | --- | --- |
| dev | ![Badge1](https://img.shields.io/badge/Version-0.3.1-success) |![Tests](https://github.com/kevin931/CytofDR/actions/workflows/ci.yml/badge.svg?branch=dev) | [![Documentation Status](https://readthedocs.org/projects/cytofdr/badge/?version=latest)](https://cytofdr.readthedocs.io/en/latest/?badge=latest) | [![codecov](https://codecov.io/gh/kevin931/CytofDR/branch/dev/graph/badge.svg?token=K9AJQLYU8N)](https://codecov.io/gh/kevin931/CytofDR) |


## About

CytofDR is a framework of dimension reduction (DR) and its evaluation for both Cytometry by Time-of-Flight (CyTOF) and general-purpose usages. It allows you to
conveniently run many different DRs at one place and then evaluate them to pick your embedding using our extensive evaluation framework! We aim to provide you with a reliable, extensible, and convenient interface for all your DR needs for both data analyses and future research!

### Key Resources

- For **detailed benchmarks and methodology explanations**, please check out [our paper](https://doi.org/10.1038/s41467-023-37478-w) in *Nature Communications*!
- For an online version of **interactive results**, please checkout [CytofDR Playground](https://dbai.biohpc.swmed.edu/cytof-dr-playground/).
- For **documentation**, please visit our free and detailed [documentation page](https://cytofdr.readthedocs.io/en/stable/index.html).

## Installation

You can install our CytofDR package, which is currentl on ``PyPI``:

```shell
pip install CytofDR
```

Python (>=3.7) is **required**. This pacackage is architecture agnostic: it should run where PyPI or conda is available. All dependencies should be automatically installed. For a list of optional dependencies, please visit our documentation page's detailed [Installation Guide](https://cytofdr.readthedocs.io/en/latest/installation.html).

Intallation should take less than a few minutes for most computers with reasonable network connections.

### Conda Installation

I personally recommend using ``conda`` to install everything since it's so easy to work with virtual environments. If you need help on how to get ``conda`` installed in the first place, take a look [here](https://docs.anaconda.com/anaconda/install/).

To install the package with ``conda``:

```shell
conda install -c kevin931 cytofdr -c conda-forge -c bioconda
```
The core dependencies should automatically install! 

### Dependencies

Our dependencies are broken down core dependencies and optional dependencies. Below is a list of core dependencies:

- scikit-learn
- numpy
- scipy
- umap-learn
- openTSNE
- phate
- annoy
- matplotlib
- seaborn

The most current compatible versions will work with ``CytofDR``, except for ``numpy``. New versions of ``numpy`` can cause issues with ``conda``. If you wish to use ``PyCytoData``, you need to install ``numpy`` version 1.20 or 1.21.

We also have some optional dependencies which are much trickier to install and manage. Refer to our [Installation Guide](https://cytofdr.readthedocs.io/en/latest/installation.html) for more details.

## PyCytoData Integration

``CytofDR`` is a member of the **PyCytoData Alliance Plus**, meaning that we're compatible with the ``PyCytoData`` package. The ``PyCytoData`` package is used mainly for loading datasets and managing every step of the CyTOF workflow. By creating and maintaining this ecosystem, we hope to create a robust workflow as a one-stop solution for CyTOF practioners using Python. To install ``PyCytoData``, you can simply use the following command:

```shell
pip install PyCytoData
```

To view how you can perform DR using ``PyCYtoData``, [this tutorial](https://pycytodata.readthedocs.io/en/latest/tutorial/dr.html) walks through every step.

## Quick Tutorial

``CytofDR`` makes it easy to run many DR methods while also evaluating them for your CyTOF samples. We have a greatly simplified pipeline for your needs. To get started, follow this example:

```python
>>> import numpy as np
>>> from CytofDR import dr
# Load Dataset
>>> expression = np.loadtxt(fname="PATH_To_file", dtype=float, skiprows=1, delimiter=",")
# Run DR and evaluate
>>> results = dr.run_dr_methods(expression, methods=["umap", "pca"])
Running PCA
Running UMAP
>>> results.evaluate(category = ["global", "local", "downstream"])
Evaluating global...
Evaluating local...
Evaluating downstream...
>>> results.rank_dr_methods()
{'PCA': 1.0, 'UMAP': 2.0}
# Save Results
>>> results.save_all_reductions(save_dir="PATH_to_DIR", delimiter=",")
>>> results.save_evaluations(path="PATH_to_FILE")
```
We strive to make our pipeline as simple as possible with natural langauge-like method names. Depending on your dataset size, the above example's runtime may vary. PCA is extremely fast, whereas can take upwards of 10 minutes if the dataset is much larger than 100,000 cells. For the `evaluate` command, the downstream command's silhouette score and clustering step can take some time, but for a small dataset, it can accomplish evaluation within a few minutes.

For large dataset, we recommend using efficient DR methods and providing your own clustering algorithm if possible.

### Example Dataset

We have included an example dataset generated by ``cytomulate`` in the `/example` folder. The data is an artificial data with 1000 cells to mimic real CyTOF data. To use the dataset, you can subsitute `PATH_to_file` with the path to the example dataset `exprs.txt`, which is in the expression matrix format.

### Examples using PyCytoData

You can use ``PyCytoData`` to load your dataset:

```python
>>> from CytofDR import dr
>>> from PyCytoData import FileIO
# Load Dataset
>>> dataset = FileIO.load_expression("PATH_To_file", col_names = True)
# Run DR and evaluate
>>> results = dr.run_dr_methods(dataset.expression_matrix, methods=["umap", "pca"])
Running PCA
Running UMAP
```
Or with a benchmark dataset:

```python
>>> from CytofDR import dr
>>> from PyCytoData import DataLoader
# Load Dataset
>>> dataset = DataLoader.load_dataset(dataset = "levine13")
# Run DR and evaluate
>>> results = dr.run_dr_methods(dataset.expression_matrix, methods=["umap", "pca"])
Running PCA
Running UMAP
```

All subsequent workflows remain the same.

### Documentation

Of course, there are many more customizations and ways you can use ``CytofDR``. So, for detailed tutorials and other guides, we suggest that you vists our [Official Documentation](https://cytofdr.readthedocs.io/en/latest/index.html).

There you will find ways to install our package and get started! Also, we offer tutorials on customizations, working with DR methods, and finally our detailed evaluation framework. We hope that you can find what you need over there!

## Latest Release: v0.3.1

This is a minor maintenance update of v0.3.x with updated references and documentation.

### Changes and New Features

- Updated referneces and citation information in all relavent documentaion pages
- Removed a warning on SAUCIE's installation documentation

### Improvements

- Update-to-date documentation and references

### Deprecations

- (Since v0.2.0) The `comparison_classes` parameter of the `EvaluationMetrics.embedding_concordance` method will no longer accept `str` input.

## Issues and Contributions

If you run into issues or have questions, feel free to open an issue [here](https://github.com/kevin931/CytofDR/issues). I'd love to help you out! We also welcome any contributions, but you may want to also look our [contribution guide](https://cytofdr.readthedocs.io/en/latest/change/contribution.html). Even if you just have an idea, that'll be great!

## References

Our preprint "Comparative Analysis of Dimension Reductions Methods for Cytometry by Time-of-Flight Data" is on bioRxiv and can be accessed [right here](https://doi.org/10.1038/s41467-023-37478-w). If you use our package in your research or deployment, a citation of our paper is highly appreciated:

```
@article{wang2023comparative,
  title={Comparative analysis of dimension reduction methods for cytometry by time-of-flight data},
  author={Wang, Kaiwen and Yang, Yuqiu and Wu, Fangjiang and Song, Bing and Wang, Xinlei and Wang, Tao},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={1--18},
  year={2023},
  publisher={Nature Publishing Group}
}
```

For a list of references of the methods, metrics, etc. used in the package, please visit our [References](https://cytofdr.readthedocs.io/en/latest/references.html) and bibliography of our paper.

