# CyTOF Dimension Reduction Framework

> A framework of dimension reduction and its evaluation for both CyTOF and general-purpose usages.

![Logo](/assets/logo.png)

| Branch | Release | CI/CD | Documentation | Code Coverage |
| --- | --- | --- | --- | --- |
| dev | ![Badge1](https://img.shields.io/badge/Version-0.0.1-success) |![Tests](https://github.com/kevin931/CytofDR/actions/workflows/ci.yml/badge.svg?branch=dev) | [![Documentation Status](https://readthedocs.org/projects/cytofdr/badge/?version=latest)](https://cytofdr.readthedocs.io/en/latest/?badge=latest) | [![codecov](https://codecov.io/gh/kevin931/CytofDR/branch/dev/graph/badge.svg?token=K9AJQLYU8N)](https://codecov.io/gh/kevin931/CytofDR) |



## About

CytofDR is a framework of dimension reduction (DR) and its evaluation for both Cytometry by Time-of-Flight (CyTOF) and general-purpose usages. It allows you to
conveniently run many different DRs at one place and then evaluate them to pick your embedding using our extensive evaluation framework! We aim to provide you with a reliable, extensible, and convenient interface for all your DR needs for both data analyses and future research!

## Installation

You can install our CytofDR package, which is currentl on ``PyPI``:

```shell
pip install CytofDR
```

Python (>=3.7) is **required**. All dependencies should be automatically installed. For a list of optional dependencies, please visit our documentation page's detailed [Installation Guide](https://cytofdr.readthedocs.io/en/latest/installation.html).


### Conda Installation

I personally recommend using ``conda`` to install everything since it's so easy to work with virtual environments. If you need help on how to get ``conda`` installed in the first place, take a look [here](https://docs.anaconda.com/anaconda/install/).

To install the package with ``conda``:

```shell
conda install -c kevin931 cytofdr -c conda-forge -c bioconda
```
The core dependencies should automatically install! 

## Quick Tutorial

``CytofDR`` makes it easy to run many DR methods while also evaluating them for your CyTOF samples. We have a greatly simplified pipeline for your needs. To get started, follow this example:

```python
>>> import numpy as np
>>> from CytofDR import dr
# Load Dataset
>>> expression = np.loadtxt(fname="PATH_To_file", dtype=float, skiprows=1, delimiter=",")
# Run DR and evaluate
>>> results = dr.run_dr_methods(expression, methods=["umap", "open_tsne", "pca"])
>>> results.evaluate(category = ["global", "local", "downstream"])
>>> results.rank_dr_methods()
# Save Results
>>> results.save_all_reductions(save_dir="PATH_to_DIR", delimiter=",")
>>> results.save_evaluations(path="PATH_to_FILE")
```
We strive to make our pipeline as simple as possible with natural langauge-like method names.

### Documentation

Of course, there are many more customizations and ways you can use ``CytofDR``. So, for detailed tutorials and other guides, we suggest that you vists our [Official Documentation](https://cytofdr.readthedocs.io/en/latest/index.html).

There you will find ways to install our package and get started! Also, we offer tutorials on customizations, working with DR methods, and finally our detailed evaluation framework. We hope that you can find what you need over there!

## Latest Release

Our lastest release is ``v0.0.1`` with the following the following release notes:

- This is the first offical pre-release of ``CytofDR``.
- Most of the pipeline is complete, including DR, evaluation, ranking, and plotting.
- Extensive documentation and tutorial complete.
- This release aims to aid the completion of our development and tool chain.
- We are on  ``conda`` and ``PyPI``!

For more release and development information, look around on our GitHub or look through our [changelog](https://cytofdr.readthedocs.io/en/latest/change/index.html). 

## Issues and Contributions

If you run into issues or have questions, feel free to open an issue [here](https://github.com/kevin931/CytofDR/issues). I'd love to help you out! We also welcome any contributions, but you may want to also look our [contribution guide](https://cytofdr.readthedocs.io/en/latest/change/contribution.html). Even if you just have an idea, that'll be great!

## References

Our preprint "Comparative Analysis of Dimension Reductions Methods for Cytometry by Time-of-Flight Data" is on bioRxiv and can be accessed [right here](https://doi.org/10.1101/2022.04.26.489549). If you use our package in your research or deployment, a citation of our paper is highly appreciated:

```
@article {Wang2022.04.26.489549,
	author = {Wang, Kaiwen and Yang, Yuqiu and Wu, Fangjiang and Song, Bing and Wang, Xinlei and Wang, Tao},
	title = {Comparative Analysis of Dimension Reduction Methods for Cytometry by Time-of-Flight Data},
	elocation-id = {2022.04.26.489549},
	year = {2022},
	doi = {10.1101/2022.04.26.489549},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {While experimental and informatic techniques around single cell sequencing (scRNA-seq) are much more advanced, research around mass cytometry (CyTOF) data analysis has severely lagged behind. However, CyTOF profiles the proteomics makeup of single cells and is much more relevant for investigation of disease phenotypes than scRNA-seq. CyTOF data are also dramatically different from scRNA-seq data in many aspects. This calls for the evaluation and development of statistical and computational methods specific for analyses of CyTOF data. Dimension reduction (DR) is one of the most critical first steps of single cell data analysis. Here, we benchmark 20 DR methods on 110 real and 425 synthetic CyTOF datasets, including 10 Imaging CyTOF datasets, for accuracy, scalability, stability, and usability. In particular, we checked the concordance of DR for CyTOF data against scRNA-seq data that were generated from the same samples. Surprisingly, we found that a less well-known method called SAUCIE is the overall best performer, followed by MDS, UMAP and scvis. Nevertheless, there is a high level of complementarity between these tools, so the choice of method should depend on the underlying data structure and the analytical needs (e.g. global vs local preservation). Based on these results, we develop a set of freely available web resources to help users select the best DR method for their dataset, and to aid in the development of improved DR algorithms tailored to the increasingly popular CyTOF technique.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/06/02/2022.04.26.489549},
	eprint = {https://www.biorxiv.org/content/early/2022/06/02/2022.04.26.489549.full.pdf},
	journal = {bioRxiv}
}
```

For a list of references of the methods, metrics, etc. used in the package, please visit our [References](https://cytofdr.readthedocs.io/en/latest/references.html) and bibliography of our paper.

