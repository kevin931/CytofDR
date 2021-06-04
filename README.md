# CyTOF Workflow

> A collection of common CyTOF analyses methods, especially dimension reduction, in Python.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [About](#about)
- [Installation](#installation)
  - [Required Dependencies](#required-dependencies)
  - [Optional Dependencies](#optional-dependencies)
  - [Conda Installation](#conda-installation)
- [Usage](#usage)
  - [Command-line Arguments](#command-line-arguments)
  - [File IO](#file-io)
  - [Dimension Reduction](#dimension-reduction)
    - [t-SNE Optimization](#t-sne-optimization)
- [Future Directions](#future-directions)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## About

This is a work in progress for CyTOF analyses and benchmarking.

## Installation
**Python (3.6, 3.7, 3.8) and pip (or conda)** are required. This is not yet a python package. So, ``git pull`` or manual downloading is required to get this working. But before you do that, make sure that you have all the dependencies installed.

### Required Dependencies
    - numpy
    - scikit-learn
    - openTSNE
    - umap-learn
    - PhenoGraph
    - scipy

### Optional Dependencies
See below for notes on how to get these installed.
    - fit-SNE
    - BH t-SNE
    - SAUCIE

### Conda Installation

I personally recommend using ``conda`` to install everything because virtual environment is very important for different parts of this project. If you need help on how to get ``conda`` installed in the first place, take a look [here]().

To install all the required dependencies, run the following commands:


```shell
    conda create --name cytof
    conda activate cytof

    conda install python=3.8 numpy scikit-learn
    conda install -c conda-forge openTSNE umap-learn
```

For installation issues and some caveats, take a look [here](https://poetic.readthedocs.io/en/latest/usage/Installation.html) for some common issues. Or, open an [issue](https://github.com/kevin931/poetic/issues) and I will be glad to help!


## Usage
This project supports dimension reduction (DR), DR evaluation, and clustering. All these components are separate at this time. See examples for tutorials. 

### Command-line Arguments
| Flag | Additional Inputs | Category | Function |
| --- | --- | --- | --- | 
| ``--cluster`` | None | Program Mode | Cluster the input file. |
| ``--evaluate`` | None | Program Mode | Evaluate embedding results. |
| ``--dr`` | None | Program Mode | Running dimension reduction algorithms. |
| ``-m`` or ``methods`` | Strings | Methods | Methods to run: applies to all modules. |
| ``-f`` or ``--files`` | Strings | File IO | Path to directory or original files. |
| ``--concat`` | None | File IO | Concatenate files in case mutiple files are read. |
| ``--delim`` | String | File IO | File delimiter (Default: \t). |
| ``-o`` or ``--out`` | String | File IO | New directory name for saving results. |
| ``--file_col_names`` | None | File IO | Whether the first line of the original file is column names. |
| ``--file_drop_col`` | Integers | File IO | The indicies of columns of the original file to be dropped. |
| ``--add_sample_index`` | None | File IO | Whether sample indicies are added as the first column. |
| ``--embedding`` | Strings | File IO | Load embedding from directory or file path. | 
| ``--embedding_col_names`` | None | File IO | Whether the first line of embedding is column names. |
| ``--embedding_drop_col`` | Integers | File IO | The indicies of columns of embedding to be dropped. |
| ``--label`` | Strings | File IO | Load label from directory or file path for evaluation. | 
| ``--label_col_names`` | None | File IO | Whether the first line of label is column names. |
| ``--label_drop_col`` | Integers | File IO | The indicies of columns of label to be dropped. |
| ``--downsample`` | Integer | Evaluation | The number of n to randomly down-sample in evaluation. | 
| ``--k-fold`` | Integer | Evaluation | The number of times to repeat down-sampling and compute average during evaluation. |
| ``--out_dims`` | Integer | DR Parameters | Output dimension. (Default: 2) |
| ``--perp`` | Integers | DR Parameters | Perplexity or a list of perplexities for t-SNE. (Default: 30) | 
| ``--early_exaggeration`` | Float | DR Parameters | Early exaggeration factor for t-SNE. (Default: 12.0) |
| ``--early_exaggeration_iter`` | Integer | DR Parameters | The iterations of early exaggerations to run. (Default: 250) |
| ``--open_tsne_method`` | String | DR Parameters | Approximation methods for openTSNE. (Default: "fft") |
| ``--tsne_learning_rate`` | Float | DR Parameters | Learning rate for t-SNE. (Default: 200.0) | 
| ``--init`` | str | DR Parameters | Initialization method. (Default: "pca" for t-SNE and "spectral" for UMAP) |
| ``--max_iter`` | Integer | DR Parameters | The maximum number of iterations to run. (Default: 1000) |


### File IO

Only expression matrices saved in the format of text files (inclusing csv and tsv) are supported. File IO and analyses/evaluation are integrated into one step. In other words, you don't have to load file or save files separately if you are using the command-line mode.

To simply read expression matrices from a file (with first line as column names and dropping columns at indicies 40 and 41) and run DR in a bare minimum fashion:

```shell

python main.py \
    -f <PATH_TO_FILE> \
    --delim , \
    -o ./results/open_tsne
    --dr
    -m open_tsne

```

Adding some file processing flags:

```shell
python main.py \
    -f <PATH_TO_FILE> \
    --delim , \
    --file_col_names \
    --file_drop_col 40 41 \
    -o ./results/open_tsne
    --dr
    -m open_tsne
```

To read original files and embeding for evaluation:

```shell

python main.py \
    -f <PATH TO file> \
    --delim , \
    --embedding <PATH_TO_EMBEDDING>
    -o ./results/open_tsne
    --evaluate
    -m all

```

Note: It is acceptable to pass in a directory for ``-f`` and ``--embedding``, in which case call files in the directory will be read. Also, flags such as ``--delim``, ``file_drop_col``, and ``--file_col_names`` are optional.

### Dimension Reduction

To perform dimension reduction, the following arguments are required: ``-f``, ``-o``, ``--dr``, and ``-m``. The acceptable strings for methods are: ``pca``, ``ica``, ``umap``, ``sklearn_tsne_original``, ``sklearn_tsne_bh``, ``open_tsne``, ``fit_sne``, ``bh_tsne``, and ``saucie``.  (Note: ``fit_sne``, ``bh_tsne``, and ``saucie`` need additional installations.)

To run t-SNE (I recommend ``open_tsne`` through the openTSNE package):

```shell

python main.py \
    -f <PATH TO file> \
    --delim , \
    -o ./results/open_tsne
    --dr
    -m open_tsne

```

All other methods use similar commands. All file IO commands apply.

#### t-SNE Optimization

As demonstrated by [Kobak & Berens (2019)](https://www.nature.com/articles/s41467-019-13056-x) and [Belkinas et al. (2019)](https://www.nature.com/articles/s41467-019-13055-y), t-SNE parameters are important for single-cell data. To perform some optimizations, an example will be like this (Note: I have not benchmarked all these optimizations. See the original paper and [the documentation](https://opentsne.readthedocs.io/en/latest/parameters.html) for recommendations.)

```shell
python main.py \
    -f <PATH TO file> \
    --delim , \
    -o ./results/open_tsne
    --dr
    -m open_tsne
    --init pca
    --perp 30 500
    --tsne_learning_rate auto
```

## Future Directions
This is quite complex! More documentation will be added, inclusing docstrings, examples, etc. More methods will also be considered. Contributions are welcomed!