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
  - [Clustering](#clustering)
  - [DR Evaluation](#dr-evaluation)
- [Special Cases with R](#special-cases-with-r)
  - [Diffusion Map](#diffusion-map)
  - [FlowSOM CLustering](#flowsom-clustering)
- [t-SNE Optimization](#t-sne-optimization)
- [Optional Installations](#optional-installations)
  - [ZIFA](#zifa)
  - [FIt-SNE](#fit-sne)
  - [SAUCIE](#saucie)
  - [BH t-SNE](#bh-t-sne)
  - [MEDist](#medist)
- [Updates](#updates)
  - [June 26, 2021](#june-26-2021)
  - [June 14, 2021](#june-14-2021)
  - [June 10, 2021](#june-10-2021)
- [Future Directions](#future-directions)
- [References](#references)

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
    - MEDist

### Conda Installation

I personally recommend using ``conda`` to install everything because virtual environment is very important for different parts of this project. If you need help on how to get ``conda`` installed in the first place, take a look [here](https://docs.anaconda.com/anaconda/install/).

To install all the required dependencies, run the following commands:


```shell
    conda create --name cytof
    conda activate cytof

    conda install python=3.8 numpy scikit-learn
    conda install -c conda-forge openTSNE umap-learn

    pip install phenograph
```

## Usage
This project supports dimension reduction (DR), DR evaluation, and clustering. All these components are separate at this time. See examples for tutorials. 

### Command-line Arguments
| Flag | Additional Inputs | Category | Function |
| --- | --- | --- | --- | 
| ``--cluster`` | None | Program Mode | Cluster the input file. |
| ``--evaluate`` | None | Program Mode | Evaluate embedding results. |
| ``--dr`` | None | Program Mode | Running dimension reduction algorithms. |
| ``--build_annoy`` | None | Program Mode | Build ANNOY models for the input file(s). |
| ``--split_train`` | float | Program Mode | Split the input file with the provided percentage of training data and the remaining as test data. Column names will be saved. |
| ``-m`` or ``methods`` | Strings | Methods | Methods to run: applies to all modules. |
| ``-f`` or ``--files`` | Strings | File IO | Path to directory or original files. |
| ``--concat`` | None | File IO | Concatenate files in case mutiple files are read. |
| ``--delim`` | String | File IO | File delimiter (Default: \t). |
| ``-o`` or ``--out`` | String | File IO | New directory name for saving results. |
| ``--no_new_dir`` | None | File IO | Disable the default ``--o`` behavior to create a new directory |
| ``--file_col_names`` | None | File IO | Whether the first line of the original file is column names. |
| ``--file_drop_col`` | Integers | File IO | The indicies of columns of the original file to be dropped. |
| ``--add_sample_index`` | None | File IO | Whether sample indicies are added as the first column. |
| ``--embedding`` | Strings | File IO | Load embedding from directory or file path. | 
| ``--embedding_col_names`` | None | File IO | Whether the first line of embedding is column names. |
| ``--embedding_drop_col`` | Integers | File IO | The indicies of columns of embedding to be dropped. |
| ``--label`` | Strings | File IO | Load label from directory or file path for evaluation. | 
| ``--label_col_names`` | None | File IO | Whether the first line of label is column names. |
| ``--label_drop_col`` | Integers | File IO | The indicies of columns of label to be dropped. |
| ``--file_annoy`` | String | File IO | Path to the pre-trained ANNOY model of the input data. |
| ``--downsample`` | Integer | Evaluation | The number of n to randomly down-sample in evaluation. | 
| ``--k-fold`` | Integer | Evaluation | The number of times to repeat down-sampling and compute average during evaluation. |
| ``--save_downsample_index`` | String | Evaluation | Save the index of downsampling to the supplied directory. |
| ``--downsample_index_files`` | String | Evaluation | Path/Directory to previously saved downsample indicies. |
| ``--eval_k_neighbors`` | Int | Evaluation | The number of neighbors to consider for evaluation |
| ``--out_dims`` | Integer | DR Parameters | Output dimension. (Default: 2) |
| ``--perp`` | Integers | DR Parameters | Perplexity or a list of perplexities for t-SNE. (Default: 30) | 
| ``--early_exaggeration`` | Float | DR Parameters | Early exaggeration factor for t-SNE. (Default: 12.0) |
| ``--early_exaggeration_iter`` | Integer | DR Parameters | The iterations of early exaggerations to run. (Default: 250) |
| ``--open_tsne_method`` | String | DR Parameters | Approximation methods for openTSNE. (Default: "fft") |
| ``--tsne_learning_rate`` | Float | DR Parameters | Learning rate for t-SNE. (Default: 200.0) | 
| ``--init`` | str | DR Parameters | Initialization method. (Default: "pca" for t-SNE and "spectral" for UMAP) |
| ``--max_iter`` | Integer | DR Parameters | The maximum number of iterations to run. (Default: 1000) |
| ``--dist_metric`` | String | DR Parameters | The distance metric of tsne and UMAP. (Default: Euclidean) |
| ``--umap_min_dist`` | Float | DR Parameters | The minimum distance between points for UMAP. (Default: 0.1) |
| ``--umap_neighbors`` | Integer | DR Parameters | The number of neighbors for UMAP. (Default: 15) |
| ``--SAUCIE_lambda_c`` | Float | DR Parameters | Information dimension regularization for SAUCIE. (Default: 0) |
| ``--SAUCIE_lambda_d`` | Float | DR Parameters | Intracluster distance regularization for SAUCIE. (Default: 0) |
| ``--SAUCIE_learning_rate`` | Float | DR Parameters | Learning rate for SAUCIE. (Default: 0.001) |
| ``--SAUCIE_steps`` | int | DR Parameters | Maximum iterations of SAUCIE. (Default: 256) |
| ``--SAUCIE_batch_size`` | int | DR Parameters | Batch size for SAUCIE. (Default: 1000) |

### File IO
Only expression matrices saved in the format of text files (inclusing csv and tsv) are supported. File IO and analyses/evaluation are integrated into one step. In other words, you don't have to load file or save files separately if you are using the command-line mode.

To simply read expression matrices from a file (with first line as column names and dropping columns at indicies 40 and 41) and run DR in a bare minimum fashion:

```shell

python main.py \
    -f <PATH_TO_FILE> \
    --delim , \
    -o ./results/open_tsne \
    --dr \
    -m open_tsne

```

Adding some file processing flags:

```shell
python main.py \
    -f <PATH_TO_FILE> \
    --delim , \
    --file_col_names \
    --file_drop_col 40 41 \
    -o ./results/open_tsne \
    --dr \
    -m open_tsne
```

By default, the ``-o`` flag always creates a new directory and recursively so if necessary. However, it will not overwrite existing directories as it simply adds a number to it. To place results in the ``-o`` directory as-is without creating a new directory by adding the ``--no_new_dir`` flag:

```shell
python main.py \
    -f <PATH_TO_FILE> \
    --delim , \
    -o ./results/open_tsne \
    --no_new_dir \
    --dr \
    -m open_tsne
```

To read original files, embedding, and labels for DR performance evaluation:

```shell
python main.py \
    -f <PATH TO file> \
    --delim , \
    --embedding <PATH_TO_EMBEDDING> \
    --labels <PATH_TO_LABELS> \
    -o ./results/open_tsne \
    --evaluate \
    -m all
```

Note: It is acceptable to pass in a directory for ``-f`` and ``--embedding``, in which case call files in the directory will be read. Also, flags such as ``--delim``, ``file_drop_col``, and ``--file_col_names`` are optional.

### Build ANNOY Model for Input Files

ANNOY can be built at the time of evaluation, but for efficiency, it can be built and saved independently. To do so,

```shell

python main.py \
    -f <PATH TO FILE> \
    --build_annoy \
    -o <PATH TO OUTPUT DIRECTORY>
```

### Dimension Reduction
To perform dimension reduction, the following arguments are required: ``-f``, ``-o``, ``--dr``, and ``-m``. The acceptable strings for methods are: ``pca``, ``ica``, ``umap``, ``sklearn_tsne_original``, ``sklearn_tsne_bh``, ``open_tsne``, ``fit_sne``, ``bh_tsne``, ``saucie``, and ``zifa``.  (Note: ``fit_sne``, ``bh_tsne``, ``saucie`` and ``zifa`` need additional installations. See instructions below.)

To run t-SNE (I recommend ``open_tsne`` through the openTSNE package):

```shell

python main.py \
    -f <PATH TO file> \
    --delim , \
    -o ./results/open_tsne \
    --dr \
    -m open_tsne

```
All other methods use similar commands. All file IO commands apply.

### Clustering
Currently, only PhenoGraph clustering is supported. To cluster, use the following example as a guide:

```shell

python main.py \
    -f <PATH TO file> \
    --delim , \
    -o ./results/phenograph \
    --cluster \
    -m phenograph

```
The results will be save in the format of a tab-separated file in ``phenograph.txt`` of the output directory.

### DR Evaluation
The python program supports ten metrics for DR evaluation, and it is designed to be used in conjunction with DR methods in this program.

To run evaluation:
```shell
python main.py \
    -f <PATH_TO_ORIGINAL_FILE> \
    --embedding <PATH_TO_EMBEDDING, file or directory> \
    --label <PATH_TO_ORIGINAL_LABEL, optional> \
    --label_embedding <PATH_TO_EMBEDDING_LABEL, optional> \
    --evaluate \
    -m all \
    -o <SAVE_DIRECTORY>
```

To run evaluation with downsampling: 

```shell
python main.py \
    -f <PATH_TO_ORIGINAL_FILE> \
    --embedding <PATH_TO_EMBEDDING, file or directory> \
    --label <PATH_TO_ORIGINAL_LABEL, optional> \
    --label_embedding <PATH_TO_EMBEDDING_LABEL, optional> \
    --evaluate \
    -m all \
    -o <SAVE_DIRECTORY> \
    --save_downsample_index <PATH_TO_SAVE_SAMPLING_INDEX, optional> \
    --downsample 5000 \
    --k_fold 5
```

To downsample and save indicies without actually running evaluation: 

```shell
python main.py \
    -f <PATH_TO_ORIGINAL_FILE> \
    --evaluate \
    --save_downsample_index <PATH_TO_SAVE_SAMPLING_INDEX, optional> \
    --downsample 5000 \
    --k_fold 5
```

To downsample with save index (If a directory with multiple index files is supplied to ``--downsample_index_files``, it will treat all the files as ``--k_fold`` to repeat average the results.): 

```shell
python main.py \
    -f <PATH_TO_ORIGINAL_FILE> \
    --evaluate \
    --embedding <PATH_TO_EMBEDDING, file or directory> \
    --label <PATH_TO_ORIGINAL_LABEL, optional> \
    --label_embedding <PATH_TO_EMBEDDING_LABEL, optional> \
    --evaluate \
    -m all \
    --downsample_index_files <PATH_TO_SAVE_SAMPLING_INDEX>

```

## Special Cases with R
There are a few special cases that are not supported by the main python programs since they are implemented with R. 
### Diffusion Map

Currently, we only included the implementation of diffusion map in R with the ``destiny`` package. To run diffusion map, you will need to install ``destiny`` from ``bioconductor``. ``diffmap.R`` requires three mandatory commandline arguments: the path to original file, the distance metric, and the directory to save the file.

```shell
Rscript diffmap.R \
    <PATH_TO_FILE> \
    euclidean \
    <PATH_TO_SAVE_DIRECTORY>

```

Note: This R implementation does not have as many options and checks as the python scripts.
### FlowSOM CLustering
FlowSOM iscurrently supported with an R script. To run FlowSOM, 

```shell 
Rscript ./flowsom.R \
    <PATH_TO_INPUT_FILE> \
    <Number of clusters, int> \
    <EXACT_PATH_TO_OUTPUT_FILE> \
    <Whether input file has column names, R Boolean>

```

## t-SNE Optimization

As demonstrated by [Kobak & Berens (2019)](https://www.nature.com/articles/s41467-019-13056-x) and [Belkinas et al. (2019)](https://www.nature.com/articles/s41467-019-13055-y), t-SNE parameters are important for single-cell data. To perform some optimizations, an example will be like this (Note: I have not benchmarked all these optimizations. See the original paper and [the documentation](https://opentsne.readthedocs.io/en/latest/parameters.html) for recommendations.)

```shell
python main.py \
    -f <PATH TO file> \
    --delim , \
    -o ./results/open_tsne \
    --dr \
    -m open_tsne \
    --init pca \
    --perp 30 500 \
    --tsne_learning_rate auto
```

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

### MEDist

This is currently a private python module unpublished, but it is required for DR evaluation. Release is planned in the future. For access, contact author.

## Updates

### June 26, 2021
- Added support for Isomap and MDS
- Added SAUCIE parameter options on the command line.
- ANNOY is used in place of sklearn's KNN for speed.
- Changed distance metric in ``metric`` from full pairwise distance to Point Cluster Distance.
- Neighborhood trustworthiness is removed from ``metric.Metric.run_metrics()`` method
- Added the ``util`` module for handling utility functions
- More commandline options added to accomodate new changes.
### June 14, 2021
- Added FlowSome.R and documentation in README. 
- Optimized README layout
- Added support for saving downsampling index for ``--evaluate``
- Added support for downsampling without actually running metrics 
### June 10, 2021
- Added diffmap support through ``diffmap.R``. No python support.
- Added support for ZIFA. See [instructions](#ZIFA) for installation details.
- Added ``--umap_min_dist`` as a UMAP parameter.
- Added ``--umap_neighbors`` as a UMAP parameter.
- Added ``dist_metric`` as a general parameter for distance metric in dimension reduction.

## Future Directions
This is quite complex! More documentation will be added, inclusing docstrings, examples, etc. More methods will also be considered. Contributions are welcomed!

## References
- Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J., & Snyder-Cappione, J. E. (2019). “Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization and analysis of large datasets.” Nature Communications, 10(5415). https://doi.org/10.1038/s41467-019-13055-y.
- Kobak, D., & Linderman, G. C. (2021). “Initialization is critical for preserving global data structure in both t-SNE and UMAP.” Nature Biotechnology, 39, 156-157. https://doi.org/10.1038/s41587-020-00809-z.