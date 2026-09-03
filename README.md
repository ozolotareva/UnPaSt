# UnPaSt
[![Python Versions](https://img.shields.io/pypi/pyversions/unpast.svg)](https://pypi.org/project/unpast/)
[![Tests status](https://github.com/ozolotareva/unpast/actions/workflows/run-tests.yml/badge.svg)](https://github.com/ozolotareva/unpast/actions/workflows/run-tests.yml)
[![PyPI version](https://badge.fury.io/py/unpast.svg)](https://badge.fury.io/py/unpast)
[![Docker Build Status](https://github.com/ozolotareva/unpast/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/ozolotareva/unpast/actions/workflows/docker-publish.yml)
[![Docker Image Pulls](https://img.shields.io/docker/pulls/freddsle/unpast)](https://hub.docker.com/r/freddsle/unpast/tags)
[![License](https://img.shields.io/pypi/l/unpast.svg)](https://github.com/ozolotareva/unpast/blob/main/LICENSE)


UnPaSt is a novel method for identification of differentially expressed biclusters.

<img src="https://raw.githubusercontent.com/ozolotareva/unpast_paper/main/docs/UnPaSt_workflow_v7_method.png">

## Quick Start

### Using UnPaSt online

[Run UnPaSt at CoSy.Bio server](https://apps.cosy.bio/unpast/)

### Local installation

UnPaSt is available on [PyPI](https://pypi.org/project/unpast/) and can be installed using pip

```bash
pip install unpast

wget https://github.com/ozolotareva/unpast/raw/refs/heads/main/unpast/tests/test_input/synthetic_clear_biclusters.tsv
unpast --exprs synthetic_clear_biclusters.tsv
```

To use --clustering WGCNA method instead of default one, you would also need to install the necessary R packages (see Requirements below).

### Running in Docker

UnPaSt is also available as a Docker image, preinstalled R packages included. To pull the Docker image:

```bash
# load image and example data
docker pull freddsle/unpast
wget https://github.com/ozolotareva/unpast/raw/refs/heads/main/unpast/tests/test_input/synthetic_clear_biclusters.tsv

# run UnPaSt in a Docker environment with current directory and user
docker run --rm -it -u $(id -u):$(id -g) -v "$(pwd)":/data \
  freddsle/unpast \
    --exprs /data/synthetic_clear_biclusters.tsv \
    --out_dir /data/results/synthetic_clear_biclusters 
```

To use some previous docker version, replace `freddsle/unpast` with `freddsle/unpast:<version>` with a specific version tag, see available tags [here](https://hub.docker.com/r/freddsle/unpast/tags).

### Development setup

Developer mode allows you to run modified UnPaSt code. This is useful for local updates or contributing to the project.

<details>
  <summary> Docker development environment </summary>

To run UnPaSt in a Docker container with the latest code from the repository, you can use the following command:

```bash
# Clone the repository to get code
git clone https://github.com/ozolotareva/unpast.git
cd unpast

# Define the command to run UnPaSt 
# using unpast.run_unpast to surpass pre-insalled version from the Docker image
command="python -m unpast.run_unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500 --verbose"

# Run UnPaSt using Docker
docker run --rm -it -u $(id -u):$(id -g) -v "$(pwd)":/data --entrypoint bash freddsle/unpast -c "cd /data && $command"
```

</details>

### Requirements

UnPaSt requires Python 3.9-3.11 and certain Python and R packages.

<details>
  <summary>Python and R dependencies</summary>

#### Python Dependencies

The Python dependencies are installed automatically when installing via pip (see pyproject.toml). 


They include (with recommended versions):

```
fisher = ">=0.1.9,<=0.1.14"
pandas = "1.3.5"
python-louvain = "0.15"
matplotlib = "3.7.1"
seaborn = "0.11.1"
numba = ">=0.51.2,<=0.55.2"
numpy = "1.22.3"
scikit-learn = "1.2.2"
scikit-network = ">=0.24.0,<0.26.0"
scipy = ">=1.7.1,<=1.7.3"
statsmodels = "0.13.2"
kneed = "0.8.1"
```

#### R Dependencies

For the WGCNA clustering method, UnPaSt requires R and specific R packages.

UnPaSt utilizes R packages for certain analyses. Ensure that you have R installed with the following packages:

- `WGCNA` (version 1.70-3 or higher)
- `limma` (version 3.42.2 or higher)

### Installing R

Ensure that R (version 4.3.1 or higher) is installed on your system. You can download R from [CRAN](https://cran.r-project.org/).

It is recommended to use `BiocManager` for installing R packages:

```R
install.packages("BiocManager")
BiocManager::install("WGCNA")
BiocManager::install("limma")
```

</details>

## API Reference 

### Input
UnPaSt requires a tab-separated file with features (e.g. genes) in rows, and samples in columns.
* Feature and sample names must be unique.
* At least 2 features and 5 samples are required.
* Data must be between-sample normalized.

### Recommendations: 
* It is recommended that UnPaSt be applied to datasets with 20+ samples.
* If the cohort is not large (<20 samples), reducing the minimal number of samples in a bicluster (`min_n_samples`) to 2 is recommended. 
* If the number of features is small, using the Louvain method for feature clustering instead of WGCNA and/or disabling feature selection by setting the binarization p-value (`p-val`) to 1 might be helpful.

### Examples
* **Simulated data example**: Biclustering of a matrix with 10 000 rows (features) and 200 columns (samples) with four implanted biclusters consisting of 500 features and 10-100 samples each. For more details, see Figure 3 and Methods [here](https://arxiv.org/abs/2408.00200).
  
```bash
mkdir -p results;

# running UnPaSt with default parameters and example data
unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500

# with different binarization and clustering methods
unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500 --binarization ward --clustering Louvain

# help
unpast -h
```
* Real data example. Analysis of a subset of 200 samples randomly chosen from TCGA-BRCA dataset, including consensus biclustering and visualization:
  [jupyter-notebook](https://github.com/ozolotareva/unpast/blob/main/notebooks/UnPaSt_examples.ipynb).
  
### Outputs
The program creates a folder `runs/run_<timestamp>/` with the results of UnPaSt run, where `<timestamp>` is the date and time of the run in the format `YYYYMMDDTHHMMSS`.

The folder contains the files
```
run_YYYYMMDDTHHMMSS
├── args.tsv
├── biclusters.tsv 
└── unpast.log
```
The file `biclusters.tsv` contains the identified biclusters, with one bicluster per line. The format of this file is as follows:
- * the first line starts with `#`, storing the parameters of UnPaSt
- * the second line contains the column headers.
- * each subsequent line represents a bicluster with the following columns:
  - **SNR**: Signal-to-noise ratio of the bicluster, calculated as the average SNR of its features.
  - **n_genes**: Number of genes in the bicluster.
  - **n_samples**: Number of samples in the bicluster.
  - **genes**: Space-separated list of gene names.
  - **samples**: Space-separated list of sample names.
  - **direction**: Indicates whether the bicluster consists of up-regulated ("UP"), down-regulated ("DOWN"), or both types of genes ("BOTH").
  - **genes_up**, **genes_down**: Space-separated lists of up- and down-resulated genes respectively.
  - **gene_indexes**: 0-based index of the genes in the input matrix.
  - **sample_indexes**: 0-based index of the samples in the input matrix.

The files `args.tsv` and `unpast.log` contain the parameters used for the run and the log of the run respectively. 

Along with the biclustering result, if `save` mode is used, UnPaSt saves the intermediate results of feature binarization. The files are stored in the `binarization/` subfolder and include:
```
binarization
├── bin_args.tsv
├── bin_background.tsv
├── bin_res.tsv
└── bin_stats.tsv
```
with the files: 
- `bin_args.tsv` contains the subset of parameters used for binarization.
- `bin_background.tsv` contains background distributions of SNR values for each evaluated bicluster size.
- `bin_res.tsv` contains binarized input data.
- `bin_stats.tsv` provides binarization statistics for each processed feature.

The binarization files can be used to restart UnPaSt with the same input and seed from the feature clustering step and skip time-consuming feature binarization. 

## Cite and related code versions

If you use UnPaSt in your research, please cite the UnPaSt preprint  
```bibtex
@article{hartung2024unpast,
  title={UnPaSt: unsupervised patient stratification by differentially expressed biclusters in omics data},
  author={Hartung, Michael and Maier, Andreas and Delgado-Chaves, Fernando and Burankova, Yuliya and Isaeva, Olga I and Patroni, F{\'a}bio Malta de S{\'a} and He, Daniel and Shannon, Casey and Kaufmann, Katharina and Lohmann, Jens and others},
  journal={arXiv preprint arXiv:2408.00200},
  year={2024}
}
```

### Code versions for reproducibility

The following table provides links to the corresponding code versions for some papers using UnPaSt.

| Paper | Corresponding code version | 
|-------|------------------------------|
| UnPaSt preprint: https://arxiv.org/abs/2408.00200v2 | [UnPaSt paper repository](https://github.com/ozolotareva/unpast_paper) |
| PathoPlex paper: https://www.nature.com/articles/s41586-025-09225-2 | [UnPaSt_PathoPlex.zip](https://github.com/ozolotareva/unpast/blob/main/notebooks/UnPaSt_PathoPlex.zip) |
