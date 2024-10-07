# UnPaSt

UnPaSt is a novel method for identification of differentially expressed biclusters.

<img src="https://apps.cosy.bio/unpast/assets/DESMOND2_steps2.png"  height="350">

## Cite
UnPaSt preprint [https://arxiv.org/abs/2408.00200](https://arxiv.org/abs/2408.00200).

## Install
![Tests status](https://github.com/ozolotareva/unpast/actions/workflows/run_tests.yml/badge.svg)

### Docker environment
UnPaSt environment is available also as a Docker image.

```bash
docker pull freddsle/unpast
git clone https://github.com/ozolotareva/unpast.git
cd unpast
mkdir -p results

# running UnPaSt with default parameters and example data
command="python unpast/run_unpast.py --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500"
docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/data --entrypoint bash freddsle/unpast -c "cd /data && PYTHONPATH=/data $command"
```

### Requirements:
```
Python (version 3.8.16):
    fisher==0.1.9
    pandas==1.3.5
    python-louvain==0.15
    matplotlib==3.7.1
    seaborn==0.11.1
    numba==0.51.2
    numpy==1.22.3
    scikit-learn==1.2.2
    scikit-network==0.24.0
    scipy==1.7.1
    statsmodels==0.13.2
    kneed==0.8.1

R (version 4.3.1):
    WGCNA==1.70-3
    limma==3.42.2
```

### Installation tips

It is recommended to use "BiocManager" for the installation of WGCNA:
```R
install.packages("BiocManager")
library(BiocManager)
BiocManager::install("WGCNA")
```

## Input
UnPaSt requires a tab-separated file with features (e.g. genes) in rows, and samples in columns.
* Feature and sample names must be unique.
* At least 2 features and 5 samples are required.
* Data must be between-sample normalized.

### Recommendations: 
* It is recommended that UnPaSt be applied to datasets with 20+ samples.
* If the cohort is not large (<20 samples), reducing the minimal number of samples in a bicluster (`min_n_samples`) to 2 is recommended. 
* If the number of features is small, using Louvain method for feature clustering instead of WGCNA and/or disabling feature selection by setting the binarization p-value (`p-val`) to 1 might be helpful.

## Examples
* Simulated data example. Biclustering of a matrix with 10000 rows (features) and 200 columns (samples) with four implanted biclusters consisting of 500 features and 10-100 samples each. For more details, see figure 3 and Methods [here](https://arxiv.org/abs/2408.00200).
  
```bash
mkdir -p results;

# running UnPaSt with default parameters and example data
python -m unpast.run_unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500

# with different binarization and clustering methods
python -m unpast.run_unpast --exprs unpast/tests/scenario_B500.exprs.tsv.gz --basename results/scenario_B500 --binarization ward --clustering Louvain

# help
python run_unpast.py -h
```
* Real data example. Analysis of a subset of 200 samples randomly chosen from TCGA-BRCA dataset, including consensus biclustering and visualization:
  [jupyter-notebook](https://github.com/ozolotareva/unpast/blob/main/notebooks/UnPaSt_examples.ipynb).
  
## Outputs
`<basename>.[parameters].biclusters.tsv` - A `.tsv` file containing the identified biclusters with the following structure:

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

Along with the biclustering result, UnPaSt creates three files with intermediate results in the output folder `out_dir`:
  - `<basename>.[parameters].binarized.tsv` with binarized input data.
  - `<basename>.[parameters].binarization_stats.tsv` provides binarization statistics for each processed feature.
  - `<basename>.[parameters].background.tsv` stores background distributions of SNR values for each evaluated bicluster size.
These files can be used to restart UnPaSt with the same input and seed from the feature clustering step and skip time-consuming feature binarization. 

## Versions
UnPaSt version used in PathoPlex paper: [UnPaSt_PathoPlex.zip](https://github.com/ozolotareva/unpast_paper/blob/main/paper/UnPaSt_PathoPlex.zip)
