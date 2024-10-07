import sys, os
import pandas as pd

def read_bic_table(file_name: str, parse_metadata: bool = False) -> pd.DataFrame:
    """
    Reads a bicluster table from a tab-separated file and processes the data into a pandas DataFrame. 
    Optionally, it can also parses metadata from the first line of the file if available.

    Args:
        file_name (str): The path to the tab-separated bicluster file.
        parse_metadata (bool, optional): If True, parses metadata from the first line of the file. 
                                         The metadata line must start with a "#" and contain key-value 
                                         pairs separated by "; ". Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the bicluster data with columns such as:
                          - "genes": set of all genes in the bicluster.
                          - "genes_up": set of upregulated genes.
                          - "genes_down": set of downregulated genes.
                          - "samples": set of all samples in the bicluster.
                          - "gene_indexes": set of gene index numbers (as integers).
                          - "sample_indexes": set of sample index numbers (as integers).
                          
        If `parse_metadata` is True and metadata exists, returns a tuple:
        (pandas.DataFrame, dict): The bicluster DataFrame and a dictionary of metadata.

    Notes:
        - If the file does not exist returns None, if the file exists but is empty, an empty DataFrame is returned.
        - Missing values in "genes_up" and "genes_down" columns are filled with empty strings.
        - The "genes", "genes_up", "genes_down", "samples", "gene_indexes", and "sample_indexes" 
          fields are processed as sets, splitting the original space-separated string values.

    Example:
        biclusters, metadata = read_bic_table('biclusters.tsv', parse_metadata=True)
    """

    if not os.path.exists(file_name):
        return 
    biclusters = pd.read_csv(file_name, sep="\t", index_col=0, comment="#")
    if len(biclusters) == 0:
        return pd.DataFrame()
    else:
        biclusters.loc[:, ["genes_up", "genes_down"]] = biclusters.loc[
            :, ["genes_up", "genes_down"]
        ].fillna("")
        biclusters["genes"] = biclusters["genes"].apply(
            lambda x: set([g for g in x.split(" ") if not g == ""])
        )
        biclusters["genes_up"] = biclusters["genes_up"].apply(
            lambda x: set([g for g in x.split(" ") if not g == ""])
        )
        biclusters["genes_down"] = biclusters["genes_down"].apply(
            lambda x: set([g for g in x.split(" ") if not g == ""])
        )
        biclusters["samples"] = biclusters["samples"].apply(lambda x: set(x.split(" ")))
        biclusters["gene_indexes"] = biclusters["gene_indexes"].apply(
            lambda x: set(map(int, set(x.split(" "))))
        )
        biclusters["sample_indexes"] = biclusters["sample_indexes"].apply(
            lambda x: set(map(int, set(x.split(" "))))
        )

    if parse_metadata:
        f = open(file_name, "r")
        metadata = f.readline()
        f.close()
        if metadata.startswith("#"):
            metadata = metadata.replace("#", "").rstrip()
            metadata = metadata.split("; ")
            metadata = dict([x.split("=") for x in metadata])
            return biclusters, metadata
    return biclusters


def write_bic_table(
    bics: pd.DataFrame,
    results_file_name: str,
    to_str: bool = True,
    add_metadata: bool = False,
    seed: int = None,
    min_n_samples: int = None,
    bin_method: str = None,
    clust_method: str = None,
    pval: float = None,
    directions: list = [],
    similarity_cutoff: float = None,
    ds: float = None,
    dch: float = None,
    m: float = None,
    max_power: int = None,
    precluster: str = None,
    merge: bool = None,
) -> None:
    """
    Writes a bicluster table to a tab-separated file and optionally adds metadata to the file. 

    Args:
        bics (pd.DataFrame): A DataFrame containing biclusters.
        results_file_name (str): The path where the results should be written.


        to_str (bool, optional): If True, converts sets of values (genes, samples, etc.) 
                                 into space-separated strings before writing. Defaults to True.
        add_metadata (bool, optional): Defaults to False. If True, writes metadata to the file's first line. 
                                       For example, if `add_metadata` is True, metadata is written to the file
                                        in a format like "#seed=123; pval=0.05; min_n_samples=5; ...".
        seed (int, optional): Seed used for generating biclusters, added to metadata if provided.
        min_n_samples (int, optional): Minimum number of samples, included in metadata.
        bin_method (str, optional): Binning method used, included in metadata.
        clust_method (str, optional): Clustering method used (e.g., "Louvain" or "WGCNA"), 
                                      included in metadata.
        pval (float, optional): P-value threshold used for significance, included in metadata.
        directions (list, optional): Directions used in biclustering (e.g., "up", "down"), 
                                     included in metadata.
        similarity_cutoff (float, optional): Similarity cutoff used in the Louvain method, 
                                             included in metadata.
        ds (float, optional): Soft-thresholding power used in WGCNA, included in metadata.
        dch (float, optional): Dynamic cut height used in WGCNA, included in metadata.
        m (float, optional): Modularity value used in Louvain method, included in metadata.
        max_power (int, optional): Maximum power used in WGCNA, included in metadata.
        precluster (str, optional): Preclustering strategy used, included in metadata.
        merge (bool, optional): Whether merging of biclusters was performed, included in metadata.

    Returns:
        None: This function does not return any value, it writes the results to a file.
    """

    if add_metadata:
        metadata = (
            "#seed="
            + str(seed)
            + "; "
            + "pval="
            + str(pval)
            + "; "
            + "min_n_samples="
            + str(min_n_samples)
            + "; "
        )
        metadata = metadata + "b=" + bin_method + "; "
        metadata = metadata + "c=" + clust_method + "; "
        if len(directions):
            metadata = metadata + "directions=" + "-".join(directions) + "; "
        if clust_method == "Louvain":
            metadata = (
                metadata
                + "simiarity_cutoff="
                + str(similarity_cutoff)
                + "; modularity="
                + str(m)
            )
        elif "WGCNA" in clust_method:
            metadata = (
                metadata
                + "ds="
                + str(ds)
                + "; dch="
                + str(dch)
                + "; max_power="
                + str(max_power)
                + "; precluster="
                + str(precluster)
            )

        else:
            print("Unknown 'clust_method'", clust_method, file=sys.stderr)
        metadata = metadata + "; merge=" + str(merge)
        with open(results_file_name, "w") as f:
            f.write(metadata + "\n")
        write_mode = "a"
    else:
        write_mode = "w"

    if len(bics) == 0:
        print("No biclusters found", file=sys.stderr)
    else:
        if to_str:
            bics["genes"] = bics["genes"].apply(lambda x: " ".join(map(str, sorted(x))))
            bics["genes_up"] = bics["genes_up"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["genes_down"] = bics["genes_down"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["samples"] = bics["samples"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["gene_indexes"] = bics["gene_indexes"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            bics["sample_indexes"] = bics["sample_indexes"].apply(
                lambda x: " ".join(map(str, sorted(x)))
            )
            if "ids" in bics.columns:
                bics["ids"] = bics["ids"].apply(lambda x: " ".join(map(str, sorted(x))))
        bics.index.name = "id"
    bics.to_csv(results_file_name, sep="\t", mode=write_mode)