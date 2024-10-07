# Usage: from unpast.utils.unpast_DE import run_de_for_unpast
# run_de_for_unpast(unpast_output_path, expression_matrix_path, counts = False, [keep_all=False,adj_p_value_cut_off = 0.05, logFC_cut_off = 1, r_script_path = None, r_executable_path = None])

import pandas as pd
import os
import logging
import subprocess
import numpy as np

# Add all logging levels
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

####################################################################################################
# Constants
####################################################################################################
DELIMITER = "\t"

####################################################################################################
# Functions
####################################################################################################


def extract_samples_to_file(unpast_df: pd.DataFrame, samples_to_compare: str) -> None:
    """
    Extract samples from the unpast dataframe.
    """
    samples = unpast_df[["samples", "n_samples"]]
    samples.to_csv(samples_to_compare, sep=DELIMITER, index_label="")


def run_add_genes_script(
    samples_to_compare: str,
    expression_matrix_path: str,
    counts: bool = False,
    adj_p_value_cut_off: float = 0.05,
    logFC_cut_off: float = 1,
    num_genes_cut_off = float('inf'),
    r_script_path: str = None,
    r_executable_path: str = None,
) -> str:
    if not r_script_path:
        # Assume add_genes.R is in the same folder
        r_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "add_genes.R")

    if r_executable_path is None:
        r_executable_path = "Rscript"
    else:
        r_executable_path = os.path.join(r_executable_path, "Rscript")

    logging.debug(
        [
            r_executable_path,
            r_script_path,
            samples_to_compare,
            expression_matrix_path,
            str(counts).upper(),
            str(adj_p_value_cut_off),
            str(logFC_cut_off),
            str(num_genes_cut_off)
        ]
    )

    process = subprocess.Popen(
        [
            r_executable_path,
            r_script_path,
            samples_to_compare,
            expression_matrix_path,
            str(counts).upper(),
            str(adj_p_value_cut_off),
            str(logFC_cut_off),
            str(num_genes_cut_off)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()

    # Decode the output and error messages
    samples_with_genes_path = stdout.decode("utf-8")

    if stdout:
        logging.debug("DE analysis temporary output was generated: %s", samples_with_genes_path)
    if stderr:
        error_message = stderr.decode("utf-8")
        logging.error("Error Output: %s", error_message)
        raise Exception(f"Error in subprocess: {error_message}")

    return samples_with_genes_path


def filter_de_genes(new_unpast_df: pd.DataFrame, columns_to_process: list) -> pd.DataFrame:
    """
    Filter de_genes_df to keep only the genes that are in the unpast_df.
    """
    # Loop through each column
    for col in columns_to_process:
        col_DE = col + "_DE"

        # Check for NaN in original column and replace the corresponding _DE column value with NaN
        new_unpast_df[col_DE] = np.where(new_unpast_df[col].isna(), np.nan, new_unpast_df[col_DE])

        # Split the string into a list for both columns
        new_unpast_df[col] = new_unpast_df[col].apply(lambda x: str(x).split() if pd.notna(x) else [])
        new_unpast_df[col_DE] = new_unpast_df[col_DE].apply(lambda x: str(x).split() if pd.notna(x) else [])

        # Keep in the _DE column only values that are in the corresponding original column
        new_unpast_df[col_DE] = new_unpast_df.apply(lambda x: [i for i in x[col_DE] if i in x[col]], axis=1)

        # Join the list back into a string
        new_unpast_df[col] = new_unpast_df[col].apply(lambda x: " ".join(x) if x else np.nan)
        new_unpast_df[col_DE] = new_unpast_df[col_DE].apply(lambda x: " ".join(x) if x else np.nan)
    return new_unpast_df


def add_columns_to_unpast_df(unpast_df: pd.DataFrame, 
                             de_genes_df: pd.DataFrame,
                             keep_all: bool = False
) -> pd.DataFrame:
    """
    Filter de_genes_df to keep only the genes that are in the unpast_df.
    Add four new columns to the original unpast table.
    """
    # add columns to unpast_df
    new_unpast_df = unpast_df.assign(
        n_genes_DE=de_genes_df["n_genes"],
        genes_DE=de_genes_df["genes"],
        genes_up_DE=de_genes_df["genes_up"],
        genes_down_DE=de_genes_df["genes_down"],
    )
    
    # filter de_genes_df to keep only the genes that are in the unpast_df
    if not keep_all:
        logging.info("Filtering DE genes not in the original UnPaSt output")
        new_unpast_df = filter_de_genes(new_unpast_df, columns_to_process=["genes", "genes_down", "genes_up"])

    # update n_genes_DE column with the number of genes in the genes_DE column
    new_unpast_df["n_genes_DE"] = new_unpast_df["genes_DE"].apply(lambda x: len(x.split()) if pd.notna(x) else 0)

    return new_unpast_df


def read_dataframe_from_file(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        logging.error("File does not exist: %s", file_path)
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if os.stat(file_path).st_size == 0:
        logging.error("File is empty: %s", file_path)
        raise ValueError(f"File is empty: {file_path}")

    df = pd.read_csv(file_path, delimiter=DELIMITER, comment='#', index_col=0)

    if df.empty:
        logging.error("UnPaSt output is empty: %s", file_path)
        raise ValueError(f"Dataframe is empty: {file_path}")

    return df


def write_result(df: pd.DataFrame, input_file_path: str, output_file_path: str) -> pd.DataFrame:
    # Checking if input file exists and if it's not empty
    if not os.path.isfile(input_file_path):
        logging.error("Input file does not exist: %s", input_file_path)
        raise FileNotFoundError(f"Input file does not exist: {input_file_path}")

    if os.stat(input_file_path).st_size == 0:
        logging.error("Input file is empty: %s", input_file_path)
        raise ValueError(f"Input file is empty: {input_file_path}")

    # Check if output directory is writable
    output_dir_name = os.path.dirname(output_file_path)
    if not os.access(output_dir_name, os.W_OK):
        logging.error("Output directory is not writable: %s", output_dir_name)
        raise PermissionError(f"Output directory is not writable: {output_dir_name}")

    with open(input_file_path, "r") as f_in, open(output_file_path, "w") as f_out:
        comment_line = f_in.readline()
        f_out.write(comment_line)
        df.to_csv(f_out, sep=DELIMITER)


def safe_remove(file_path: str) -> None:
    try:
        os.remove(file_path)
    except OSError as e:
        logging.error(f"Error: {file_path} : {e.strerror}")


def run_de_for_unpast(
    unpast_output_path: str,
    expression_matrix_path: str,
    counts: bool = False,
    keep_all: bool = False,
    adj_p_value_cut_off: float = 0.05,
    logFC_cut_off: float = 1,
    num_genes_cut_off: float = float('inf'),
    r_script_path: str = None,
    r_executable_path: str = None,
) -> None:
    """
    A function that runs differential expression analysis using the limma package in R for unpast biclusters.
    """
    # Read unpast output
    # Read file with rownames in the first columns, first line is the comment line, colnames in the second line
    logging.info("Reading UnPaSt biclusters output and extracting samples to compare")
    # Check file existance, emptiness, and read DataFrame
    unpast_df = read_dataframe_from_file(unpast_output_path)
    # get the samples to compare
    samples_to_compare = os.path.join(
        os.path.dirname(unpast_output_path),
        f"{os.path.splitext(os.path.basename(unpast_output_path))[0][:-len('_biclusters')]}_samples{os.path.splitext(os.path.basename(unpast_output_path))[1]}",
    )
    # save samples to file
    extract_samples_to_file(unpast_df, samples_to_compare)

    # run add_genes.R script
    logging.info("Running DE analysis for unpast biclusters, takes a while...")
    samples_with_genes_path = run_add_genes_script(
        samples_to_compare,
        expression_matrix_path,
        counts,
        adj_p_value_cut_off,
        logFC_cut_off,
        num_genes_cut_off,
        r_script_path,
        r_executable_path,
    )
    
    logging.info("Adding DE genes to UnPaSt output table")
    # Read the unpast output file into a pandas dataframe.
    # Read file with rownames in the first columns, first line is the comment line, colnames in the second line
    de_genes_df = pd.read_csv(samples_with_genes_path.strip(), delimiter=DELIMITER, header=0, index_col=0)
     
    # add columns to unpast_df and filter de_genes_df to keep only the genes that are in the unpast_df
    new_unpast_df = add_columns_to_unpast_df(unpast_df, de_genes_df, keep_all=keep_all)

    # write new_unpast_df to file
    # use the original unpast_df file name and add _DE to the end
    # also add the comment line from the original unpast_df to the top of the new file
    output_path_de = unpast_output_path.replace(".tsv", f'_DE.pval{adj_p_value_cut_off}.logFC{logFC_cut_off}.tsv')
    #write_result(new_unpast_df, unpast_output_path, output_path_de)

    # remove the temporary files
    logging.info("Removing temporary files")
    safe_remove(samples_to_compare)
    safe_remove(samples_with_genes_path.strip())
    
    # keep only columns with genes
    cols = ["n_genes","genes","n_genes_DE","genes_DE","genes_up_DE","genes_down_DE"]
    de_df = new_unpast_df.loc[:,cols]
    cols = ["genes","genes_DE","genes_up_DE","genes_down_DE"]
    de_df.loc[:,cols] = de_df.loc[:,cols].fillna("").applymap(lambda row: set(row.split(" ")))
    return de_df