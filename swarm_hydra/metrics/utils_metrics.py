import logging
from omegaconf import OmegaConf, DictConfig
import h5py
import numpy as np
import pandas as pd
import os

# A logger for this file
log = logging.getLogger(__name__)


def processing_folder_structure(
    experi_dir: str, model_name: str, m_r_name: str = "metrics_results"
) -> tuple:
    """..."""
    # Load experiment datasets and runtime cfgs
    subfolder_names = sorted(
        [
            pth
            for pth in os.listdir(experi_dir)
            if os.path.isdir(os.path.join(experi_dir, pth))
        ]
    )
    subfolder_names.remove(".hydra")
    if "metrics_results" in subfolder_names:
        subfolder_names.remove(m_r_name)

    cfg_experi = {}
    hdf5_experi = {}
    subset_subfolder_names = []
    for exp_fo_na in subfolder_names:
        if model_name in exp_fo_na:
            subset_subfolder_names.append(exp_fo_na)
            # Loading .yaml cfg
            runtime_cfg_name_res = (
                experi_dir + "/.hydra/" + exp_fo_na + "_runtime_config.yaml"
            )
            runtime_cfg = OmegaConf.load(runtime_cfg_name_res)
            # log.info("Runtime config details:")
            # log.info(runtime_cfg)
            cfg_experi[exp_fo_na] = runtime_cfg
            # Loading .hdf5 dataset
            h5df_dict = hdf5_data_extraction_fn(
                experi_dir + "/" + exp_fo_na + "/dataset.hdf5"
            )
            # log.info(f"Dataset hdf5 abs path {experi_dir+'/'+exp_fo_na+'/dataset.hdf5'}.")
            hdf5_experi[exp_fo_na] = h5df_dict

    return (
        cfg_experi,
        hdf5_experi,
        sorted(subset_subfolder_names, key=lambda x: int(x.split("_rep")[-1])),
    )


def hdf5_data_extraction_fn(dataset_pth: str) -> dict:
    """"""
    hdf5_f = h5py.File(dataset_pth, "r")
    hdf5_dict = {}
    for key, val in hdf5_f.items():
        hdf5_dict[key] = np.array(val)
        # log.info(f'k:{key}') # , v:{val}')

    return hdf5_dict


def sort_array_by_indices(arr: np.array, sorted_indices: np.array) -> np.array:
    """
    Sort a 2D or 3D NumPy array based on a corresponding 1D or 2D array of sorted indices.

    Args:
        arr (np.ndarray): The input array to be sorted (2D or 3D).
        sorted_indices (np.ndarray): The array containing sorted indices (1D or 2D).

    Returns:
        np.ndarray: The sorted array.
    """

    if sorted_indices.ndim == 1:
        return arr[sorted_indices]  # 2D array with 1D indices
    elif arr.ndim == 2 and sorted_indices.ndim == 2:
        return np.take_along_axis(arr, sorted_indices, axis=1)  # 2D sorting
    elif arr.ndim == 3 and sorted_indices.ndim == 2:
        return np.take_along_axis(arr, sorted_indices[..., None], axis=1)  # 3D sorting
    else:
        raise ValueError("Invalid combination of array and sorted_indices dimensions.")


def sort_arrays_by_indices(arrs: list, sorted_indices: np.array) -> list:
    """
    Sort a list of NumPy array based on a corresponding array of sorted indices.

    Args:
        arr (list): The input list of arrays to be sorted.
        sorted_indices (np.ndarray): The array containing sorted indices (1D or 2D).

    Returns:
        list: The sorted list of arrays.
    """
    sorted_arrs = []
    for arr in arrs:
        sorted_arrs.append(sort_array_by_indices(arr, sorted_indices))
    return sorted_arrs


def normalizing_quantities(
    to_normalize: np.array, q_type: str, params: dict
) -> np.array:
    """Normalize all the 'physical' values to be agnostic always of the reference frame …"""
    if q_type == "abs_position":
        return (to_normalize - params["min_val"]) / (
            params["max_val"] - params["min_val"]
        )
    elif q_type == "directional":
        return to_normalize / np.linalg.norm(to_normalize, axis=-1, keepdims=True)
    else:
        raise ValueError(f"q_type {q_type} not recognized.")


def normalizing_multiple_quantities(
    to_normalize: list, q_type: list, params: dict
) -> list:
    """
    Normalize a list of NumPy array based on their q_type.

    Args:
        to_normalize (list): The input list of arrays to be normalized.
        q_type (list): The array containing sorted indices (1D or 2D).

    Returns:
        list: The list of normalized arrays.
    """
    normalized_arrs = []
    for arr, q_t in zip(to_normalize, q_type):
        normalized_arrs.append(normalizing_quantities(arr, q_t, params))
    return normalized_arrs


def open_dataframe(file_path: str, columns: list) -> pd.DataFrame:
    """
    Opens an existing CSV file as a DataFrame or creates a new one with the specified columns.
    A default index column is automatically included.
    """
    # if os.path.exists(file_path):
    #     df = pd.read_csv(file_path, index_col=0)  # Load existing DataFrame with index
    # else:
    #     df = pd.DataFrame(columns=columns)  # Create new DataFrame
    log.info(f"Does the .csv exists? : {os.path.exists(file_path)}")
    return pd.DataFrame(columns=columns)  # Create new DataFrame


def append_row_to_dataframe(df: pd.DataFrame, row_data: dict) -> pd.DataFrame:
    """
    Appends a new row to the DataFrame, ensuring correct column alignment.
    """
    df = df.copy()  # Avoid modifying original
    df.loc[len(df)] = row_data
    return df


def export_dataframe(
    df: pd.DataFrame, file_path: str, latex_columns: list[list] | None
):
    """
    Exports the DataFrame to CSV and a subset of columns to a LaTeX table.
    """
    df.to_csv(file_path, index=True)  # Save DataFrame to CSV with index

    # Export subset of columns to LaTeX
    if latex_columns:
        for idx, latex_columns in enumerate(latex_columns):
            latex_df = df[latex_columns]
            latex_table = latex_df.to_latex(
                index=False, escape=False
            )  # Convert to LaTeX
            with open(file_path.replace(".csv", f"_{idx}.tex"), "w") as f:
                f.write(latex_table)  # Save LaTeX table


def yaml_to_dataframe(configs: dict[DictConfig]) -> pd.DataFrame:
    """Combines multiple hydra YAML configs into a single DataFrame."""
    data = {}
    for _, config_val in configs.items():
        cfg_dict = OmegaConf.to_container(config_val)
        for key, val in cfg_dict.items():
            data.setdefault(key, []).append(val)

    # Ensure all rows have the same length
    max_len = max(map(len, data.values()))
    for key in data:
        while len(data[key]) < max_len:
            data[key].append(None)  # Fill missing values

    return pd.DataFrame.from_dict(
        data, orient="index", columns=[f"Config {i+1}" for i in range(len(configs))]
    )
