import numpy as np
import pandas as pd
import os, sys, time, glob
import json
import warnings
from tqdm import tqdm
import nflows.utils as torchutils
from IPython.display import clear_output
from time import time
from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from os.path import exists
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open_json(
    file_name, dir_path
):
    ''' 
    Opens a json file, loads data as a dictionary, and closes the file 
    Inputs:
        file_name: /name of json file.json
        dir_path: directory containing json files 
    Outputs:
        data: dictionary containing json content
    '''
    f = open(dir_path + file_name)
    data = json.load(f)
    f.close()
    return data

def get_names(
    path, label, set, num
):
    ''' 
    Gets the file path for the fixed data
    Inputs:
        path: string, directory to point to
        label: string, label assigned during nmma light curve generation
        set: int, number in directory name
        num: int, number of files to unpack
    Outputs: 
        file_names: list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}_batch_{}/{}_{}_{}.json'.format(
            label, set, label, set, i
        )
        file_names[i] = one_name
    return file_names

def json_to_df(
    file_name, dir_path, detection_limit, bands
):
    ''' 
    Flattens a light curve json file into a DataFrame
    Inputs:
        file_name: light curve json file name
        detection_limit: float, photometric detection limit
        bands: list, contains the json photometry keys as strings
    Outputs:
        df_unpacked: DataFrame containing the photometry data, time, 
                      and number of total detections across all bands
    '''
    data = open_json(file_name, dir_path)
    df = pd.DataFrame.from_dict(data, orient="columns")
    df_unpacked = pd.DataFrame(columns=bands)
    counter = 0
    for j in range(len(bands)):
        df_unpacked[['t', bands[j], 'x']] = pd.DataFrame(
            df[bands[j]].tolist(), index= df.index
        )
        for val in df_unpacked[bands[j]]:
            if val != detection_limit:
                counter += 1
            else:
                pass
    df_unpacked['num_detections'] = np.full(len(df_unpacked), counter)
    df_unpacked = df_unpacked.drop(columns=['x'])
    return df_unpacked

def extract_number(
    file_name
):
    '''
    Gets the number in a file name
    Inputs:
        file_name: string, name of file that has a number
    Outputs:
        Number in file name, or inf if none
    '''
    try:
        return int("".join(filter(str.isdigit, file_name)))
    except ValueError:
        return float("inf")

def directory_json_to_df(
    dir_path, label, detection_limit, bands
):
    '''
    Takes a directory of light curves and converts them to DataFrames
    Inputs:
        dir_path: directory containing light curve files
        label: string, label used when generating the light curve data
        detection_limit: float, photometric detection limit
        bands: list, contains the json photometry keys as strings
    Outputs:
        df_list: list, contains all files as DataFrames
    '''
    df_list = []
    for file in sorted(os.listdir(dir_path), key=extract_number):
        if file.endswith(".json") and file.startswith(label):
            df = json_to_df(
                file, dir_path, detection_limit, bands
            )
            df['simulation_id'] = extract_number(file)
            df_list.append(df)
    return df_list

def pad_the_data(df, t_min, t_max, step, data_filler, bands):
    '''
    Takes DataFrames and adds filler values to both ends, preserves 
    original time information.
    Inputs:
        df: DataFrame with 't' and photometric columns 
        t_min: float, global minimum start time
        t_max: float, global maximum end time
        step: float, time step between rows
        data_filler: value to use in the filler rows
        bands: list of photometric columns to fill
                (e.g. ['ztfg', 'ztfr', 'ztfi'])
    Outputs:
        df_padded: DataFrame with original data and padded rows,
                    covering full time range
    '''
    df = df.copy()
    df = df.sort_values('t').reset_index(drop=True)
    full_time = np.arange(t_min, t_max, step)
    num_points = len(full_time)
    t_start = df['t'].min()
    t_end = df['t'].max()
    prepend_times = full_time[full_time < t_start]
    append_times = full_time[full_time > t_end]

    def make_filler(times):
        return pd.DataFrame({
            't': times,
            **{f: data_filler for f in bands},
            'simulation_id': np.nan,
            'num_detections': np.nan
        })

    prepend_df = make_filler(prepend_times)
    append_df = make_filler(append_times)
    df_padded = pd.concat([prepend_df, df, append_df], ignore_index=True)
    df_padded = df_padded.sort_values('t').reset_index(drop=True)
    assert np.isclose(df_padded['t'].min(), t_min), \
    f"Start time is {df_padded['t'].min()}, expected {t_min}"
    assert np.isclose(df_padded['t'].max(), t_max - step), \
    f"End time is {df_padded['t'].max()}, expected {t_max - step}"
    try:
        assert len(df_padded) == num_points+1, \
        f"Length is {len(df_padded)}, expected {num_points+1}"
    except AssertionError as e:
        count = num_points + 1 - len(df_padded)
        addt_times = np.arange(t_max, t_max+(step*count), step)
        addt_append = make_filler(addt_times)
        df_padded = pd.concat([df_padded, addt_append], ignore_index=True)
        df_padded = df_padded.sort_values('t').reset_index(drop=True)
        assert len(df_padded) == num_points+1, \
        f"Length is {len(df_padded)}, expected {num_points+1}"
    return df_padded

def pad_all_dfs(
    df_list, t_min, t_max, step, data_filler, bands
):
    '''
    Pads multiple DataFrames at a time
    Inputs: 
        df_list: list of DataFrames to pad
    Outputs:
        padded_df_list: list of DataFrames after padding
    '''
    padded_df_list = []
    for i in tqdm(range(len(df_list))):
        df = df_list[i]
        sim_num = df.iloc[0, df.columns.get_loc('simulation_id')]
        det_num = df.iloc[0, df.columns.get_loc('num_detections')]
        df = pad_the_data(
            df, t_min, t_max, step, data_filler, bands
        )
        df['simulation_id'] = np.full(len(df), sim_num)
        df['num_detections'] = np.full(len(df), det_num)
        padded_df_list.append(df)
    return padded_df_list

def find_min_max_t(
    df_list
):
    '''
    Finds the maximum and minimum time of all provided DataFrames 
    Inputs:
        df_list: list of DataFrames
    Outputs:
        t_min: float, minimum time across all DataFrames
        t_max: float, maximum time across all DataFrames
    '''
    t_mins = []
    t_maxs = []
    for i in range(len(df_list)):
        t_mins.append(df_list[i]['t'].min())
        t_maxs.append(df_list[i]['t'].max())
    t_min = min(t_mins)
    t_max = max(t_maxs)
    return t_min, t_max

def grab_injection(
    inj_file, dir_path
):
    '''
    Reads in the injection file
    Inputs:
        inj_file: string, injection file name
        dir_path: string, directory path
    Outputs:
        inj_df: DataFrame containing the injection parameters
    '''
    data = open_json(inj_file, dir_path)
    content = data['injections']['content']
    inj_df = pd.DataFrame.from_dict(content)
    return inj_df

def load_light_curves_df(
    dir_path, 
    inj_file, 
    label, 
    detection_limit, 
    bands, 
    step, 
    data_filler,
    num_repeats=1,
    add_batch_id=False,
):
    '''
    Converts NMMA generated light curves to a DataFrame
    Inputs:
        dir_path: string, directory path
        inj_file: string, injection file name
        label: string, label assigned during nmma light curve generation
        detection_limit: float, photometric detection limit
        bands: list of photometric columns to fill
               (e.g. ['ztfg', 'ztfr', 'ztfi'])
        step: float, time step between rows
        data_filler: float, value to use in the filler rows
        num_repeats: int, number of repeated injections
        add_batch_id: bool, adds batching based on repeats
    Outputs:
        lc_df: DataFrame, contains light curve and injection data
    '''
    df_list = directory_json_to_df(
        dir_path=dir_path, 
        label=label, 
        detection_limit=detection_limit, 
        bands=bands)
    t_min, t_max = find_min_max_t(df_list)
    num_points = len(np.arange(t_min, t_max, step)) + 1
    padded_list = pad_all_dfs(
        df_list, 
        t_min=t_min, 
        t_max=t_max, 
        step=step, 
        data_filler=data_filler, 
        bands=bands)
    all_padded_lcs = pd.concat(padded_list).reset_index(drop=True)
    inj_df = grab_injection(inj_file=inj_file, dir_path=dir_path)
    lc_df = all_padded_lcs.merge(inj_df, on='simulation_id')
    if num_repeats <= 0:
        print('Warning: num_repeats must be at least 1 (for one lc!).' + 
              'Defaulting to 1.')
        num_repeats = 1
    if add_batch_id == True:
        lc_df['batch_id'] = lc_df.index // (num_points * num_repeats)
    return lc_df

def df_to_tensor(
    lc_df, params, bands, num_repeats, num_points
):
    '''
    Converts DataFrames into pytorch tensors
    Inputs:
        lc_df: DataFrame, contains data and injection parameters
        params: list of injection parameters
        bands: list of photometric columns to fill
               (e.g. ['ztfg', 'ztfr', 'ztfi'])
        num_repeats: int, number of repeated injections
        num_points: int, length of each light curve
    Outputs:
        tensor_data: list of tensors containing lc data
        tensor_params: list of tensors containing lc params
    '''
    num_channels = len(bands)
    num_batches = len(lc_df['batch_id'].unique())
    tensor_data = []
    tensor_params = []
    for idx in tqdm(range(0, num_batches)):
        lc_data = torch.tensor(
            lc_df[bands].loc[lc_df['batch_id'] == idx].values.reshape(
                num_repeats, num_points, num_channels
            ), 
            dtype=torch.float32
        ).transpose(1, 2)
        lc_params = torch.tensor(
            lc_df[params].loc[lc_df['batch_id'] == idx].iloc[::num_points].values, 
            dtype=torch.float32
        ).unsqueeze(2).transpose(1,2)
        tensor_data.append(lc_data)
        tensor_params.append(lc_params)
    return tensor_data, tensor_params

def load_embedding_dataset(
    dir_path_var,
    inj_file_var, 
    label_var,
    dir_path_fix,
    inj_file_fix,
    label_fix,
    detection_limit, 
    bands, 
    step, 
    data_filler,
    params,
    num_repeats=1,
):
    '''
    Loads NMMA generated lc's into tensors suitable for training
    Inputs:
        dir_path_fix: string, directory path for fixed lcs
        dir_path_var: string, directory path for varied lcs
        inj_file_fix: string, injection file name for fixed lcs
        inj_file_var: string, injection file name for varied lcs
        label_fix: string, label assigned to fixed lcs
        label_var: string, label assigned to varied lcs
        detection_limit: float, photometric detection limit
        bands: list of photometric columns to fill
               (e.g. ['ztfg', 'ztfr', 'ztfi'])
        step: float, time step between rows
        data_filler: float, value to use for data padding
        params: list of injection parameters
        num_repeats: int, number of repeated injections
    Outputs:
        lc_data_fix: list of tensors containing fixed lc data
        lc_params_fix: list of tensors containing fixed lc params
        lc_data_var:list of tensors containing varied lc data
        lc_params_var: list of tensors containing varied lc params
    '''
    if num_repeats <= 0:
        print('Warning: num_repeats must be at least 1 (for one lc!).' + 
              'Defaulting to 1.')
        num_repeats = 1
        
    df_list_var = directory_json_to_df(
        dir_path=dir_path_var, 
        label=label_var, 
        detection_limit=detection_limit, 
        bands=bands)
    t_min, t_max = find_min_max_t(df_list_var)
    num_points = len(np.arange(t_min, t_max, step)) + 1
    padded_list_var = pad_all_dfs(
        df_list_var, 
        t_min=t_min, 
        t_max=t_max, 
        step=step, 
        data_filler=data_filler, 
        bands=bands)
    all_padded_lcs_var = pd.concat(padded_list_var).reset_index(drop=True)
    inj_df_var = grab_injection(inj_file=inj_file_var, dir_path=dir_path_var)
    lc_df_var = all_padded_lcs_var.merge(inj_df_var, on='simulation_id')
    lc_df_var['batch_id'] = lc_df_var.index // (num_points * num_repeats)
    lc_data_var, lc_params_var = df_to_tensor(
        lc_df=lc_df_var,
        params=params,
        bands=bands,
        num_repeats=num_repeats,
        num_points=num_points
    )

    if dir_path_fix and inj_file_fix and label_fix:
        df_list_fix = directory_json_to_df(
            dir_path=dir_path_fix, 
            label=label_fix, 
            detection_limit=detection_limit, 
            bands=bands)
        padded_list_fix = pad_all_dfs(
            df_list_fix, 
            t_min=t_min, 
            t_max=t_max, 
            step=step, 
            data_filler=data_filler, 
            bands=bands)
        all_padded_lcs_fix = pd.concat(padded_list_fix).reset_index(drop=True)
        inj_df_fix = grab_injection(inj_file=inj_file_fix, dir_path=dir_path_fix)
        lc_df_fix = all_padded_lcs_fix.merge(inj_df_fix, on='simulation_id')
        lc_df_fix['batch_id'] = lc_df_fix.index // (num_points * num_repeats)
        lc_data_fix, lc_params_fix = df_to_tensor(
            lc_df=lc_df_fix,
            params=params,
            bands=bands,
            num_repeats=num_repeats,
            num_points=num_points
        )
    else:
        lc_data_fix, lc_params_fix = None, None
    
    return lc_data_var, lc_params_var, lc_data_fix, lc_params_fix 

class Embedding_Data(Dataset):
    def __init__(
        self, 
        lc_data_var, 
        lc_params_var, 
        lc_data_fix=None, 
        lc_params_fix=None
    ):
        super().__init__()
        self.lc_data_var = lc_data_var
        self.lc_params_var = lc_params_var
        self.lc_data_fix = lc_data_fix
        self.lc_params_fix = lc_params_fix

    def __len__(self):
        return len(self.lc_data_var)

    def __getitem__(self, idx):
        return (
            self.lc_data_var[idx],
            self.lc_params_var[idx],
            self.lc_data_fix[idx] if self.lc_data_fix is not None else torch.tensor([]),
            self.lc_params_fix[idx] if self.lc_params_fix is not None else torch.tensor([])
        )
