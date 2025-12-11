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
from .resnet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bands = ['ztfg', 'ztfr', 'ztfi']
detection_limit = 22.0
num_repeats = 50
num_channels = 3
num_points = 121

t_zero = 44242.00021937881
t_min = 44240.00050450478
t_max = 44269.99958898723
days = int(round(t_max - t_min))
time_step = 0.25

def open_json(file_name, dir_path):
    '''
    Opens a json file, loads the data as a dictionary, and closes the file
    Inputs:
        file_name = /name of json file.json
        dir_path = directory containing json files
    Returns:
        data = dictionary containing json content
    '''
    f = open(dir_path + file_name)
    data = json.load(f)
    f.close()
    return data

def get_names(path, label, set, num):
    '''
    Gets the file path for the fixed data
    Inputs:
        path = string, directory to point to
        label = string, label assigned during nmma light curve generation
        set = int, number in directory name
        num = int, number of files to unpack
    Returns:
        file_names = list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}_batch_{}/{}_{}_{}.json'.format(label, set, label, set, i)
        file_names[i] = one_name
    return file_names

def json_to_df(file_names, num_sims, detection_limit=detection_limit, bands=bands):
    '''
    Flattens json files into a dataframe
    Inputs:
        file_names = list, contains full path file names as strings
        num_sims = int, number of files to unpack
        detection_limit = float, photometric detection limit
        bands = list, contains the json photometry keys as strings
    Returns:
        df_list = list of dataframes containing the photometry data, time, and number of total detections across all bands
    '''
    df_list = [0] * num_sims
    for i in tqdm(range(num_sims)):
        data = json.load(open(file_names[i], "r"))
        df = pd.DataFrame.from_dict(data, orient="columns")
        df_unpacked = pd.DataFrame(columns=bands)
        counter = 0
        for j in range(len(bands)):
            df_unpacked[['t', bands[j], 'x']] = pd.DataFrame(df[bands[j]].tolist(), index= df.index)
            for val in df_unpacked[bands[j]]:
                if val != detection_limit:
                    counter += 1
                else:
                    pass
        df_unpacked['num_detections'] = np.full(len(df_unpacked), counter)
        df_unpacked['sim_id'] = np.full(len(df_unpacked), i)
        df_unpacked = df_unpacked.drop(columns=['x'])
        df_list[i] = df_unpacked
    return df_list

def gen_prepend_filler(column_list, detection_limit, t_min, t_max, step = 0.25):
    '''
    front end padding
    Inputs:
        column_list: list of columns of the dataframe
        detection_limit: number that is used as the filler, ie the detection limit
        t_min: minimum time
        t_max: maximum time
        step: time increment
    Outputs:
        filler_df: dataframe to pad the existing data
    '''
    # Fill according to step size, regardless of count
    pre = np.arange(start=t_min, stop=t_max, step=step)
    prefill_dict = {}
    for col in column_list:
        if col == 't':
            prefill_dict['t'] = pre
        else:
            prefill_dict[col] = [detection_limit]*len(pre)
    prefill_df = pd.DataFrame(prefill_dict)
    return prefill_df

def gen_append_filler(column_list, detection_limit, t_min, count, step=0.25):
    '''
    back end padding
    Inputs:
        column_list: list of columns of the dataframe
        detection_limit: number that is used as the filler, ie the detection limit
        t_min: minimum time
        t_max: maximum time
        step: time increment
    Outputs:
        filler_df: dataframe to pad the existing data
    '''
    # Fill according to min value and specified count
    aft = np.arange(start=t_min, stop=t_min+(count*step), step=step)
    aftfill_dict = {}
    for col in column_list:
        if col == 't':
            aftfill_dict['t'] = aft
        else:
            aftfill_dict[col] = [detection_limit]*len(aft)
    aftfill_df = pd.DataFrame(aftfill_dict)
    return aftfill_df

def pad_the_data(actual_df, column_list, desired_count=num_points, filler_time_step=time_step, filler_data=detection_limit):
    '''
    pads both ends of the light curve dataframe to ensure consistent number of data points across all data
    Inputs:
        actual_df: existing light curve data
        desired_count: desired length of data, ie number of points for the light curve
        filler_time_step: time increment
        filler_data: number that is used as the filler, ie the detection limit
    Outputs:
        cat_df: padded dataframe
    '''
    actual_df.iloc[:, actual_df.columns.get_loc('t')] = actual_df.iloc[:, actual_df.columns.get_loc('t')].apply(lambda x: x - t_min)
    cat_df = actual_df
    cat_count = len(cat_df)
    prepended_count = 0
    if (actual_df['t'].min() >= filler_time_step):
        filler_max_time = actual_df['t'].min() - filler_time_step  # stop one time step before current min
        prepend_filler_df = gen_prepend_filler(column_list, filler_data, 0, filler_max_time, filler_time_step)
        prepended_count = len(prepend_filler_df)
        cat_df = pd.concat([prepend_filler_df, actual_df], ignore_index=True)
        cat_count = len(cat_df)
    append_count = desired_count - cat_count
    if append_count > 0:
        max_t = cat_df['t'].max()
        steps_per_count = 1/filler_time_step
        filler_min_time = int(max_t*steps_per_count)/steps_per_count + filler_time_step  # start at next time step
        append_filler_df = gen_append_filler(column_list, filler_data, filler_min_time, append_count)
        cat_df = pd.concat([cat_df, append_filler_df], ignore_index=True)
        cat_count = len(cat_df)
    assert(len(cat_df) == desired_count)
    return cat_df

def pad_all_dfs(df_list):
    '''
    Pads multiple dataframes at a time
    Inputs:
        df_list: list of dataframes to pad
    Outputs:
        padded_df_list: list of dataframes after padding
    '''
    padded_df_list = []
    for i in tqdm(range(len(df_list))):
        df = df_list[i]
        sim_num = df.iloc[0, df.columns.get_loc('sim_id')]
        det_num = df.iloc[0, df.columns.get_loc('num_detections')]
        df = pad_the_data(df)
        df['sim_id'] = np.full(len(df), sim_num)
        df['num_detections'] = np.full(len(df), det_num)
        padded_df_list.append(df)
    return padded_df_list

def load_in_data(data_dir, name, csv_no, num_points=num_points, num_repeats=num_repeats):
    '''
    Loading in multiple saved csv files containing light curve data as one dataframe
    Inputs:
        data_dir: directory containing the csv files
        csv_no: number of csv files to load in
        num_points: number of data points per light curve
        num_repeats: repeats of injection parameters to determine batches
    Outputs:
        data_df: single dataframe containing the data
    '''
    data_list = []
    for i in range (0, csv_no):
        data_list.append(pd.read_csv(data_dir + '{}_{}.csv'.format(name, i)))
    data_df = pd.concat(data_list)
    num_sims = int(len(data_df)/num_points)
    sim_list = []
    sim_no = 0
    for i in range(0, num_sims):
        for j in range(0, num_points):
            sim_list.append(sim_no)
        sim_no += 1
    data_df['sim_id'] = sim_list
    batch_list = []
    batch_no = 0
    num_batches = int((len(data_df)/num_points)/num_repeats)
    data_df = data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    for i in range(0, num_batches):
        for j in range(0, num_points*num_repeats):
            batch_list.append(batch_no)
        batch_no += 1
    data_df['batch_id'] = batch_list
    return data_df

def match_fix_to_var(data_dir, name1, name2, start, stop, num_points=num_points, num_repeats=num_repeats):
    '''
    Matches the shifted injection light curve data to its fixed counterpart
    Inputs:
        data_dir: directory containing the data
        name1: label of the varied csv files
        name2: label of the fixed csv files
        start: starting csv number
        stop: ending csv number
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        fixed_data_df: returns the fixed portion of the light curve data
        varied_data_df: returns the shifted/varied portion of the light curve data
    '''
    # initiate list for dataframes
    fixed_list = []
    varied_list = []
    # do all data processing for a given number of dataframes
    for i in range (start, stop):
        # load in the data
        df_var = pd.read_csv(data_dir + '{}_{}.csv'.format(name1, i))
        df_fix = pd.read_csv(data_dir + '{}_{}.csv'.format(name2, i))
        # match the two dataframes to each other based on sim id
        matched = df_var.merge(df_fix, left_on=['sim_id', df_var.groupby('sim_id').cumcount()],
                               right_on=['sim_id', df_fix.groupby('sim_id').cumcount()])
        # grab the fixed and varied portions of the dataframe
        fix_df = matched.iloc[:, 12:]
        var_df = matched.iloc[:, :12]
        # adjust columns and column names
        fix_df.columns = fix_df.columns.str.rstrip('_y')
        var_df.columns = var_df.columns.str.rstrip('_x')
        var_df = var_df.drop(columns=['key_1'])
        # add to list of dataframes
        fixed_list.append(fix_df)
        varied_list.append(var_df)
    # concatenate the list of dataframes together
    fixed_data_df = pd.concat(fixed_list)
    varied_data_df = pd.concat(varied_list)
    # overwrite the simulation id's and add batch id's
    num_sims = int(len(fixed_data_df)/num_points)
    sim_list = []
    sim_no = 0
    for i in range(0, num_sims):
        for j in range(0, num_points):
            sim_list.append(sim_no)
        sim_no += 1
    fixed_data_df['sim_id'] = sim_list
    varied_data_df['sim_id'] = sim_list
    batch_list = []
    batch_no = 0
    num_batches = int((len(fixed_data_df)/num_points)/num_repeats)
    fixed_data_df = fixed_data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    varied_data_df = varied_data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    for i in range(0, num_batches):
        for j in range(0, num_points*num_repeats):
            batch_list.append(batch_no)
        batch_no += 1
    fixed_data_df['batch_id'] = batch_list
    varied_data_df['batch_id'] = batch_list

    return fixed_data_df, varied_data_df

def matched(data_dir, name1, name2, start, stop, num_points=num_points, num_repeats=num_repeats):
    '''
    Matches light curves with the same injection parameters
    Inputs:
        data_dir: file path for data
        name1: label of the varied csv files
        name2: label of the fixed csv files
        start: starting csv number
        stop: ending csv number
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        matched_df: combined dataframe of the shifted and fixed light curves
    '''
    # initiate list for dataframes
    matched_list = []
    # do all data processing for a given number of dataframes
    for i in range (start, stop):
        # load in the data
        df_var = pd.read_csv(data_dir + '{}_{}.csv'.format(name1, i))
        df_fix = pd.read_csv(data_dir + '{}_{}.csv'.format(name2, i))
        # match the two dataframes to each other based on sim id
        matched = df_var.merge(df_fix, left_on=['sim_id', df_var.groupby('sim_id').cumcount()],
                               right_on=['sim_id', df_fix.groupby('sim_id').cumcount()])
        matched_list.append(matched)
    matched_df = pd.concat(matched_list)
    return matched_df

def add_batch_sim_nums_all(df, num_points=num_points, num_repeats=num_repeats):
    '''
    Adds a simulation and batch id number to each light curve
    Inputs:
        df: dataframe containing light curve data
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        None
    '''
    num_batches_split = int((len(df)/num_points)/num_repeats)
    batch_list_split = []
    batch_no = 0
    for i in range(0, num_batches_split):
        for j in range(0, num_repeats*num_points):
            batch_list_split.append(batch_no)
        batch_no += 1
    df['batch_id'] = batch_list_split

    num_sims_split = int(len(df)/num_points)
    sim_list_split = []
    sim_no = 0
    for i in range(0, num_sims_split):
        for j in range(0, num_points):
            sim_list_split.append(sim_no)
        sim_no += 1
    df['sim_id'] = sim_list_split

def get_test_names(path, label, set, num):
    '''
    Gets the file path for the fixed data
    Inputs:
        path = string, directory to point to
        label = string, label assigned during nmma light curve generation
        set = int, number in directory name
        num = int, number of files to unpack
    Returns:
        list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}{}_{}.json'.format(label, set, i)
        file_names[i] = one_name
    return file_names

def repeated_df_to_tensor(df_varied, df_fixed, batches):
    '''
    Converts dataframes into pytorch tensors
    Inputs:
        df_varied: dataframe containing the shifted light curve information
        df_fixed: dataframe containing the analagous fixed light curve information
        batches: number of unique mass, velocity, and lanthanide injections
    Outputs:
        data_shifted_list: list of tensors of shape [repeats, channels, num_points] containing the shifted light curve photometry
        data_unshifted_list: list of tensors of shape [repeats, channels, num_points] containing the fixed light curve photometry
        param_shifted_list: list of tensors of shape [repeats, 1, 5] containing the injection parameters of the shifted light curves
        param_unshifted_list: list of tensors of shape [repeats, 1, 5] containing the injection parameters of the fixed light curves
    '''
    data_shifted_list = []
    data_unshifted_list = []
    param_shifted_list = []
    param_unshifted_list = []
    for idx in tqdm(range(0, batches)):
        data_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels),
                                    dtype=torch.float32).transpose(1, 2)
        data_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels),
                                    dtype=torch.float32).transpose(1, 2)
        param_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[::num_points, 6:11].values,
                                    dtype=torch.float32).unsqueeze(2).transpose(1,2)
        param_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[::num_points, 5:10].values,
                                    dtype=torch.float32).unsqueeze(2).transpose(1,2)
        data_shifted_list.append(data_shifted)
        data_unshifted_list.append(data_unshifted)
        param_shifted_list.append(param_shifted)
        param_unshifted_list.append(param_unshifted)
    return data_shifted_list, data_unshifted_list, param_shifted_list, param_unshifted_list

class Paper_data(Dataset):
    def __init__(self, data_shifted_paper, data_unshifted_paper,
                 param_shifted_paper, param_unshifted_paper,
                 num_batches_paper_sample):
        super().__init__()
        self.data_shifted_paper = data_shifted_paper
        self.data_unshifted_paper = data_unshifted_paper
        self.param_shifted_paper = param_shifted_paper
        self.param_unshifted_paper = param_unshifted_paper
        self.num_batches_paper_sample = num_batches_paper_sample

    def __len__(self):
        return self.num_batches_paper_sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.param_shifted_paper[idx].to(device),
            self.param_unshifted_paper[idx].to(device),
            self.data_shifted_paper[idx].to(device),
            self.data_unshifted_paper[idx].to(device)
        )
