"""
Script to convert the data into training set. 

Change the EXP_TYPES if you have different experiments.
Data must be in 02_Pozyx_Positioning_Data and the corresponding labels in 
03_Labels.
"""
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib
from matplotlib import patches
import sys, os
from utils import utils
import dataframe_image as dfi
from PIL import Image
from collections import defaultdict
from scipy.signal import find_peaks
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)
import json
from datetime import datetime
import argparse
from typing import List, Any

from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

cpus = cpu_count() - 1
# ==================================================
# HELPER FUNCTIONS

def parse_label_actions(parsed_labels: List[dict[str, Any]]) -> dict[str, List[dict[str, str]]]:
    actions = defaultdict(list)
    for label in parsed_labels:
        begin_or_end, action_name = label["Label"].split("_")
        # Assume that a BEGIN timestamp will be immediately followed by an END timestamp
        num_same_action = len(actions[action_name])
        if num_same_action == 0:
            actions[action_name].append({})
        if begin_or_end in actions[action_name][num_same_action - 1]:
            actions[action_name].append({begin_or_end: label["Timestamp"]})
        else:
            actions[action_name][num_same_action - 1][begin_or_end] = label["Timestamp"]

    return actions

# ==================================================

# Parse arguments
parser = argparse.ArgumentParser(description='Split the raw data using the labels.')
parser.add_argument('--exp', metavar='-x', type=str, nargs='+', required=True,
                    help='the experiment type (space separated).')
args = parser.parse_args()

# MARK: - Config
EXP_TYPES = args.exp
# ACTION_PERIOD = ['grab something', ]
# tagId = "0x683f"
# regions_fp = Path().joinpath("outputs", "REGIONS", "2023-03-14 12:15:31.794149.json")
output_dir = Path().joinpath("outputs", "RAW_SPLIT_DATA")
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
position_data_dir = Path().joinpath("data", "02_Pozyx_Positioning_Data")
label_data_dir = Path().joinpath("data", "03_Labels")

# MARK: - Preprocessing
# Cleaning the data such as handling_spikes and MAV

all_cleaned_data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict) # {name_of_file: pd.DataFrame}

def split_exp_to_actions(experiment):
    for data_fp in position_data_dir.joinpath(experiment).glob("*.csv"):
        try:
            label_fp = Path().joinpath('data', '03_Labels', experiment, data_fp.name.replace(".csv", ".txt"))
            labels = utils.extract_time_labels(label_fp)
            actions = parse_label_actions(labels)
            data = pd.read_csv(data_fp)

            data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Pressure', 'TagId']
            data = data.set_index('Timestamp')
            data = data.sort_index()

            for action_name in actions:
                for ind, time_range in enumerate(actions[action_name]):
                    action_begin_time, action_end_time = time_range["BEGIN"], time_range["END"]
                    action_segment = data[action_begin_time:action_end_time]

                    # DEBUG - Show segment
                    # ax = action_segment[['POS_X', 'POS_Y', 'POS_Z']].plot(legend=False)
                    # ax.set_title(action_name)
                    # plt.show()

                    # Output the to .csv
                    if not output_dir.joinpath(action_name).exists():
                        output_dir.joinpath(action_name).mkdir(parents=True, exist_ok=True)
                    output_fn = data_fp.name.replace(".csv", "") + "_" + action_name + "_" + str(ind) + ".csv"
                    if not output_dir.joinpath(action_name, output_fn).exists():
                        action_segment.to_csv(output_dir.joinpath(action_name, output_fn))
        except (KeyError, IndexError) as e:
            print("\n\n")
            print("Error in: ", data_fp, action_name)
            print(actions)
            print(type(e), e)

for _ in tqdm(ThreadPool(cpus).imap_unordered(split_exp_to_actions, EXP_TYPES), total=len(EXP_TYPES), desc="Experiment"):
    pass