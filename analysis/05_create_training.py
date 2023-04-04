"""
Script to convert the data into training set. 

Change the EXP_TYPES if you have different experiments.
Data must be in 02_Pozyx_Positioning_Data and the corresponding labels in 
03_Labels.
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib
from matplotlib import patches
from utils import utils
import dataframe_image as dfi
from PIL import Image
from collections import defaultdict
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)
import json
from datetime import datetime

# MARK: - Config
# EXP_TYPES = ['ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS']
EXP_TYPES = ['MINCE','MOP','TIESHOES','BRUSHTEETH','ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS']
# ACTION_PERIOD = ['grab something', ]
tagId = "0x683f"
regions_fp = Path().joinpath("outputs", "REGIONS", "2023-03-14 12:15:31.794149.json")
output_dir = Path().joinpath("outputs", "TRAINING")


# MARK: - Preprocessing
# Cleaning the data such as handling_spikes and MAV

all_cleaned_data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict) # {name_of_file: pd.DataFrame}

for experiment in EXP_TYPES: 
    means = pd.DataFrame()
    stds = pd.DataFrame()
    durations = pd.DataFrame()
    for data_path in Path().joinpath('data', '02_Pozyx_Positioning_Data', experiment).glob('*.csv'):
        data = pd.read_csv(data_path)

        data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Pressure', 'TagId']
        data = data.set_index('Timestamp')

        cleaned_data = (data
                    .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'ACC_X', 'ACC_Y', 'ACC_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch', 'Pressure']]
                    .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                    .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], 20)
                    )
        all_cleaned_data[experiment][data_path.name.replace('.csv', '')] = cleaned_data

print('Done Cleaning!')

# MARK: - Windowing
# Separate the data into windows dict of ACTIVITY: [pd.DataFrame] 

windows: dict[str, list] = defaultdict(list)
WINDOW_WIDTH = 2 # seconds
WINDOW_STRIDE = 1 # seconds

for experiment in EXP_TYPES:
    for data_name in all_cleaned_data[experiment]:
        cleaned_data = all_cleaned_data[experiment][data_name]

        label_fp = Path().joinpath('data', '03_Labels', experiment, data_name + ".txt")
        labels = utils.extract_time_labels(label_fp)

        # Get the activity periods assume these periods occur between quiet standing.
        # Note that the transitional periods should be taken out (from quiet standing to doing the activity)
        activity_periods: list[dict[str, str]] = [] 
        i = 0
        while i < len(labels):
            if 'quiet' in labels[i]['Label']:
                i += 1
                # Find the next quiet standing
                start = labels[i]
                if i >= len(labels) - 1: break
                while 'quiet' not in labels[i]['Label']:
                    i += 1
                end = labels[i]
                i -= 1
                activity_periods.append([ start, end ])
            i += 1

        for period in activity_periods:
            start = float(period[0]['Timestamp'])
            end = float(period[1]['Timestamp'])
            
            while start < end - WINDOW_WIDTH:
                # DEBUG
                # ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['POS_X', 'POS_Y', 'POS_Z'], labels)
                # ax.axvline(start, color='green')
                # ax.axvline(start+WINDOW_WIDTH, color='green')
                # plt.show()
                windows[experiment].append(cleaned_data.loc[start: start+WINDOW_WIDTH])
                start += WINDOW_STRIDE


        quiet_periods: list[dict[str, str]] = []
        i = 0
        while i < len(labels):
            if 'quiet' in labels[i]['Label']:
                quiet_periods.append([labels[i], labels[i+1]])
            i += 1

        for period in quiet_periods:
            start = float(period[0]['Timestamp'])
            end = float(period[1]['Timestamp'])
            
            while start < end - WINDOW_WIDTH:
                # DEBUG
                # ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['POS_X', 'POS_Y', 'POS_Z'], labels)
                # ax.axvline(start, color='green')
                # ax.axvline(start+WINDOW_WIDTH, color='green')
                # plt.show()
                windows["UNDEFINED"].append(cleaned_data.loc[start:start+WINDOW_WIDTH])
                start += WINDOW_STRIDE

print('Done Windowing!')

# MARK: - Feature Extraction
# Extract the features from each window and label with activity

training_data = pd.DataFrame()

for experiment in windows:
    for data in windows[experiment]:
        mean = data.mean()
        mean.index = ['MEAN_' + ind for ind in mean.index]

        median = data.median()
        median.index = ['MEDIAN_' + ind for ind in median.index]

        std = data.std()
        std.index = ['STD_' + ind for ind in std.index]

        mode = data.copy().pipe(utils.round_cols, ['POS_X', 'POS_Y', 'POS_Z'], 50).mode().iloc[0, :] # 50 mm = 5 cm, mode may output 2 rows
        mode.index = ['MODE_' + ind for ind in mode.index]

        max_value = data.max()
        max_value.index = ['MAX_' + ind for ind in max_value.index]

        min_value = data.min()
        min_value.index = ['MIN_' + ind for ind in min_value.index]

        regions = None
        with open(regions_fp) as f:
            regions = json.load(f)
        mode_location = pd.Series(data.copy().pipe(utils.determine_location, regions).loc[:, 'Location'].mode()[0], index=["LOCATION"])

        activity_type = pd.Series([experiment], index=['ACTIVITY'])

        feature_vector = pd.concat([mean, median, std, mode, max_value, min_value, mode_location, activity_type])

        training_data = pd.concat([training_data, feature_vector], axis=1)

print("Training Data Set Complete")

# MARK: - Output
# Output the Labelled Data

output_dir.mkdir(parents=True, exist_ok=True)

(training_data
    .T
    .reset_index(drop=True)
    .to_csv(output_dir.joinpath(f"{datetime.now()}_W{WINDOW_WIDTH}_S{WINDOW_STRIDE}_training.csv"), index=False)
)


