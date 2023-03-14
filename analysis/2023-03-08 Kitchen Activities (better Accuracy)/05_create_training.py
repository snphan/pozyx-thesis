"""
Script to convert the data into training set
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib
from matplotlib import patches
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from analysis.utils import utils
import dataframe_image as dfi
from PIL import Image
from collections import defaultdict
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)
import json


EXP_TYPES = ['ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS']
# ACTION_PERIOD = ['grab something', ]
tagId = "0x683f"
regions_fp = Path().joinpath("04_outputs", "REGIONS", "2023-03-14 12:15:31.794149.json")



##################################################
# Preprocessing
##################################################
all_cleaned_data = defaultdict(lambda: {}) # {name_of_file: pd.DataFrame}
for experiment in EXP_TYPES: 
    means = pd.DataFrame()
    stds = pd.DataFrame()
    durations = pd.DataFrame()
    for data_path in Path().joinpath('02_Pozyx_Positioning_Data', experiment).glob('*.csv'):
        data = pd.read_csv(data_path)

        data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'Pressure', 'TagId']
        data = data.set_index('Timestamp')

        cleaned_data = (data
                    .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'ACC_X', 'ACC_Y', 'ACC_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch', 'Pressure']]
                    .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                    .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], 20)
                    )
        all_cleaned_data[experiment][data_path.name.replace('.csv', '')] = cleaned_data

##################################################
# Windowing
##################################################
for experiment in EXP_TYPES:
    for data_name in all_cleaned_data[experiment]:
        cleaned_data = all_cleaned_data[experiment][data_name]

        label_fp = Path().joinpath('03_Labels', experiment, data_name + ".txt")
        labels = utils.extract_time_labels(label_fp)

        # Get the activity periods
        activity_periods = []
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
            activity_data = cleaned_data.loc[start:end]

            mean = activity_data.mean()
            mean.index = ['MEAN_' + ind for ind in mean.index]

            median = activity_data.median()
            median.index = ['MEDIAN_' + ind for ind in median.index]

            std = activity_data.std()
            std.index = ['STD_' + ind for ind in std.index]

            mode = activity_data.copy().pipe(utils.round_cols, ['POS_X', 'POS_Y', 'POS_Z'], 50).mode().iloc[0, :] # 50 mm = 5 cm, mode may output 2 rows
            mode.index = ['MODE_' + ind for ind in mode.index]

            max_value = activity_data.max()
            max_value.index = ['MAX_' + ind for ind in max_value.index]

            min_value = activity_data.min()
            min_value.index = ['MIN_' + ind for ind in min_value.index]

            regions = None
            with open(regions_fp) as f:
                regions = json.load(f)
            mode_location = pd.Series(activity_data.copy().pipe(utils.determine_location, regions).loc[:, 'Location'].mode()[0], index=["LOCATION"])

            print(pd.concat([mean, median, std, mode, max_value, min_value, mode_location]))
            # Find Location


# Feature Extraction

# Output the labelled data


