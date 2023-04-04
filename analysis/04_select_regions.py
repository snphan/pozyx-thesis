"""
Creates a json file with the regions defined

Change the EXP_TYPES in the config to show where the mean of each activity occurs
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
from utils import utils
import dataframe_image as dfi
from PIL import Image
from datetime import datetime
import json

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)


# MARK: - Config
##################################################
# EXP_TYPES = ['ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS'] # Select activities to see their average location
EXP_TYPES = []
tagId = "0x683f"
##################################################

mean_means = pd.DataFrame(columns=['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch'])
std_means = pd.DataFrame(columns=['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch'])

mean_stds = pd.DataFrame(columns=['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch'])
mean_durations = pd.DataFrame(columns=['Duration'])

for experiment in EXP_TYPES: 
    means = pd.DataFrame()
    stds = pd.DataFrame()
    durations = pd.DataFrame()
    for data_path in Path().joinpath('data', '02_Pozyx_Positioning_Data', experiment).glob('*.csv'):
        data = pd.read_csv(data_path)
        label_fp = Path().joinpath('data', '03_Labels', experiment, data_path.name.replace(".csv", ".txt"))

        labels = utils.extract_time_labels(label_fp)

        data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'Pressure', 'TagId']
        data = data.set_index('Timestamp')

        cleaned_data = (data
                    .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch']]
                    .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                    .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], 20)
                    )
        # Convert mm to cm
        cleaned_data.loc[:, ['POS_X', 'POS_Y', 'POS_Z']] = cleaned_data.loc[:, ['POS_X', 'POS_Y', 'POS_Z']] / 10

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

        # Calculate the features
        for period in activity_periods:
            start = float(period[0]['Timestamp'])
            end = float(period[1]['Timestamp'])
            activity_data = cleaned_data.loc[start:end]
            mean = activity_data.mean()
            std = activity_data.std()
            duration = end - start

            means = pd.concat([means, mean], axis=1)
            stds = pd.concat([stds, std], axis=1)
            durations = pd.concat([durations, pd.DataFrame([duration], index=["Duration"])], axis=1)
    
    mean_means = pd.concat([mean_means, means.mean(axis=1).rename(experiment).to_frame().T])
    std_means = pd.concat([std_means, means.std(axis=1).rename(experiment).to_frame().T])
    mean_stds = pd.concat([mean_stds, stds.mean(axis=1).rename(experiment).to_frame().T])
    mean_durations = pd.concat([mean_durations, durations.mean(axis=1).rename(experiment).to_frame().T])


table_styles = [{
    'selector': 'caption',
    'props': [
        ('font-size', '16px')
    ]
}]

export_table_columns = ['POS_X (cm)', 'POS_Y (cm)', 'POS_Z (cm)', 'LINACC_X (mg)', 'LINCACC_Y (mg)', 'LINACC_Z (mg)', 'GYRO_X (dps)', 'GYRO_Y (dps)', 'GYRO_Z (dps)', 'Heading (deg)', 'Roll (deg)', 'Pitch (deg)']
mean_means.columns = export_table_columns
std_means.columns = export_table_columns
mean_stds.columns = export_table_columns

# Output the Summary Stats Table #

# Plot the means
cont_select = True
regions = {}

while cont_select:
    bg_options = {
        "ILS": {"path": "ISL HQ Screenshot-rotated.png", "multiplier": 6.3, },
    }
    bg_path = bg_options["ILS"]["path"]
    bg_multiplier = bg_options["ILS"]["multiplier"]
    img = Image.open(bg_path).convert("L")
    img = np.asarray(img)
    ax = plt.subplot()
    ax.imshow(img, extent=[0,img.shape[1]*bg_multiplier,0,img.shape[0]*bg_multiplier], cmap='gray', vmin=0, vmax=255)

    for mean_row, std_mean in zip(mean_means.iterrows(), std_means.iterrows()):
        ax.scatter(mean_row[1]['POS_X (cm)']*10, mean_row[1]['POS_Y (cm)']*10, color='red', s=20, marker="o")
        ax.text(mean_row[1]['POS_X (cm)']*10, mean_row[1]['POS_Y (cm)']*10, mean_row[1].name, size=20, color='red', rotation=30)
        ax.add_patch(patches.Ellipse((mean_row[1]['POS_X (cm)']*10, mean_row[1]['POS_Y (cm)']*10), std_mean[1]['POS_X (cm)']*10, std_mean[1]['POS_Y (cm)']*10, fill=False, color='red', alpha=0.5))

    ax.set_xlabel("POS_X (mm)")
    ax.set_ylabel("POS_Y (mm)")
    ax.set_title("Select a Region")

    for k,v in regions.items():
        v = np.array(v)
        ax.fill(v[:,0], v[:,1], alpha=0.2)
        ax.text(v[0, 0], v[0, 1], k)

    max_xlim = ax.get_xlim() # get current x_limits to set max zoom out
    max_ylim = ax.get_ylim() # get current y_limits to set max zoom out
    f = utils.zoom_factory(ax, max_xlim, max_ylim, base_scale=1.)
    pts = [list(item) for item in plt.ginput(-1, timeout=-1)] # convert to list for json format
    plt.close()
    region_name = input("What is this region?: ")
    regions[region_name] = pts
    cont_select = True if input("Continue? (y/n): ").lower() == "y" else False


output_dir = Path().joinpath('outputs', 'REGIONS')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir.joinpath(str(datetime.now()) + '.json'), 'w') as f:
    f.write(json.dumps(regions))

print(regions)