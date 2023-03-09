import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from analysis.utils import utils
import dataframe_image as dfi

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)


EXP_TYPES = ['ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS']
tagId = "0x683f"

mean_means = pd.DataFrame(columns=['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch'])
mean_stds = pd.DataFrame(columns=['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch'])
mean_durations = pd.DataFrame(columns=['Duration'])

for experiment in EXP_TYPES: 
    means = pd.DataFrame()
    stds = pd.DataFrame()
    durations = pd.DataFrame()
    for data_path in Path().joinpath('02_Pozyx_Positioning_Data', experiment).glob('*.csv'):
        data = pd.read_csv(data_path)
        label_fp = Path().joinpath('03_Labels', experiment, data_path.name.replace(".csv", ".txt"))

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
mean_stds.columns = export_table_columns
print(mean_durations)

dfi.export((mean_means)
    .style
    .set_precision(1)
    .set_caption("Mean of Mean for each Experiment")
    .set_table_styles(table_styles)
    .background_gradient(axis=0), 'means.png'
    )

dfi.export((mean_stds)
    .style
    .set_precision(1)
    .set_caption("Mean of Stds for each Experiment")
    .set_table_styles(table_styles)
    .background_gradient(axis=0), 'stds.png'
    )

dfi.export((mean_durations)
    .style
    .set_precision(1)
    .set_caption("Mean of Durations for each Experiment")
    .set_table_styles(table_styles)
    .background_gradient(axis=0), 'durations.png'
    )