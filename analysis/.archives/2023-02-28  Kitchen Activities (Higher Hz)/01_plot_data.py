# Fix for importing utils
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from analysis.utils import utils

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}

matplotlib.rc('font', **font)


labels_dir = "03_Labels"
data_dir = "02_Pozyx_Positioning_Data"
NUM_NORM_POINTS = 300
ANCHOR_CONFIG = ["9H"]
SUBJECT_INITIALS = "HG"
TYPE = "ASSEMBLESANDWICH"
TAG_ID =  ["0x683f"]
TRIAL = 2

for anchor in ANCHOR_CONFIG:
    for tagId in TAG_ID:
        processed_trials = pd.DataFrame()
        ANCHORS = anchor


        label_fn = f"{TYPE}_{SUBJECT_INITIALS}_A{ANCHORS}_{TRIAL}.txt"
        data_fn = f"{TYPE}_{SUBJECT_INITIALS}_A{ANCHORS}_{TRIAL}.csv"

        label_fp = Path('.').joinpath(labels_dir, TYPE, label_fn)
        data_fp = Path('.').joinpath(data_dir, TYPE, data_fn)

        labels = utils.extract_time_labels(label_fp)

        data = pd.read_csv(data_fp)
        data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'TagId']
        data = data.set_index('Timestamp')


        data = data[data["TagId"] == tagId]
        data = data.drop("TagId", axis=1)

        cleaned_data = (data
                        .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch']]
                        .rolling(20).mean() # 20 point moving average
                        )

        ######################################################################
        # PLOTTING
        ######################################################################

        ax = sns.histplot(data=np.diff(cleaned_data.index), binwidth=0.001)
        ax.set_title('dt Plot of Data for ' + label_fn)
        ax.set_xlabel('dt (s)')
        ax.set_ylabel('Occurences')

        ax = utils.subplot_pozyx_data_with_timings(cleaned_data, ['POS_X','POS_Y', 'POS_Z'], labels)
        # ax.set_title(label_fn + " POS")
        ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['LINACC_X','LINACC_Y','LINACC_Z'], labels, ylim=(-1000, 1000), ylabel="Acceleration (mg)")
        ax.set_title(label_fn + " ACC")
        ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['GYRO_X','GYRO_Y','GYRO_Z'], labels, ylim=(-500, 500), ylabel="Angular Velocity (dps)")
        ax.set_title(label_fn + " GYRO")

        ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['Heading', 'Roll', 'Pitch'], labels, ylim=(-500, 500), ylabel="Euler Angle (deg)")
        ax.set_title(label_fn + " Orientation")

    plt.show()