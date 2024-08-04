"""
Run as python 00_fix_labels.py <EXP_TYPE> <SUBJECT_INITIALS> <TRIAL_NUM>

"""

# Fix for importing utils
import sys, os
# SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
# sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from utils import utils

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
# font = {'family' : 'Ubuntu',
#         'size'   : 22}

# matplotlib.rc('font', **font)


labels_dir = "03_Labels"
data_dir = "02_Pozyx_Positioning_Data"
NUM_NORM_POINTS = 300
ANCHOR_CONFIG = ["9H"]
try: 
    SUBJECT_INITIALS = [sys.argv[2]]
    EXP_TYPES = [sys.argv[1]]
    TAG_ID =  ["0x682f"]
    TRIAL_NUM = int(sys.argv[3])
except IndexError as e:
    print("Help: Run as python 00_fix_labels.py <EXP_TYPE> <SUBJECT_INITIALS> <TRIAL_NUM>")
    sys.exit(0)
skip_all = False

for anchor in ANCHOR_CONFIG:
    for TYPE in EXP_TYPES:
        for initial in SUBJECT_INITIALS:
            for tagId in TAG_ID:
                processed_trials = pd.DataFrame()
                ANCHORS = anchor

                label_fn = f"{TYPE}_{initial}_A{ANCHORS}_{TRIAL_NUM}.txt"
                data_fn = f"{TYPE}_{initial}_A{ANCHORS}_{TRIAL_NUM}.csv"
                label_fp = Path('.').joinpath('data', labels_dir, TYPE, label_fn)
                data_fp = Path('.').joinpath('data', data_dir, TYPE, data_fn)

                output_dir = Path().joinpath('outputs', 'NEW_LABELS', TYPE)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_fp = output_dir.joinpath(label_fp.name)

                output_fp.parent.mkdir(exist_ok=True, parents=True)

                add_labels = input("Add Labels (y/i/n)? ").lower()
                    
                if add_labels == "n":
                    if output_fp.exists(): 
                        if skip_all: continue
                        selection = input(f"Overwrite {output_fp.name}? (y/n/skip_all) ").lower()
                        if selection == "skip_all": 
                            skip_all = True
                            continue
                        if selection != "y":
                            continue

                    labels = utils.extract_time_labels(label_fp)

                    data = pd.read_csv(data_fp)
                    data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'Pressure', 'TagId']
                    data = data.set_index('Timestamp')


                    data = data[data["TagId"] == tagId]
                    data = data.drop("TagId", axis=1)

                    ######################################################################
                    #  CLEANING DATA
                    ######################################################################

                    cleaned_data = (data
                                    .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch']]
                                    .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                                    .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], 20)
                                    )

                    ######################################################################
                    # PLOTTING
                    ######################################################################

                    ax = sns.histplot(data=np.diff(cleaned_data.index), binwidth=0.001)
                    ax.set_title('dt Plot of Data for ' + label_fn)
                    ax.set_xlabel('dt (s)')
                    ax.set_ylabel('Occurences')

                    # Plot the position in cm
                    ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['POS_X','POS_Y', 'POS_Z'], labels, title=label_fp.name)
                    ax.set_title(f"{label_fp.name} Change n = {len(labels)}")
                    # ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['LINACC_X','LINACC_Y','LINACC_Z'], labels, ylim=(-1000, 1000), ylabel="Acceleration (mg)")
                    # ax.set_title(label_fn + " ACC")
                    # ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['GYRO_X','GYRO_Y','GYRO_Z'], labels, ylim=(-500, 500), ylabel="Angular Velocity (dps)")
                    # ax.set_title(label_fn + " GYRO")

                    # ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['Heading', 'Roll', 'Pitch'], labels, ylim=(-500, 500), ylabel="Euler Angle (deg)")
                    # ax.set_title(label_fn + " Orientation")
                
                    pts = np.array(plt.ginput(n=len(labels), timeout=-1))
                    output_items = []
                    for prev_label, new_timestamp in zip(labels, pts[:, 0]):
                        output_items.append(f"{prev_label['Label']}: {new_timestamp}")

                    with open(output_fp, 'w') as f:
                        f.write(('\n').join(output_items))

                    plt.close()
                else:
                    # We add some labels 
                    data = pd.read_csv(data_fp)
                    data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'Pressure', 'TagId']
                    data = data.set_index('Timestamp')


                    data = data[data["TagId"] == tagId]
                    data = data.drop("TagId", axis=1)

                    ######################################################################
                    #  CLEANING DATA
                    ######################################################################

                    cleaned_data = (data
                                    .loc[:, ['POS_X', 'POS_Y', 'POS_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'Heading', 'Roll', 'Pitch']]
                                    # .pipe(utils.handle_spikes, ['POS_Z', 'POS_Y'], [1500.0, 800.0]) 
                                    .pipe(utils.MAV_cols, ['POS_X', 'POS_Y', 'POS_Z'], 20)
                                    )
                    start_time = cleaned_data.index[0]
                    cleaned_data.index = cleaned_data.index - start_time

                    ######################################################################
                    # PLOTTING
                    ######################################################################
                
                    keep_adding = add_labels
                    labels = []
                    if label_fp.exists():                         
                        labels = utils.extract_time_labels(label_fp)
                        # Some preprocessing to align adding labels timestamp with video
                        labels = [{'Timestamp': str(float(item['Timestamp']) - start_time), 'Label': item['Label']} for item in labels]

                    # Options for actions
                    actions_fp = Path('.').joinpath('data', labels_dir, TYPE, 'actions.txt')
                    actions = []
                    begin_end = ["BEGIN", "END"]
                    if actions_fp.exists():
                        with open(actions_fp, 'r') as f:
                            actions = f.read().splitlines()

                    
                    while keep_adding in ["i", "y"]: 
                        prev_end = labels[-1]["Timestamp"] if labels else 0  # Keep track of the end time of each action for chained actions (OPEN, GRAB, CLOSE)
                        fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[5,2], figsize=(15,20))
                        ax = utils.plot_pozyx_data_with_timings(cleaned_data, ['POS_X','POS_Y', 'POS_Z'], labels, title=label_fp.name, ax=ax1)
                        # ax2 = utils.plot_pozyx_data_with_timings(cleaned_data, ['GYRO_X','GYRO_Y','GYRO_Z'], labels, ylim=(-500, 500), ylabel="Angular Velocity (dps)", ax=ax2)
                        # ax2.set_title(label_fn + " GYRO")
                        ax2 = utils.plot_pozyx_data_with_timings(cleaned_data, ['Heading','Roll','Pitch'], labels, ylim=(-200, 360), ylabel="Orientation (deg)", ax=ax2)
                        ax2.set_title(label_fn + " Orientation")

                        pts = []
                        if keep_adding == "y":
                            while len(pts) < 2:
                                pts = np.array(plt.ginput(n=2, timeout=-1, mouse_add=MouseButton.RIGHT, mouse_pop=MouseButton.MIDDLE))
                        if keep_adding == "i":
                            plt.draw()
                            plt.pause(0.001)
                            new_begin = input("BEGIN: ")
                            new_end = input("END: ")

                            try:
                                if len(new_end.split(":")) == 2:
                                    end_time = new_end.split(":")
                                    parsed_end = float(end_time[0]) * 60 + float(end_time[1])
                                else:
                                    parsed_end = float(new_end)

                                if new_begin == "p": parsed_begin = prev_end
                                elif len(new_begin.split(":")) == 2:
                                    begin_time = new_begin.split(":")
                                    parsed_begin = float(begin_time[0]) * 60 + float(begin_time[1])
                                else:
                                    parsed_begin = float(new_begin)
                            except ValueError:
                                print("invalid input")

                            pts = np.array([[parsed_begin], [float(parsed_end)]])

                        plt.close()

                        if actions:
                            label_created = False
                            while not label_created:
                                try:
                                    actions_options = "\n".join([f"({ind}) {action}" for ind, action in enumerate(actions)])
                                    action_ind = input(f"\nChoose an action or (C)ancel: \n{actions_options}\n\n")
                                    if action_ind.lower() == "c": break
                                    action_name = actions[int(action_ind)]
                                    confirm = input("Confirm Label: " + action_name + " (y/n)? ")
                                    if confirm == "y":
                                        label_created = True
                                except ValueError as e:
                                    print("Invalid Option")
                            if action_ind.lower() != "c":
                                labels.append({'Timestamp': pts[:, 0][0], 'Label': 'BEGIN_' + action_name})
                                labels.append({'Timestamp': pts[:, 0][1], 'Label': 'END_' + action_name})
                            else: print("\nCancelled label.\n")
                        else:
                            label_name = input("What is the label name or (C)ancel?: ")
                            if label_name.lower() != "c":
                                labels.append({'Timestamp': pts[:, 0][0], 'Label': label_name})
                            else: print("\nCancelled label.\n")


                        with open(output_fp, 'w') as f:
                            write_text = ('\n').join([f"{label['Label']}: {str(float(label['Timestamp']) + start_time)}" for label in labels])
                            f.write(write_text)

                        keep_adding = input("Keep Adding (y/i/n)? ").lower()


                