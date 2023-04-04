import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)


EXP_TYPES = ['ASSEMBLESANDWICH', 'GETPLATE', 'OPENFREEZER', 'OPENFRIDGE', 'SLICETOMATO', 'WASHHANDS']
tagId = "0x683f"
ALL_DATA_FILES = [file for exp in EXP_TYPES for file in Path(__file__).resolve().parent.joinpath('02_Pozyx_Positioning_Data', exp).glob('*.csv')]
ALL_LABEL_FILES = [file.parent.joinpath('03_Labels', file.name.replace('.csv', '.txt')) for file in ALL_DATA_FILES]

all_dt = np.array([])
for data_path, label_path in zip(ALL_DATA_FILES, ALL_LABEL_FILES):
    data = pd.read_csv(data_path)

    data.columns = ['Timestamp', 'POS_X', 'POS_Y', 'POS_Z', 'Heading', 'Roll', 'Pitch', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'LINACC_X', 'LINACC_Y', 'LINACC_Z', 'TagId']
    data = data.set_index('Timestamp')


    data = data[data["TagId"] == tagId]
    data = data.drop("TagId", axis=1)

    all_dt = np.concatenate([all_dt, np.diff(data.index)])

ax = sns.histplot(data=all_dt, binwidth=0.001)
ax.set_title(f"dt of all Data. Mean = {np.around(np.mean(all_dt), decimals=5)}s, std = {np.around(np.std(all_dt), decimals=5)}s")
ax.set_xlabel("dt (s)")
ax.set_ylabel("Occurrences")

plt.show()