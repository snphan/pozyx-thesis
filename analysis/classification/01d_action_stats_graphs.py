"""
Create the plots for the average durations by disability.

Sample Command

python 01d_action_stats_graphs.py --root-dir ../outputs/RAW_SPLIT_DATA --actions PEEL CUT STIR SCOOP --grouping Continuous
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Styling
font = {"family": "Times"}
matplotlib.rc("font", **font)
sns.set_style("white")

parser = argparse.ArgumentParser("Counts the number of actions")
parser.add_argument(
    "--root-dir", required=True, help="The root dir where all of the actions are."
)
parser.add_argument("--grouping", required=False, help="The grouping of actions")
parser.add_argument(
    "--disability",
    required=False,
    help="space separated list of disabilities. OE AB KG EG. If none then all",
    nargs="+",
)
parser.add_argument(
    "--actions",
    required=False,
    help="space separated list of actions.",
    nargs="+",
)
args = parser.parse_args()

root_dir = Path(args.root_dir)
actions = []

for action_path in root_dir.glob("*"):
    action = action_path.name
    for repetition_path in action_path.glob("*.csv"):
        repetition_df = pd.read_csv(repetition_path, index_col=0)
        duration = repetition_df.index[-1] - repetition_df.index[0]
        disability = repetition_path.name.split("-")[0]
        actions.append(
            {"action": action, "duration": duration, "disability": disability}
        )

actions_df = pd.DataFrame(actions)

# Make the graphs
average_durations = (
    actions_df.groupby(["action", "disability"])
    .mean()
    .unstack(level=0)
    .transpose()
    .reset_index(level=0, drop=True)
)

# filter
average_durations = average_durations.loc[
    average_durations.index.values if not args.actions else args.actions,
    average_durations.columns.values if not args.disability else args.disability,
]
print(average_durations.head())
axs = average_durations.plot(kind="barh")
axs.set_title(
    f"Average duration of {'All' if not args.grouping else args.grouping} actions by disability"
)
axs.set_xlabel("Duration (s)")
plt.tight_layout()
plt.show()
