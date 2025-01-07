"""
This file is to be used with the RAW_SPLIT_DATA created from 00b_split_wild_data.py
creates an input for a table in LATEX.

Example command

python 01c_action_stats.py --root-dir ../outputs/RAW_SPLIT_DATA
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import OrderedDict

parser = argparse.ArgumentParser("Counts the number of actions")
parser.add_argument(
    "--root-dir", required=True, help="The root dir where all of the actions are."
)
parser.add_argument(
    "--disability",
    nargs="+",
    required=False,
    help="Space separated list of the disability. AB, OE, KG, EG",
)
args = parser.parse_args()

root_dir = Path(args.root_dir)

counts = []

for action_path in root_dir.glob("*"):
    action = action_path.name
    # Get the average time elapsed for each action
    times = []
    count = 0
    for repetition_path in action_path.glob("*.csv"):
        try:
            disability = repetition_path.name.split("-")[0]
            if args.disability and disability not in args.disability:
                continue
            repetition_df = pd.read_csv(repetition_path, index_col=0)
            duration = repetition_df.index[-1] - repetition_df.index[0]
            times.append(duration)
            count += 1
        except Exception as e:
            raise Exception(f"{e} at file {repetition_path.name}")
    counts.append(
        {
            "action": action,
            "count": count,
            "avg_time": np.mean(times),
            "std_time": np.std(times),
        }
    )

counts_df = pd.DataFrame(counts).sort_values("avg_time", ascending=False)

for row in counts_df.iterrows():
    values = row[1]
    print(
        f"{values['action']:<20} & {values['count']:<5} & {round(values['avg_time'], 2):<6} & {round(values['std_time'], 2):<6} \\\\"
    )

print("\n\n")
print(f"Disabilities: {'ALL' if not args.disability else args.disability}")
print(f"Number of classes {len(counts)}")
print(f"Number of fine-grained-actions {np.sum(counts_df['count'])}")
