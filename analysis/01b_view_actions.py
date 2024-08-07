import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser("Plots and views the Action Data")
parser.add_argument("--root-dir", required=True, help="The root dir where all of the actions are.")
parser.add_argument("--actions", required=True, help="Which actions to view", nargs="+")
parser.add_argument("--height", default=3, type=int, help="height of grid")
parser.add_argument("--width", default=3, type=int, help="width of grid")
args = parser.parse_args()

root_dir = Path(args.root_dir)

plots_per_window = args.height * args.width

for action in args.actions:
    action_paths = list(root_dir.joinpath(action).glob("*.csv"))
    for ind in range(0, len(action_paths), plots_per_window):
        actions_paths_to_plot = action_paths[ind:ind+plots_per_window]
        fig, axs = plt.subplots(args.height, args.width, figsize=(10, 10))
        
        for i in range(args.height):
            for j in range(args.width):
                pd.read_csv(actions_paths_to_plot[(i*args.width) + j], index_col=0)[["POS_X", "POS_Y", "POS_Z"]].plot(ax=axs[i, j], legend=False)
                axs[i,j].set_title(actions_paths_to_plot[(i*args.width) + j].name)

        fig.suptitle(f"Plot of {action} actions")
        fig.tight_layout()
        plt.show()

        


