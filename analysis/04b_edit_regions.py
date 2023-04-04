"""
Script to change any of the regions file created in 04_outputs/REGIONS

Currently only works for the ILS
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
from datetime import datetime
import json

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
font = {'family' : 'Ubuntu',
        'size'   : 22}
matplotlib.rc('font', **font)


def plot_ILS(regions):

    bg_options = {
        "ILS": {"path": "ISL HQ Screenshot-rotated.png", "multiplier": 6.3, },
    }
    bg_path = bg_options["ILS"]["path"]
    bg_multiplier = bg_options["ILS"]["multiplier"]
    img = Image.open(bg_path).convert("L")
    img = np.asarray(img)
    ax = plt.subplot()
    ax.imshow(img, extent=[0,img.shape[1]*bg_multiplier,0,img.shape[0]*bg_multiplier], cmap='gray', vmin=0, vmax=255)

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

    return ax


output_dir = Path().joinpath('04_outputs', 'REGIONS')
output_dir.mkdir(parents=True, exist_ok=True)
existing_files = list(output_dir.glob("*.json"))


for ind, fp in enumerate(existing_files):
    print(f"({ind}) {fp}")

edit_file_ind = int(input(f"Select the file you want to edit 0-{len(existing_files)-1}: "))

if edit_file_ind >= len(existing_files) or edit_file_ind < 0: quit()

regions = {}
with open(existing_files[edit_file_ind], 'r') as f:
    regions = json.load(f)

while True:
    ax = plot_ILS(regions)
    print("-"*50)
    print("Existing Regions: ")
    print("-"*50)
    for ind, region in enumerate(regions):
        print(f"{region}")
    print("-"*50)
    edit_region = input("Select a region to edit or type name of new region, or (done): ").lower()

    if edit_region == "done": break

    pts = [list(item) for item in plt.ginput(-1, timeout=-1)] # convert to list for json format
    plt.close()
    regions[edit_region] = pts

with open(existing_files[edit_file_ind], 'w') as f:
    f.write(json.dumps(regions))

print(regions)