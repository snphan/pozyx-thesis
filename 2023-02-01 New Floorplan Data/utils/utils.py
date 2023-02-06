import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from scipy import interpolate
from PIL import Image

#MARK: PREPROCESSING
def extract_time_labels(fp):
    """
    Expects the file to have lines that read: 
    LABEL_NAME: TIMESTAMP

    Eg. Go to fridge: 16900000.104

    Returns
    -------
    List[{'Timestamp': float, 'Label': str}]
    """
    with open(fp) as f:
        labels = f.readlines()

    label_keys = ["Timestamp", "Label"]
    labels = [label.replace("\n","").split(": ")[::-1] for label in labels]
    labels =  [{label_keys[0]: label[0], label_keys[1]:label[1]} for label in labels]
    return labels

def copy_df_format(df):
    """
    Returns an empty df with the corresponding df
    """
    copy_df = pd.DataFrame(columns=df.columns)
    copy_df.index.name = df.index.name
    return copy_df

#MARK: CLEANING DATA

def remove_periods(df, periods):
    """Deletes the selected periods"""
    for period in periods:
        mask = (df.index > period[0]) & (df.index < period[1])
        df = df.drop(df.index[mask])
    return df

def df_iterp1d(df, num_points):

    interp_df = pd.DataFrame()
    interp_df.index.name = df.index.name

    x = df.index.values
    xnew = np.linspace(x[0], x[-1], num=num_points)

    for col in df.columns:
        col_data = df[col].values
        f = interpolate.interp1d(x, col_data)
        new_col_data = f(xnew)
        new_df = pd.DataFrame({'Timestamp': xnew, col: new_col_data}).set_index('Timestamp')
        interp_df = pd.concat([interp_df, new_df], axis=1)

    return interp_df

def interp1d_periods(df, periods, num_points):
    for period in periods:
        mask = (df.index > period[0]) & (df.index < period[1])
        interp_df = df_iterp1d(df[mask], num_points)
        df = df.drop(df.index[mask])
        df = pd.concat([df, interp_df])
    return df

#MARK: VISUALIZATION

def plot_pozyx_data_with_timings(data: pd.DataFrame, columns: list[str], labels: list[dict], title="Data with Timings", ylim=(-1000, 15000)):
    """
    Args
    ----

    labels: list of dictionary {'Timestamp': <TIMESTAMP>, 'Label': <LABEL_NAME>} 
    """
    ax = data.loc[:, columns].plot(figsize=(20,10))
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.set_ylabel("Position (mm)")
    for label in labels:
        if label["Label"] != "10 sec elapsed":
            ax.axvline(float(label['Timestamp']), color="red")
            ax.text(float(label['Timestamp'])+2,0,label["Label"], rotation=90, color="red", size="large")
        else:
            ax.axvline(float(label['Timestamp']), color="black")
    return ax


def plot_interactive_pozyx_data(data, title, xlim=(0,12000), ylim=(0,12000), ground_truth=[], bg_path="", bg_multiplier=1):
    @widgets.interact(to_index=(0, 2000),window=(1,100))
    def f(to_index=0,window=1):
        fig = plt.figure(figsize=(10,10))
        data_clean = data.rolling(window).mean()
        plt.plot(data_clean.iloc[:to_index, 0], data_clean.iloc[:to_index, 1])
        plt.xlabel("X(mm)")
        plt.ylabel("Y(mm)")
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(np.arange(xlim[0], xlim[1], 500), rotation=20)
        if bg_path:
            img = Image.open(bg_path).convert("L")
            img = np.asarray(img)
            plt.imshow(img, extent=[0,1803*bg_multiplier,0,1683*bg_multiplier], cmap='gray', vmin=0, vmax=255)
        for point in ground_truth:
            plt.scatter(point['x'], point['y'], color='red', s=2, marker="x")
            plt.text(point['x'] + 50, point['y'], point['label'], color='red')