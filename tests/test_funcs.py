import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter


def calc_moving_average(yp, window_size=3):
    # from scipy import signal
    # from scipy.signal import savgol_filter
    i = 0
    moving_averages = []  # Initialize an empty list to store moving averages
    while i < len(yp) - window_size + 1:
        window = yp[i: i + window_size]  # Store elements from i to i+window_size in list to get the current window
        window_average = np.median(window)  # round(sum(window) / window_size, 2)
        moving_averages.append(window_average)  # Store the median of current window in moving-median list
        i += 1  # Shift window to right by one position
    last_value = moving_averages[-1]
    values_missing = len(yp) - len(moving_averages)
    for i in range(values_missing):
        moving_averages.append(last_value)
    return moving_averages
