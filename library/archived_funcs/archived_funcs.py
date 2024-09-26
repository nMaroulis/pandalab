from numpy import median as np_median, mean as np_mean, min as np_min, max as np_max


def calc_moving_average(yp, window_size=3, filter_type='Mean'):
    # from scipy import signal
    # from scipy.signal import savgol_filter
    i = 0
    moving_averages = []  # Initialize an empty list to store moving averages
    # Loop through the array to consider - every window of size 3
    while i < len(yp) - window_size + 1:
        window = yp[i: i + window_size]  # Store elements from i to i+window_size in list to get the current window
        # Calculate the Median of current window
        if filter_type == 'Median':
            window_average = np_median(window)  # round(sum(window) / window_size, 2)
        elif filter_type == 'Min':
            window_average = np_min(window)  # round(sum(window) / window_size, 2)
        elif filter_type == 'Max':
            window_average = np_max(window)  # round(sum(window) / window_size, 2)
        else:  # Mean
            window_average = np_mean(window)  # round(sum(window) / window_size, 2)

        moving_averages.append(window_average)  # Store the median of current window in moving-median list
        i += 1  # Shift window to right by one position
    last_value = moving_averages[-1]
    values_missing = len(yp) - len(moving_averages)
    for i in range(values_missing):
        moving_averages.append(last_value)

    return moving_averages

