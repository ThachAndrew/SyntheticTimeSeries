import numpy as np
from sklearn.linear_model import LinearRegression

def fit_AR_model(series, start, end, lags):
    series = series[start:end]

    train_lagged_values = []
    train_series = []

    for ts_idx in range(len(series) - lags[-1]):
        ts = ts_idx + lags[-1]
        train_series += [series[ts]]

        ts_train_lagged_values = []
        for lag in lags:
            ts_train_lagged_values += [series[ts - lag]]

        train_lagged_values += [ts_train_lagged_values]

    AR_model = LinearRegression().fit(train_lagged_values, train_series)

    print(AR_model.coef_)

def generate_aggregate_series(series, start_index, end_index, window_size):
    agg_series = np.array([])

    if (end_index - start_index + 1) % window_size != 0:
        print("Series length not divisible by window length, quitting.")
        exit(1)

    num_windows = int((end_index - start_index + 1) / window_size)
    for window_idx in range(num_windows):
        window_start = window_idx * window_size + start_index
        window_end = (window_idx + 1) * window_size + start_index

        window_mean = np.mean(series[window_start:window_end])
        agg_series = np.append(agg_series, window_mean)

    return agg_series
