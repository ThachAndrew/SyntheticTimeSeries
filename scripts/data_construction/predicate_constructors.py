import numpy as np
from ar_forecast import ar_forecast, top_down_adjust_ar_forecast

def lag_n_predicate(n, start, end, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for time_step in range(start + n, end + 1):
        out_file_lines += str(time_step) + "\t" + str(time_step - n) + "\n"

    out_file_handle.write(out_file_lines)

def series_predicate(series_list, series_ids, start_index, end_index, out_path, include_values=True):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_idx, series in enumerate(series_list):
        for time_step_idx in range(end_index - start_index + 1):
            time_step = time_step_idx + start_index

            out_file_lines += str(series_ids[series_idx]) + "\t" + str(time_step)

            if include_values:
                out_file_lines += "\t" + str(series[time_step_idx])

            out_file_lines += "\n"

    out_file_handle.write(out_file_lines)

def series_cluster_predicate(series_ids, cluster_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    cluster_series_map = dict()

    for idx, series_id in enumerate(series_ids):
        cluster = int(idx / cluster_size)
        out_file_lines += str(series_id) + "\t" + str(cluster) + "\n"
        if cluster not in cluster_series_map:
            cluster_series_map[cluster] = []
        cluster_series_map[cluster] += [series_id]

    out_file_handle.write(out_file_lines)

    return cluster_series_map

def cluster_mean_predicate(cluster_series_map, start_timestep, end_timestep, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for cluster in cluster_series_map:
        for timestep in range(start_timestep, end_timestep + 1):
            out_file_lines += str(cluster) + "\t" + str(timestep) + "\n"

    out_file_handle.write(out_file_lines)

def cluster_oracle_predicate(series_list, cluster_series_map, noise_sigma, start_index, end_index, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for cluster_id in cluster_series_map:
        agg_series = np.sum([series_list[idx][start_index:end_index+1] for idx in cluster_series_map[cluster_id]], axis=0)
        agg_series = agg_series + np.random.normal(scale=noise_sigma, size=len(agg_series))

        for t in range(len(agg_series)):
            out_file_lines += str(cluster_id) + "\t" + str(t + start_index) + "\t" + str(agg_series[t] / len(cluster_series_map[cluster_id])) + "\n"

    out_file_handle.write(out_file_lines)

def ar_baseline_predicate(series_list, coefs_and_biases, series_ids, oracle_series_list, start_index, end_index, n, out_path, adj_out_path):
    out_file_handle = open(out_path, "w")
    adj_out_file_handle = open(adj_out_path, "w")
    out_file_lines = ""
    adj_out_file_lines = ""

    for series_idx, series in enumerate(series_list):
        series = series[start_index:end_index+1]

        coefs, bias = coefs_and_biases[series_idx]
        forecast = np.clip(ar_forecast(series, coefs, bias, n), 0, 1)
        adj_forecast = top_down_adjust_ar_forecast(forecast, oracle_series_list[series_idx])

        for time_step_idx in range(n):
            out_file_lines += str(series_ids[series_idx]) + "\t" + str(end_index + time_step_idx + 1) + "\t" + str(round(forecast[time_step_idx], 6)) + "\n"
            adj_out_file_lines += str(series_ids[series_idx]) + "\t" + str(end_index + time_step_idx + 1) + "\t" + str(
                round(adj_forecast[time_step_idx], 6)) + "\n"

    out_file_handle.write(out_file_lines)
    adj_out_file_handle.write(adj_out_file_lines)

def series_block_predicate(series_ids, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_id in series_ids:
        out_file_lines += str(series_id) + "\t" + str(series_id) + "\n"

    out_file_handle.write(out_file_lines)

def agg_series_predicate(series_ids, start_index, end_index, window_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    if (end_index - start_index + 1) % window_size != 0:
        print("Series length not divisible by window length, quitting.")
        exit(1)

    num_windows = int((end_index - start_index + 1) / window_size)

    for series_id in series_ids:
        for window_idx in range(num_windows):
            out_file_lines += str(series_id) + "\t" + str(window_idx) + "\n"

    out_file_handle.write(out_file_lines)

def naive_prediction_predicate(agg_series_list, series_ids, window_start_idx, window_end_idx, window_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_idx, series in enumerate(agg_series_list):
        for window_idx in range(window_start_idx, window_end_idx):
            for ts_idx in range(window_size):
                out_file_lines += str(series_ids[series_idx]) + "\t" + str(window_idx * window_size + ts_idx) + "\t" + str(series[window_idx]) + "\n"

    out_file_handle.write(out_file_lines)

# The end index is one after the last time step we're including in the aggregation windows
def time_in_aggregate_window_predicate(start_index, end_index, window_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    if (end_index - start_index + 1) % window_size != 0:
        print("Series length not divisible by window length, quitting.")
        exit(1)

    num_windows = int((end_index - start_index + 1) / window_size)

    for window_idx in range(num_windows):
        window_start = window_idx * window_size + start_index
        window_end = (window_idx + 1) * window_size + start_index

        times_in_window = np.arange(window_start, window_end)

        for time_step in times_in_window:
            out_file_lines += str(time_step) + "\t" + str(window_idx) + "\n"

    out_file_handle.write(out_file_lines)

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

# Takes aggregate series as in put and adds noise to them
def oracle_series_predicate(series_list, series_ids, start_index, end_index, noise_sigma, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series in series_list:
        for x in range(len(series)):
            std = np.std(series[x])
            series[x] += np.random.normal(scale=noise_sigma * std)

            # Might be necessary to clip depending on the amount of noise added.
            series[x] = np.clip(series[x], 0, 1)

    for series_idx, series in enumerate(series_list):
        for time_step_idx in range(end_index - start_index + 1):
            time_step = time_step_idx + start_index
            out_file_lines += str(series_ids[series_idx]) + "\t" + str(time_step) + "\t" + str(series[time_step_idx]) + "\n"

    out_file_handle.write(out_file_lines)
