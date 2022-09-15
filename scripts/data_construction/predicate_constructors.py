import copy

import numpy as np
from ar_forecast import ar_forecast, top_down_adjust_ar_forecast, fp_adjust_ar_forecast, cluster_ar_forecast_adjust

# Timestep lag predicate, gives the n-lagged timestep pairs of timesteps in [start + n, end] inclusive
def lag_n_predicate(n, start, end, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for time_step in range(start + n, end + 1):
        out_file_lines += str(time_step) + "\t" + str(time_step - n) + "\n"

    out_file_handle.write(out_file_lines)

def lags_predicate(lags, start, end, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for time_step in range(start + len(lags), end + 1):
        out_file_lines += str(time_step)

        for lag in lags:
            out_file_lines += "\t" + str(time_step - lag)

        out_file_lines += "\n"

    out_file_handle.write(out_file_lines)


# Write series observations (or omit the values to get targets) in range [start_index, end_index] inclusive
def series_predicate(series_list, series_ids, start_index, end_index, out_path, include_values=True):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_idx, series in enumerate(series_list):
        for time_step_idx in range(end_index - start_index + 1):
            time_step = time_step_idx + start_index

            out_file_lines += str(series_ids[series_idx]) + "\t" + str(time_step)

            if include_values:
                out_file_lines += "\t" + str(series[time_step])

            out_file_lines += "\n"

    out_file_handle.write(out_file_lines)

# Groups series together into equally-sized clusters and assigns them an ID.
# Returns a map from cluster ID to the series IDs.
def series_cluster_predicate(series_list, series_ids, cluster_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    cluster_series_map = dict()

    for idx, series_id in enumerate(series_ids):
        cluster = int(idx / cluster_size)
        out_file_lines += str(series_id) + "\t" + str(cluster) + "\n"
        if cluster not in cluster_series_map:
            cluster_series_map[cluster] = []
        cluster_series_map[cluster] += [series_id]

    cluster_agg_series_list = []

    for cluster_id in cluster_series_map:
        cluster_agg_series_list += [np.sum([series_list[series_id] for series_id in cluster_series_map[cluster_id]], axis=0)]

    out_file_handle.write(out_file_lines)

    return cluster_series_map, cluster_agg_series_list

# Cluster mean predicate. It's an open predicate used in the arithmetic rule relating a cluster's mean to the cluster oracle value.
def cluster_mean_predicate(cluster_series_map, start_timestep, end_timestep, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for cluster in cluster_series_map:
        for timestep in range(start_timestep, end_timestep + 1):
            out_file_lines += str(cluster) + "\t" + str(timestep) + "\n"

    out_file_handle.write(out_file_lines)

def sim_agg_forecast(cluster_series, t, h, z, forecast_variance_scale):
    agg_series = np.sum(cluster_series, axis=0)
    agg_var = np.var(agg_series[:z])

    sigma_sq = forecast_variance_scale * agg_var

    forecast_window_truth = agg_series[t:t+h]

    # Clip into a range where agg forecast values divided by the number of series aggregated is <= 1
    forecast = [np.clip(agg + np.random.normal(0, sigma_sq), 0, len(cluster_series)) for i, agg in enumerate(forecast_window_truth)]

    return forecast

def bias_ar_forecast_cluster(base_forecasts, agg_series):
    series_sum = np.sum(base_forecasts, axis=0)
    cluster_series = copy.deepcopy(base_forecasts)

    for timestep_idx, agg_forecast_val in enumerate(agg_series):
        adj_term = (agg_forecast_val - series_sum[timestep_idx])/len(cluster_series)
        for series_idx in range(len(cluster_series)):
            cluster_series[series_idx][timestep_idx] += adj_term

    return cluster_series


# Cluster oracle predicate, contains the true aggregate series values with respect to each cluster.
# Option to add noise.
# Values are for timesteps in [start_index, end_index] inclusive.
def cluster_oracle_predicate(series_list, cluster_series_map, noise_sigma, init_segment_size, start_index, end_index, window_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    forecasts = np.empty((len(cluster_series_map.keys()), window_size * int((end_index - start_index + 1)/window_size)))

    for cluster_id in cluster_series_map:
        #agg_series = np.sum([series_list[idx][start_index:end_index+1] for idx in cluster_series_map[cluster_id]], axis=0)
        #agg_series = agg_series + np.random.normal(scale=noise_sigma * np.std(agg_series), size=len(agg_series))

        forecast_start = start_index

        cluster_forecast = []

        # TODO: return forecasts for use in baseline
        while forecast_start < end_index:
            forecast = sim_agg_forecast(np.array([series_list[idx] for idx in cluster_series_map[cluster_id]]), forecast_start, window_size, init_segment_size, noise_sigma)

            for i in range(len(forecast)):
                out_file_lines += str(cluster_id) + "\t" + str(forecast_start + i) + "\t" + str(forecast[i] / len(cluster_series_map[cluster_id])) + "\n"

            cluster_forecast += forecast
            forecast_start += window_size

        forecasts[cluster_id] = np.array(cluster_forecast)

    out_file_handle.write(out_file_lines)

    return forecasts


def ar_baseline_predicate(series_list, coefs_and_biases, series_ids, oracle_series_list,
                            start_index, end_index, n, out_path, adj_out_path):
    out_file_handle = open(out_path, "w")
    adj_out_file_handle = open(adj_out_path, "w")
    out_file_lines = ""
    adj_out_file_lines = ""

    base_ar_forecasts = [[] for series in series_list]

    for series_idx, series in enumerate(series_list):
        series = series[start_index:end_index+1]

        coefs, bias = coefs_and_biases[series_idx]
        forecast = np.clip(ar_forecast(series, coefs, bias, n), 0, 1)
        base_ar_forecasts[series_idx] = forecast

        adj_forecast = top_down_adjust_ar_forecast(forecast, oracle_series_list[series_idx])

        for time_step_idx in range(n):
            out_file_lines += str(series_ids[series_idx]) + "\t" + str(end_index + time_step_idx + 1) + "\t" + str(round(forecast[time_step_idx], 6)) + "\n"
            adj_out_file_lines += str(series_ids[series_idx]) + "\t" + str(end_index + time_step_idx + 1) + "\t" + str(
                round(adj_forecast[time_step_idx], 6)) + "\n"

    out_file_handle.write(out_file_lines)
    adj_out_file_handle.write(adj_out_file_lines)

    return base_ar_forecasts

def cluster_equal_bias_ar_forecasts_predicate(series_list, coefs_and_biases, cluster_oracle_series_list, cluster_size, start_idx, end_idx, init_size, n, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    n_clusters = int(len(series_list) / cluster_size)

    for cluster_idx in range(n_clusters):
        base_forecasts = [[] for x in range(cluster_size)]

        for series_cluster_idx in range(cluster_size):
            series_idx = (cluster_idx * cluster_size) + series_cluster_idx

            coefs, bias = coefs_and_biases[series_idx]
            forecast = np.clip(ar_forecast(series_list[series_idx][start_idx:end_idx + 1], coefs, bias, n), 0, 1)
            base_forecasts[series_cluster_idx] = forecast

        coherent_forecasts = bias_ar_forecast_cluster(base_forecasts, cluster_oracle_series_list[cluster_idx][end_idx - init_size + 1:end_idx - init_size + n + 1])

        for time_step_idx in range(n):
            for series_cluster_idx in range(cluster_size):
                out_file_lines += str((cluster_idx * cluster_size) + series_cluster_idx) + "\t" + str(end_idx + time_step_idx + 1) + "\t" + str(coherent_forecasts[series_cluster_idx][time_step_idx]) + "\n"

    out_file_handle.write(out_file_lines)

def fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map, cluster_agg_series_list, cluster_size, init_size, start, end, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for cluster_id in cluster_series_map:
        base_forecasts = [base_forecast_series_list[series_id] for series_id in cluster_series_map[cluster_id]]
        agg_series = cluster_agg_series_list[cluster_id]

        coherent_forecasts = fp_adjust_ar_forecast(base_forecasts, agg_series[start - init_size:end + 1 - init_size])

        for series_idx in range(len(cluster_series_map[cluster_id])):
            for timestep in range(start, end + 1):
                out_file_lines += str(cluster_series_map[cluster_id][series_idx]) + "\t" + str(timestep) + "\t" + str(coherent_forecasts[series_idx][timestep - start]) + "\n"

    out_file_handle.write(out_file_lines)

# Series blocking for the AR arithmetic rules.
def series_block_predicate(series_ids, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_id in series_ids:
        out_file_lines += str(series_id) + "\t" + str(series_id) + "\n"

    out_file_handle.write(out_file_lines)

# Agg series predicate. It's an open predicate used in an arithmetic rule relating the mean of the current forecast window to the oracle value.
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

# TODO @Alex: Predict based on historical mean, not true mean of current forecast window.
def naive_prediction_predicate(agg_series_list, series_ids, window_start_idx, window_end_idx, window_size, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_idx, series in enumerate(agg_series_list):
        for window_idx in range(window_start_idx, window_end_idx):
            for ts_idx in range(window_size):
                out_file_lines += str(series_ids[series_idx]) + "\t" + str(window_idx * window_size + ts_idx) + "\t" + str(series[window_idx]) + "\n"

    out_file_handle.write(out_file_lines)

# IsInWindow predicate. Relates timesteps to the forecast windows that contain them.
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

# Compute aggregate series of an individual base-level series (a series of means of equally-sized, adjacent segments)
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

#  Series mean predicate for mean prior
def series_mean_predicate(series_list, series_ids, end_time_step, out_path):
    out_file_lines = ""
    out_file_handle = open(out_path, "w")

    for idx, series in enumerate(series_list):
        trunc_series = series[:end_time_step]

        out_file_lines += str(series_ids[idx]) + "\t" + str(np.mean(trunc_series)) + "\n"

    out_file_handle.write(out_file_lines)


# Oracle series predicate. Equivalent to the true aggregate series value but optionally has noise added to it.
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
