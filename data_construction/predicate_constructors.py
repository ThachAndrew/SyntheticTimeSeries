import numpy as np

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

def series_block_predicate(series_ids, out_path):
    out_file_handle = open(out_path, "w")
    out_file_lines = ""

    for series_id in series_ids:
        out_file_lines += str(series_id) + "\t" + str(series_id) + "\n"

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
