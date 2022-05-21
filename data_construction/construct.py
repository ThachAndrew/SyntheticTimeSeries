import command_constructor
import predicate_constructors
import estimate_AR_coefs

import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.stattools import kpss, adfuller

NUM_SERIES =  10
INITIAL_SEGMENT_SIZE = 100
NUM_FORECAST_WINDOWS = 100
WINDOW_SIZE = 10

SERIES_LENGTH = INITIAL_SEGMENT_SIZE + (NUM_FORECAST_WINDOWS) * WINDOW_SIZE

SEED = 55555
REGEN_SEED_MAX = 10 ** 8

DATA_PATH = "data"

def generate_wn(n, sigma=1):
    return np.random.normal(0, sigma, size=n)

def generate_ar(n, phis, sigma=1):
    p = len(phis)
    adj_n = n + p
    e_series = generate_wn(adj_n, sigma)

    ar = [e_series[0]]
    for i in range(1, adj_n):
        visible_phis = phis[0:min(p, i)]
        visible_series = ar[i - min(p, i):i]

        reversed_phis = visible_phis[::-1]

        ar_t = e_series[i] + np.dot(reversed_phis, visible_series)

        ar.append(ar_t)

    ar = ar[p:]

    return ar

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(max(len(ar_phis), P * period))
    phis[0:len(ar_phis)] = ar_phis
    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

def generate_multiple_series(num_series, length, p, P, period, seed, enforce_stationarity=True):
    series = np.empty((num_series, length))
    coefs = np.empty((num_series, max(p, P * period)))

    np.random.seed(seed)

    for x in range(num_series):
        ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
        sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

        all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
        char_polynomial = np.append(1, all_coefs)[::-1]

        if enforce_stationarity:
            while np.min(np.abs(np.roots(char_polynomial))) < 1:
                ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
                sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

                all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
                char_polynomial = np.append(1, all_coefs)[::-1]

        coefs[x] = all_coefs
        series[x]  = np.array([generate_ar(length, all_coefs)])

    return series, coefs

# TODO @Alex: estimate AR coefficients, automate constructing the PSL model
def build_psl_data(generated_series, num_windows, experiment_dir, forecast_window_dirs):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Construct series ID constants
    series_ids = np.arange(len(generated_series))

    # Static predicates
    initial_window_dir = os.path.join(experiment_dir, forecast_window_dirs[0])

    if not os.path.exists(initial_window_dir):
        os.makedirs(initial_window_dir)

    predicate_constructors.lag_n_predicate(1, 1, SERIES_LENGTH - 1,
                                           os.path.join(initial_window_dir, "Lag1_obs.txt"))
    predicate_constructors.time_in_aggregate_window_predicate(0, SERIES_LENGTH - 1, WINDOW_SIZE,
                                                              os.path.join(initial_window_dir,
                                                                           "IsInWindow_obs.txt"))

    # First time step series values
    predicate_constructors.series_predicate(generated_series, series_ids, 0, INITIAL_SEGMENT_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_obs.txt"),
                                            include_values=True)

    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_target.txt"),
                                            include_values=False)

    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_truth.txt"),
                                            include_values=True)

    open(os.path.join(initial_window_dir, "commands.txt"), "w").write(
        command_constructor.create_forecast_window_commands(generated_series, series_ids, INITIAL_SEGMENT_SIZE,
                                                            INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE, 0))

    for window_idx in range(1, num_windows - 1):
        forecast_window_dir = os.path.join(experiment_dir, forecast_window_dirs[window_idx])

        if not os.path.exists(forecast_window_dir):
            os.makedirs(forecast_window_dir)

        start_time_step = INITIAL_SEGMENT_SIZE + (window_idx * WINDOW_SIZE)
        end_time_step = start_time_step + WINDOW_SIZE - 1

        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_target.txt"),
                                                include_values=False)

        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_truth.txt"),
                                                include_values=True)

        open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
            command_constructor.create_forecast_window_commands(generated_series, series_ids, start_time_step, end_time_step, WINDOW_SIZE, window_idx))

def main():
    P = 0
    p = 2
    period = 10

    np.random.seed(SEED)

    generated_series, coefs = generate_multiple_series(NUM_SERIES, SERIES_LENGTH, p, P, period, seed=SEED)

    test_series = []
    for x in range(int(SERIES_LENGTH / 2)):
        test_series += [5, 10]

    generated_series = np.append(generated_series, [test_series], axis=0)

    num_windows = NUM_FORECAST_WINDOWS
    experiment_dir = os.path.join(DATA_PATH, "test_experiment")
    forecast_window_dirs = [str(time_step).zfill(3) for time_step in range(num_windows)]

    build_psl_data(generated_series, num_windows, experiment_dir, forecast_window_dirs)

if __name__ == '__main__':
    main()


"""

    #for idx, series in enumerate(generated_series[:-1]):
    #    print(coefs[idx])
    #    estimate_AR_coefs.fit_AR_model(series, 0, 300, [1, 2])
    #    estimate_AR_coefs.fit_AR_model(estimate_AR_coefs.generate_aggregate_series(series, 0, 299, 10), 0, 300, [1,2])
    #    print("--")

    #print(generated_series[-1])
    #print(estimate_AR_coefs.generate_aggregate_series(generated_series[-1], 0, 999, 10))

def test_predicates():
    P = 0
    p = 2
    period = 10

    np.random.seed(SEED)

    generated_series, coefs = generate_multiple_series(NUM_SERIES, SERIES_LENGTH, p, P, period, seed=SEED)

    test_series = []
    for x in range(int(SERIES_LENGTH / 2)):
        test_series += [5, 10]

    generated_series = np.append(generated_series, [test_series], axis=0)

    series_ids = np.arange(len(generated_series))

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    time_step_test_dir = os.path.join(DATA_PATH, "001")

    if not os.path.exists(time_step_test_dir):
        os.makedirs(time_step_test_dir)

    predicate_constructors.lag_n_predicate(1, 1, INITIAL_SEGMENT_SIZE, os.path.join(time_step_test_dir, "Lag1_obs.txt"))
    predicate_constructors.series_predicate(generated_series, series_ids, 0, INITIAL_SEGMENT_SIZE -  1, os.path.join(time_step_test_dir, "Series_obs.txt"), include_values=True)
    predicate_constructors.aggregate_series_predicate(generated_series, series_ids, 0, INITIAL_SEGMENT_SIZE - 1, period, os.path.join(time_step_test_dir, "AggregateSeries_obs.txt"))
    predicate_constructors.time_in_aggregate_window_predicate(0, INITIAL_SEGMENT_SIZE -  1, period, os.path.join(time_step_test_dir, "IsInWindow_obs.txt"))

"""
