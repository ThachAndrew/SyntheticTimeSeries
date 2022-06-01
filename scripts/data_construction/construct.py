import command_constructor
import predicate_constructors
import estimate_AR_coefs
import time_series_noise
from ar_forecast import ar_forecast

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np
import os
from statsmodels.tsa.stattools import kpss, adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf

import math

NUM_SERIES_GENERATED = 500
VARIANCE_LOWER_PERCENTILE = 20
VARIANCE_UPPER_PERCENTILE = 80
NUM_SERIES = int(np.rint(NUM_SERIES_GENERATED * (VARIANCE_UPPER_PERCENTILE - VARIANCE_LOWER_PERCENTILE)))
WINDOW_SIZE = 30

INITIAL_SEGMENT_SIZE = 1000
INITIAL_SEGMENT_SIZE += (WINDOW_SIZE - (INITIAL_SEGMENT_SIZE % WINDOW_SIZE))
NUM_FORECAST_WINDOWS = 30

SERIES_LENGTH = INITIAL_SEGMENT_SIZE + (NUM_FORECAST_WINDOWS) * WINDOW_SIZE

MIN_VARIANCE = 10 ** -2

SEED = 55555
REGEN_SEED_MAX = 10 ** 8

GAUSSIAN_NOISE_MU = 0
GAUSSIAN_NOISE_SIGMA = 1

DATA_PATH = "data"
MODEL_PATH = "timeseries_models"

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
        var = np.NAN

        while np.isnan(var) or np.isinf(var) or var < MIN_VARIANCE:
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
            series[x] = np.array([generate_ar(length, all_coefs)])

            var = np.var(series[x])

    return series, coefs

def build_psl_data(generated_series, coefs_and_biases, num_windows, experiment_dir, forecast_window_dirs):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Construct series ID constants
    series_ids = np.arange(len(generated_series))

    # Static predicates
    initial_window_dir = os.path.join(experiment_dir, forecast_window_dirs[0])

    if not os.path.exists(initial_window_dir):
        os.makedirs(initial_window_dir)

    # Time step and window lags
    predicate_constructors.lag_n_predicate(1, 0, SERIES_LENGTH - 1,
                                           os.path.join(initial_window_dir, "Lag1_obs.txt"))
    predicate_constructors.lag_n_predicate(2, 0, SERIES_LENGTH - 1,
                                           os.path.join(initial_window_dir, "Lag2_obs.txt"))
    predicate_constructors.lag_n_predicate(30, 0, SERIES_LENGTH - 1,
                                           os.path.join(initial_window_dir, "Lag30_obs.txt"))

    predicate_constructors.lag_n_predicate(1, 0, int(SERIES_LENGTH/WINDOW_SIZE),
                                           os.path.join(initial_window_dir, "PeriodLag1_obs.txt"))
    predicate_constructors.lag_n_predicate(2, 0, int(SERIES_LENGTH/WINDOW_SIZE),
                                           os.path.join(initial_window_dir, "PeriodLag2_obs.txt"))

    # Series block for specialization
    predicate_constructors.series_block_predicate(series_ids, os.path.join(initial_window_dir, "SeriesBlock_obs.txt"))

    # Assign times to windows for use in hierarchical rule, IsInWindow predicate
    predicate_constructors.time_in_aggregate_window_predicate(0, SERIES_LENGTH - 1, WINDOW_SIZE,
                                                              os.path.join(initial_window_dir,
                                                                           "IsInWindow_obs.txt"))


    # First time step series values
    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE - (WINDOW_SIZE + 1), INITIAL_SEGMENT_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_obs.txt"),
                                            include_values=True)

    agg_series = [predicate_constructors.generate_aggregate_series(series, 0, SERIES_LENGTH - 1, WINDOW_SIZE) for series in generated_series]
    predicate_constructors.oracle_series_predicate(agg_series, series_ids, 0, int(round((SERIES_LENGTH/WINDOW_SIZE))) - 1, 0.0, os.path.join(initial_window_dir, "OracleSeries_obs.txt"))
    # AR Baseline; not used in model, but used for evaluation.
    predicate_constructors.ar_baseline_predicate(generated_series, coefs_and_biases, series_ids, [series[int((INITIAL_SEGMENT_SIZE + WINDOW_SIZE)/WINDOW_SIZE - 1)] for series in agg_series], 0, INITIAL_SEGMENT_SIZE - 1, WINDOW_SIZE,
                                                 os.path.join(initial_window_dir,  "ARBaseline_obs.txt"), os.path.join(initial_window_dir,  "ARBaselineAdj_obs.txt"))

    #exit(1)
    predicate_constructors.agg_series_predicate(series_ids, 0, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE,
                                                os.path.join(initial_window_dir, "AggSeries_target.txt"))

    # Truth & targets for initial forecast window
    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_target.txt"),
                                            include_values=False)

    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_truth.txt"),
                                            include_values=True)

    # First forecast window commands.
    open(os.path.join(initial_window_dir, "commands.txt"), "w").write(
        command_constructor.create_forecast_window_commands(generated_series, series_ids, INITIAL_SEGMENT_SIZE,
                                                            INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE, 0, int(np.rint(INITIAL_SEGMENT_SIZE / WINDOW_SIZE))))

    for window_idx in range(1, num_windows):
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

        # AR Baseline; not used in model, but used for evaluation.
        predicate_constructors.ar_baseline_predicate(generated_series, coefs_and_biases, series_ids, [series[int((INITIAL_SEGMENT_SIZE + WINDOW_SIZE)/WINDOW_SIZE - 1 + window_idx)] for series in agg_series], 0,
                                                     start_time_step - 1, WINDOW_SIZE,
                                                     os.path.join(forecast_window_dir, "ARBaseline_obs.txt"), os.path.join(forecast_window_dir, "ARBaselineAdj_obs.txt"))

        open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
            command_constructor.create_forecast_window_commands(generated_series, series_ids, start_time_step, end_time_step, WINDOW_SIZE, window_idx,
                                                                int(np.rint(INITIAL_SEGMENT_SIZE / WINDOW_SIZE)) + window_idx))

def fit_ar_models(generated_series, start_idx, end_idx, lags):
    coefs_and_biases = dict()

    for series_idx in range(len(generated_series)):
        estimated_ar_coefs, bias = estimate_AR_coefs.fit_AR_model(generated_series[series_idx], start_idx, end_idx,
                                                              lags)
        coefs_and_biases[series_idx] = [estimated_ar_coefs, bias]

    return coefs_and_biases

def gen_hts_model(generated_series, coefs_and_biases, model_name, lags, hierarchical_rule_weight=1.0):
    if not os.path.exists(os.path.join(MODEL_PATH, str(model_name))):
        os.makedirs(os.path.join(MODEL_PATH, str(model_name)))

    # Generate model with hierarchical and AR rules
    hts_model_file = open(os.path.join(MODEL_PATH, str(model_name), "hts.psl"), "w")
    hts_model_lines = "Series(S, +T) / |T| = AggSeries(S, P). {T: IsInWindow(T, P)}\n" + str(hierarchical_rule_weight) + ": AggSeries(S, P) = OracleSeries(S, P) ^2\n\n"

    for series_idx in range(len(generated_series)):
        estimated_ar_coefs, bias = coefs_and_biases[series_idx]

        hts_model_lines += "1.0: Series(S, T) + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = "

        for idx, coef in enumerate(estimated_ar_coefs):
            hts_model_lines += str(coef) + " * Series(S, T_Lag" + str(lags[idx]) + ") + 0.0 * Lag" + str(lags[idx]) + "(T, T_Lag" + str(lags[idx]) + ")  + "

        hts_model_lines += str(bias) + "\n"

    hts_model_file.write(hts_model_lines)

    hts_data_file = open(os.path.join(MODEL_PATH, str(model_name), "hts-eval.data"), "w")

    hts_data_lines = "predicates:\n"
    for lag in lags:
        hts_data_lines += "   Lag" + str(lag) + "/2: closed\n"

    hts_data_lines += """   Series/2: open
   PeriodLag1/2: closed
   PeriodLag2/2: closed
   OracleSeries/2: closed
   AggSeries/2: open
   IsInWindow/2: closed
   SeriesBlock/2: closed

observations:\n"""

    for lag in lags:
        hts_data_lines += "   Lag" + str(lag) + ":   ../data/hts/eval/000/Lag" + str(lag) + "_obs.txt"

    hts_data_lines += """
       Series: ../data/hts/eval/000/Series_obs.txt
   PeriodLag1: ../data/hts/eval/000/PeriodLag1_obs.txt
   PeriodLag2: ../data/hts/eval/000/PeriodLag2_obs.txt
   IsInWindow: ../data/hts/eval/000/IsInWindow_obs.txt
   SeriesBlock: ../data/hts/eval/000/SeriesBlock_obs.txt
   OracleSeries: ../data/hts/eval/000/OracleSeries_obs.txt

targets:
   Series: ../data/hts/eval/000/Series_target.txt

truth:
   Series: ../data/hts/eval/000/Series_truth.txt
    """

    hts_data_file.write(hts_data_lines)

    return coefs_and_biases

# normalizes a series to a range of [0,1]
def normalize(series):
    min_element = min(series)
    max_element = max(series)

    return [(float(i)-min_element)/(max_element-min_element) for i in series]

def main():
    # Order of (S)AR model
    P = 1
    p = 1
    period = 30

    # Controls Gaussian noise to be added to every generated series
    add_noise = True
    noise_mu = 0
    noise_sigma = 2

    # Lags present in the (S)AR model, computed from its order.
    lags = np.zeros(p + P)
    lags[:p] = np.arange(p) + 1
    lags[p:] = period * (np.arange(P) + 1)
    lags = lags.astype(int)

    # Generate random AR data
    np.random.seed(SEED)
    generated_series, true_coefs = generate_multiple_series(NUM_SERIES_GENERATED, SERIES_LENGTH, p, P, period, seed=SEED, enforce_stationarity=False)

    # Filter out series that aren't between the two thresholds.
    filtered_generated_series = []
    filtered_true_coefs = []

    # Add noise and normalize
    for series_idx in range(len(generated_series)):
        if add_noise:
            generated_series[series_idx] = time_series_noise.add_gaussian_noise(generated_series[series_idx], noise_mu,
                                                                            noise_sigma, SEED)
        generated_series[series_idx] = normalize(generated_series[series_idx])

    variances = [np.var(series) for series in generated_series]

    # Compute upper and lower variance thresholds, which requires discarding inf. or undef. variance series
    upper_var_threshold = np.percentile(variances, VARIANCE_UPPER_PERCENTILE)
    lower_var_threshold = np.percentile(variances, VARIANCE_LOWER_PERCENTILE)

    # Filter
    for series_idx, series in enumerate(generated_series):
        if np.isnan(np.var(series)) or np.isinf(np.var(series)):
            continue

        if lower_var_threshold <= np.var(series) and np.var(series) <= upper_var_threshold:
            filtered_generated_series += [series]
            filtered_true_coefs += [true_coefs[series_idx]]

    generated_series = filtered_generated_series
    coefs_and_biases = fit_ar_models(generated_series, 0, INITIAL_SEGMENT_SIZE, lags)
    gen_hts_model(generated_series, coefs_and_biases, "hierarchical_test_model", lags, hierarchical_rule_weight=5.0)

    # Set up first experiment
    experiment_dir = os.path.join(DATA_PATH, "test_experiment_noise_seasonal", "eval")
    forecast_window_dirs = [str(window_idx).zfill(3) for window_idx in range(NUM_FORECAST_WINDOWS)]
    build_psl_data(generated_series, coefs_and_biases, NUM_FORECAST_WINDOWS, experiment_dir, forecast_window_dirs)

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
"""

    #coefs_and_biases = [[] for series in generated_series]

    for x in range(20, 30):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        coefs_t, bias_t = estimate_AR_coefs.fit_AR_model(generated_series_ns[x], 0, len(generated_series_ns[x]) - 1, [1, 2, 30])

        print(ar_forecast(generated_series_ns[x], coefs_t, bias_t, 30))
        cmap = ListedColormap(['b', 'g'])
        segments = np.concatenate([generated_series_ns[x][-120:], ar_forecast(generated_series_ns[x], coefs_t, bias_t, 30)])

        lc = LineCollection(segments, cmap=cmap)

        print(len(generated_series_ns[x]))
        ax1.plot(generated_series_ns[x][-120:])


        #agg_series = predicate_constructors.generate_aggregate_series(generated_series_ns[x], 0, len(generated_series_s[x]) - 1, WINDOW_SIZE)
        ax2.add_collection(lc)

        ax1.plot()
        ax2.plot()
        plt.show()

    exit(1)
"""
"""
hts_model_lines += "#Series " + str(series_idx) + "\n"

# Create AR rules
for idx, coef in enumerate(estimated_ar_coefs):
    if coef > 0:
        hts_model_lines += str(coef) + ": Series(S, T) - Series(S, T_Lag" + str(idx + 1) + ") + 0.0 * Lag" + str(idx + 1) + "(T, T_Lag" + str(idx + 1) + ") + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = 0.0 ^2"
    else:
        hts_model_lines += str(-1 * coef) + ": Series(S, T) + Series(S, T_Lag" + str(idx + 1) + ") + 0.0 * Lag" + str(idx + 1) + "(T, T_Lag" + str(idx + 1) + ") + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = 0.0 ^2"

    hts_model_lines += "\n"

# Bias rule
if bias > 0:
    hts_model_lines += str(bias / 2) + ": Series(S, T) + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = 1.0\n"
else:
    hts_model_lines += str(bias / 2) + ": Series(S, T) + 0.0 * SeriesBlock(S, '" + str(series_idx) + "') = 0.0\n"

hts_model_lines += "\n"
"""