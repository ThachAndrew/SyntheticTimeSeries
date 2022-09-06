import command_constructor
import predicate_constructors
import estimate_AR_coefs
import time_series_noise

from datetime import datetime
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.vector_ar.var_model import VAR

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = "data"
MODEL_PATH = "timeseries_models"
EXPERIMENT_NAME = "test_experiment_t"

NUM_SERIES_GENERATED = 250
VARIANCE_LOWER_PERCENTILE = 40
VARIANCE_UPPER_PERCENTILE = 60
WINDOW_SIZE = 8

INITIAL_SEGMENT_SIZE = 1000
INITIAL_SEGMENT_SIZE += (WINDOW_SIZE - (INITIAL_SEGMENT_SIZE % WINDOW_SIZE))
NUM_FORECAST_WINDOWS = 30

SERIES_LENGTH = INITIAL_SEGMENT_SIZE + (NUM_FORECAST_WINDOWS) * WINDOW_SIZE

MIN_VARIANCE = 10 ** -1

SEED = 55555

# Experiment 1 only: base series noise scale
BASE_SERIES_NOISE_SIGMAS = [0.25]

# Experiment 2 only: oracle series noise scale
ORACLE_SERIES_NOISE_SIGMAS = []

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

# Return k series of length n with:
# - given AR coefs
# - specified means (probably 0s here)
# - a kxk covariance matrix
# (Optional scale for initializing values)
def generate_ar_cluster(series_count, n, coefs, means, e_cov, seed, init_value_scale=1):
    p = coefs.shape[1]
    adj_n = n + p

    np.random.seed(seed)
    errors = np.random.multivariate_normal(means, e_cov, n).T
    initial_values = np.random.normal(0, init_value_scale, (series_count, p))

    final_series = np.zeros((series_count, p+n))
    for series_idx in range(series_count):
        for i in range(p):
            final_series[series_idx][i] = initial_values[series_idx][i]

        for i in range(p, adj_n):
            visible_series = final_series[series_idx][i - p:i]
            final_series[series_idx][i] = errors[series_idx][i - p] + np.dot(coefs[series_idx][::-1], visible_series)

    # Slice off initial values
    return final_series[:, p:]
    # return ar

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(max(len(ar_phis), P * period))
    phis[0:len(ar_phis)] = ar_phis
    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

# Returns array of stationary coefs in a_1, a_2, ..., a_p order
def generate_cluster_coefs(num_series, p, seed=333, enforce_stationarity=True):
    np.random.seed(seed)

    all_coefs = np.empty((num_series, p))

    for x in range(num_series):
        coefs = np.array([(x - 0.5) * 2 for x in np.random.rand(p)])
        char_polynomial = np.append(1, -1 * coefs)[::-1]

        if enforce_stationarity:
            while np.min(np.abs(np.roots(char_polynomial))) < 1:
                print(char_polynomial)
                print("Resampling")
                coefs = np.array([(x - 0.5) * 2 for x in np.random.rand(p)])
                # [a1, a2] -> [-a2 -a1 1]
                char_polynomial = np.append(1, -1 * coefs)[::-1]

        all_coefs[x] = coefs

    return all_coefs

def build_psl_data(generated_series, coefs_and_biases, cluster_oracle_noise_sigma, oracle_noise_sigma, lags, num_windows, experiment_dir, forecast_window_dirs,
                   cluster_hierarchy=True, cluster_size=5):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Construct series ID constants
    series_ids = np.arange(len(generated_series))
    cluster_ids = np.arange(int(len(generated_series) / cluster_size))

    # Static predicates
    initial_window_dir = os.path.join(experiment_dir, forecast_window_dirs[0])

    if not os.path.exists(initial_window_dir):
        os.makedirs(initial_window_dir)

    # Time step and window lags
    # LagN predicates
    for lag in lags:
        predicate_constructors.lag_n_predicate(lag, 0, SERIES_LENGTH - 1,
                                               os.path.join(initial_window_dir, "Lag" + str(lag) + "_obs.txt"))

    # Series block for specialization
    # SeriesBlock predicate
    predicate_constructors.series_block_predicate(series_ids, os.path.join(initial_window_dir, "SeriesBlock_obs.txt"))

    # Assign times to windows for use in hierarchical rule, IsInWindow predicate
    # IsInWindow predicate
    predicate_constructors.time_in_aggregate_window_predicate(0, SERIES_LENGTH - 1, WINDOW_SIZE,
                                                              os.path.join(initial_window_dir,
                                                                           "IsInWindow_obs.txt"))


    # Series predicate
    # Initial segment series values
    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE - (WINDOW_SIZE + 1), INITIAL_SEGMENT_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_obs.txt"),
                                            include_values=True)

    # Truth & targets for initial forecast window
    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_target.txt"),
                                            include_values=False)

    predicate_constructors.series_predicate(generated_series, series_ids, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                                            os.path.join(initial_window_dir, "Series_truth.txt"),
                                            include_values=True)


    if cluster_hierarchy:
        # Cluster stations, then sum series within clusters to generate oracle series, and create targets for the ClusterMean predicate
        # SeriesCluster predicate and ClusterOracle predicate
        cluster_series_map, cluster_agg_series_list = predicate_constructors.series_cluster_predicate(generated_series, series_ids, cluster_size, os.path.join(initial_window_dir, "SeriesCluster_obs.txt"))
        cluster_forecasts = predicate_constructors.cluster_oracle_predicate(generated_series, cluster_series_map, cluster_oracle_noise_sigma, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE, SERIES_LENGTH - 1, WINDOW_SIZE, os.path.join(initial_window_dir, "ClusterOracle_obs.txt"))
        # ClusterMean predicate
        predicate_constructors.cluster_mean_predicate(cluster_series_map, INITIAL_SEGMENT_SIZE, SERIES_LENGTH - 1, os.path.join(initial_window_dir, "ClusterMean_target.txt"))

    # Compute aggregate series used to generate oracle values
    agg_series = [predicate_constructors.generate_aggregate_series(series, 0, SERIES_LENGTH - 1, WINDOW_SIZE) for series in generated_series]

    # OracleSeries predicate.
    predicate_constructors.oracle_series_predicate(agg_series, series_ids, 0, int(round(SERIES_LENGTH/WINDOW_SIZE)) - 1, oracle_noise_sigma,
                                                   os.path.join(initial_window_dir, "OracleSeries_obs.txt"))

    # Naive prediction predicate (WIP, need to make it based on historical means)
    predicate_constructors.naive_prediction_predicate(agg_series, series_ids, int(INITIAL_SEGMENT_SIZE / WINDOW_SIZE), int(INITIAL_SEGMENT_SIZE / WINDOW_SIZE) + 1, WINDOW_SIZE, os.path.join(initial_window_dir, "NaiveBaseline_obs.txt"))

    # AR Baseline; not used in model, but used for evaluation.
    # ARBaseline and ARBaselineAdj predicates.
    base_forecast_series_list = predicate_constructors.ar_baseline_predicate(generated_series, coefs_and_biases, series_ids, [series[int((INITIAL_SEGMENT_SIZE + WINDOW_SIZE)/WINDOW_SIZE - 1)] for series in agg_series], 0, INITIAL_SEGMENT_SIZE - 1, WINDOW_SIZE,
                                                 os.path.join(initial_window_dir,  "ARBaseline_obs.txt"), os.path.join(initial_window_dir,  "ARBaselineAdj_obs.txt"))

    predicate_constructors.cluster_equal_bias_ar_forecasts_predicate(generated_series, coefs_and_biases, cluster_forecasts, cluster_size,
                                              0, INITIAL_SEGMENT_SIZE - 1, INITIAL_SEGMENT_SIZE, WINDOW_SIZE, os.path.join(initial_window_dir, "ARBaselineNaiveTD_obs.txt"))
    if cluster_hierarchy:
        #predicate_constructors.cluster_ar_baseline_predicate(generated_series, coefs_and_biases, cluster_agg_series_list, INITIAL_SEGMENT_SIZE, WINDOW_SIZE, cluster_size, os.path.join(initial_window_dir,  "ARBaselinec_obs.txt"))
        #exit(1)
        predicate_constructors.fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map, cluster_forecasts, cluster_size, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                             os.path.join(initial_window_dir, "ARBaselineFP_obs.txt"))

    # AggSeries predicate
    predicate_constructors.agg_series_predicate(series_ids, 0, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE,
                                                os.path.join(initial_window_dir, "AggSeries_target.txt"))

    # First forecast window commands.
    open(os.path.join(initial_window_dir, "commands.txt"), "w").write(
        command_constructor.create_forecast_window_commands(generated_series, series_ids, cluster_ids, INITIAL_SEGMENT_SIZE,
                                                            INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE, 0, int(np.rint(INITIAL_SEGMENT_SIZE / WINDOW_SIZE)),
                                                            cluster=cluster_hierarchy))

    for window_idx in range(1, num_windows):
        forecast_window_dir = os.path.join(experiment_dir, forecast_window_dirs[window_idx])

        if not os.path.exists(forecast_window_dir):
            os.makedirs(forecast_window_dir)

        start_time_step = INITIAL_SEGMENT_SIZE + (window_idx * WINDOW_SIZE)
        end_time_step = start_time_step + WINDOW_SIZE - 1

        # Write series target/truth for current forecast window. Currently only being used for evaluation.
        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_target.txt"),
                                                include_values=False)

        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_truth.txt"),
                                                include_values=True)

        # Naive prediction for current forecast window, WIP
        predicate_constructors.naive_prediction_predicate(agg_series, series_ids, window_idx + int(INITIAL_SEGMENT_SIZE/WINDOW_SIZE), window_idx + int(INITIAL_SEGMENT_SIZE/WINDOW_SIZE) + 1, WINDOW_SIZE, os.path.join(forecast_window_dir, "NaiveBaseline_obs.txt"))


        # AR Baseline; not used in model, but used for evaluation.
        predicate_constructors.ar_baseline_predicate(generated_series, coefs_and_biases, series_ids, [series[int((INITIAL_SEGMENT_SIZE + WINDOW_SIZE)/WINDOW_SIZE - 1 + window_idx)] for series in agg_series], 0, start_time_step - 1, WINDOW_SIZE, os.path.join(forecast_window_dir, "ARBaseline_obs.txt"), os.path.join(forecast_window_dir, "ARBaselineAdj_obs.txt"))
        predicate_constructors.fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map,
                                                        cluster_forecasts, cluster_size, INITIAL_SEGMENT_SIZE,
                                                        INITIAL_SEGMENT_SIZE + (window_idx * WINDOW_SIZE), INITIAL_SEGMENT_SIZE + ((window_idx + 1) * WINDOW_SIZE) - 1,
                                                        os.path.join(forecast_window_dir, "ARBaselineFP_obs.txt"))

        predicate_constructors.cluster_equal_bias_ar_forecasts_predicate(generated_series, coefs_and_biases,
                                                                         cluster_forecasts, cluster_size,
                                                                         0, start_time_step - 1, INITIAL_SEGMENT_SIZE,
                                                                         WINDOW_SIZE,
                                                                         os.path.join(forecast_window_dir,
                                                                                      "ARBaselineNaiveTD_obs.txt"))
        open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
            command_constructor.create_forecast_window_commands(generated_series, series_ids, cluster_ids, start_time_step, end_time_step, WINDOW_SIZE, window_idx,
                                                                int(np.rint(INITIAL_SEGMENT_SIZE / WINDOW_SIZE)) + window_idx, cluster=cluster_hierarchy))

# Fits AR models to a list of series
def fit_ar_models(generated_series, start_idx, end_idx, p):
    coefs_and_biases = dict()

    for series_idx in range(len(generated_series)):
        print("Fitting " + str(series_idx))
        m = SARIMAX(generated_series[series_idx][start_idx:end_idx], trend='c', order=(p, 0, 0))
        r = m.fit(disp=False, return_params=True)
        estimated_ar_coefs = r[1:-1]

        #ols_coefs, ols_bias = estimate_AR_coefs.fit_AR_model(generated_series[series_idx], start_idx, end_idx, [1, 2])

        bias = r[0]

        #print(str(estimated_ar_coefs) + " " + str(bias))
        #print(str(ols_coefs) + " " + str(ols_bias))


        coefs_and_biases[series_idx] = [estimated_ar_coefs, bias]

    return coefs_and_biases

# Generate hierarchical time series PSL models and data files.
def gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, model_name, lags, temporal_hierarchical_rule_weight=1.0, cluster_hierarchical_rule_weight=1.0, temporal_rules=True, cluster_rules=False, cluster_hard=False, mean_hard=False):
    if not os.path.exists(os.path.join(MODEL_PATH, experiment_name_dir, str(model_name))):
        os.makedirs(os.path.join(MODEL_PATH, experiment_name_dir, str(model_name)))

    # Generate model with hierarchical and AR rules
    hts_model_file = open(os.path.join(MODEL_PATH, experiment_name_dir, str(model_name), "hts.psl"), "w")
    if temporal_rules:
        hts_model_lines = "Series(S, +T) / |T| = AggSeries(S, P). {T: IsInWindow(T, P)}\n"
        if not mean_hard:
            hts_model_lines += str(temporal_hierarchical_rule_weight) + ": AggSeries(S, P) = OracleSeries(S, P) ^2\n\n"
        else:
            hts_model_lines += "AggSeries(S, P) = OracleSeries(S, P) .\n\n"
    else:
        hts_model_lines = "##\n\n"

    if cluster_rules:
        hts_model_lines += "Series(+S, T) / |S| = ClusterMean(C, T). {S: SeriesCluster(S, C)} \n"
        if not cluster_hard:
            hts_model_lines += str(cluster_hierarchical_rule_weight) + ": ClusterMean(C, T) = ClusterOracle(C, T) ^2\n"
        else:
            hts_model_lines += "ClusterMean(C, T) = ClusterOracle(C, T) .\n"

        # Add AR rules
    for series_idx in range(len(generated_series)):
        estimated_ar_coefs, bias = coefs_and_biases[series_idx]

        hts_model_lines += "1.0: Series(S, T) + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = "

        for idx, coef in enumerate(estimated_ar_coefs):
            hts_model_lines += str(coef) + " * Series(S, T_Lag" + str(lags[idx]) + ") + 0.0 * Lag" + str(lags[idx]) + "(T, T_Lag" + str(lags[idx]) + ")  + "

        hts_model_lines += str(bias) + "\n"

    hts_model_file.write(hts_model_lines)

    # Generate data file
    hts_data_file = open(os.path.join(MODEL_PATH, experiment_name_dir, str(model_name), "hts-eval.data"), "w")
    hts_data_lines = "predicates:\n"
    for lag in lags:
        hts_data_lines += "   Lag" + str(lag) + "/2: closed\n"

    hts_data_lines += """   Series/2: open
   OracleSeries/2: closed
   AggSeries/2: open
   IsInWindow/2: closed
   SeriesBlock/2: closed
   ClusterMean/2: open
   SeriesCluster/2: closed
   ClusterOracle/2: closed

observations:\n"""

    for lag in lags:
        hts_data_lines += "   Lag" + str(lag) + ":   ../data/hts/eval/000/Lag" + str(lag) + "_obs.txt \n"

    hts_data_lines += """   Series: ../data/hts/eval/000/Series_obs.txt
   IsInWindow: ../data/hts/eval/000/IsInWindow_obs.txt
   SeriesBlock: ../data/hts/eval/000/SeriesBlock_obs.txt
   OracleSeries: ../data/hts/eval/000/OracleSeries_obs.txt
   SeriesCluster: ../data/hts/eval/000/SeriesCluster_obs.txt
   ClusterOracle: ../data/hts/eval/000/ClusterOracle_obs.txt

targets:
   Series: ../data/hts/eval/000/Series_target.txt
   

truth:
   Series: ../data/hts/eval/000/Series_truth.txt
    """

    hts_data_file.write(hts_data_lines)

    return coefs_and_biases

# Normalizes a series to a range of [0,1]
def normalize(series):
    min_element = min(series)
    max_element = max(series)

    return [(float(i)-min_element)/(max_element-min_element) for i in series]

def normalize_multiple(series_list):
    min_element = min([min(series) for series in series_list])
    max_element = max([max(series) for series in series_list])

    normalized_series_list = []

    for series_idx in range(len(series_list)):
        normalized_series_list += [[(float(e) - min_element) / (max_element - min_element) for e in series_list[series_idx]]]

    return normalized_series_list

def set_up_experiment(exp_name, series_count, p, cluster_size, err_means, err_cov_matrix, forecast_variance_scale, init_segment_size, num_windows, experiment_name_dir):
    lags = [l + 1 for l in np.arange(p)]
    generated_series, coefs = generate_dataset(series_count, cluster_size, p, init_segment_size + (NUM_FORECAST_WINDOWS*WINDOW_SIZE), err_means, err_cov_matrix, seed=1234)

    # All off-diagonal elements (cross-series covariances) are equal
    cross_cov = err_cov_matrix[0][1]

    for cluster_idx in range(int(series_count / cluster_size)):
        generated_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size] = normalize_multiple(generated_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size])


    coefs_and_biases = fit_ar_models(generated_series, 0, INITIAL_SEGMENT_SIZE-1, lags)

    #print(coefs_and_biases)

    experiment_dir = os.path.join(DATA_PATH, experiment_name_dir, "eval")
    forecast_window_dirs = [str(window_idx).zfill(3) for window_idx in range(NUM_FORECAST_WINDOWS)]

    build_psl_data(generated_series, coefs_and_biases, forecast_variance_scale, 0.0, lags,
                   num_windows, experiment_dir, forecast_window_dirs,
                   cluster_hierarchy=True, cluster_size=cluster_size)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_10", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_combined", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, mean_hard=True)

    options_file_handle = open(os.path.join(DATA_PATH, experiment_name_dir, "options.txt"), "w")
    options_file_lines = "p\t" + str(p) + "\nWindow_size\t" + str(WINDOW_SIZE) + \
                         "\nSeed\t" + str(SEED) + "\ncluster_forecast_noise_variance\t" + str(forecast_variance_scale) + "\nCluster_size\t" + str(cluster_size) + "\n"
    options_file_handle.write(options_file_lines)

def generate_dataset(series_count, cluster_size, p, n, means, e_cov_matrix, seed=1234):
    coefs = generate_cluster_coefs(cluster_size, p)
    series_list = generate_ar_cluster(cluster_size, n, coefs, means, e_cov_matrix, seed)

    all_coefs = coefs

    while series_list.shape[0] < series_count:
        seed += 1

        coefs = generate_cluster_coefs(cluster_size, p, seed=seed)
        series_list = np.append(series_list, generate_ar_cluster(cluster_size, n, coefs, means, e_cov_matrix, seed), axis=0)
        all_coefs = np.append(all_coefs, coefs, axis=0)

    return series_list, all_coefs

# Given a known cluster in the form of [[Y_{1, 0}, Y_{1, 1}, ... Y_{1, t}], [Y_{2, 0}, Y_{2, 1}, ..., Y_{2, t}] .... ]],
# compute a simulated forecast for the aggregate of these series.
#
# cluster_series: cluster_size x t array containing all series data for this cluster
# t: Last observation before start of forecast window
# h: Forecast window length
# z: number of observations that were used in training / parameter estimation
# forecast_variance_scale: constant multiplier to the true aggregate series variance used to define the scale of the noise added to the pseudo-forecast

def main():
    series_count = 12
    cluster_size = 4

    n = 1000
    p = 4
    cross_covs = [0.25, 0.5, 0.75, 1]

    forecast_variance_scales = [0.25, 0.5, 0.75, 1]


    exp_name = "E1"

    #data, coefs = generate_dataset(series_count, cluster_size, p, n, err_means, err_cov_matrix, seed=4444)

    for forecast_variance_scale in forecast_variance_scales:
        for cross_cov in cross_covs:
            # Experiment 1 setup, cross-series correlated noise terms in k otherwise independent AR series
            err_means = np.zeros(cluster_size)
            err_cov_matrix = np.full((cluster_size, cluster_size), cross_cov)
            np.fill_diagonal(err_cov_matrix, 1)

            exp_dir = os.path.join(exp_name, "clus_or_variance_" + str(forecast_variance_scale), "cross_cov_" + str(cross_cov))
            set_up_experiment(exp_name, series_count, p, cluster_size, err_means, err_cov_matrix, forecast_variance_scale,
                      INITIAL_SEGMENT_SIZE, NUM_FORECAST_WINDOWS, exp_dir)

def set_up_experiment_old(p, P, period, add_noise, base_noise_scale, cluster_noise_scale, oracle_noise_scale, experiment_data_dir, experiment_model_name,
                      temporal_hierarchy_rule_weight=100.0,
                      cluster_hierarchy_rule_weight=100.0,
                      cluster_hierarchy=False, cluster_size=5):
    # Lags present in the (S)AR model, computed from its order.
    lags = np.zeros(p + P)
    lags[:p] = np.arange(p) + 1
    lags[p:] = period * (np.arange(P) + 1)
    lags = lags.astype(int)

    # Generate random AR data
    np.random.seed(SEED)
    generated_series, true_coefs = generate_multiple_series(NUM_SERIES_GENERATED, SERIES_LENGTH, p, P, period,
                                                            seed=SEED, enforce_stationarity=True)

    # Filter out series that aren't between the two thresholds.
    filtered_generated_series = []
    filtered_true_coefs = []

    # Add noise and normalize
    for series_idx in range(len(generated_series)):
        if add_noise:
            generated_series[series_idx] = time_series_noise.add_gaussian_noise(generated_series[series_idx], 0,
                                                                                base_noise_scale, SEED)
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

    if cluster_hierarchy:
        gen_hts_model(generated_series, coefs_and_biases, experiment_model_name, lags,
                      temporal_hierarchical_rule_weight=temporal_hierarchy_rule_weight, cluster_hierarchical_rule_weight=cluster_hierarchy_rule_weight,
                      cluster_rules=True)
    else:
        gen_hts_model(generated_series, coefs_and_biases, experiment_model_name, lags, temporal_hierarchical_rule_weight=100.0)
        gen_hts_model(generated_series, coefs_and_biases, experiment_model_name + "_joint", lags,
                      temporal_hierarchical_rule_weight=100.0)

    # Set up first experiment
    experiment_dir = os.path.join(DATA_PATH, experiment_data_dir, "eval")
    forecast_window_dirs = [str(window_idx).zfill(3) for window_idx in range(NUM_FORECAST_WINDOWS)]
    build_psl_data(generated_series, coefs_and_biases, cluster_noise_scale, oracle_noise_scale, lags,
                   NUM_FORECAST_WINDOWS, experiment_dir, forecast_window_dirs, cluster_hierarchy=cluster_hierarchy)

    options_file_handle = open(os.path.join(DATA_PATH, experiment_data_dir, "options.txt"), "w")
    options_file_lines = "p\t" + str(p) + "\nP\t" + str(P) + "\nWindow_size\t" + str(WINDOW_SIZE) + \
                         "\nSeed\t" + str(SEED) + "\nBase_series_noise_scale\t" + str(base_noise_scale) + \
                         "\nOracle_series_noise_scale\t" + str(oracle_noise_scale) + "\nCluster_size\t" + str(cluster_size) + "\n"
    options_file_handle.write(options_file_lines)

if __name__ == '__main__':
    main()
