import copy

import command_constructor
import predicate_constructors
import estimate_AR_coefs
import time_series_noise

from datetime import datetime
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss
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
DISCARDED_SEGMENT_LENGTH = 500
"""
NUM_SERIES_GENERATED = 250
VARIANCE_LOWER_PERCENTILE = 40
VARIANCE_UPPER_PERCENTILE = 60
window_size = 4

series_length = initial_segment_size + (NUM_FORECAST_WINDOWS) * window_size
"""
SEED = 55555

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
    adj_n = n + p + DISCARDED_SEGMENT_LENGTH

    np.random.seed(seed)
    errors = np.random.multivariate_normal(means, e_cov, n).T
    initial_values = np.random.normal(0, init_value_scale, (series_count, p))

    final_series = np.zeros((series_count, adj_n))
    for series_idx in range(series_count):
        for i in range(p):
            final_series[series_idx][i] = initial_values[series_idx][i]

        for i in range(p, adj_n):
            visible_series = final_series[series_idx][i - p:i]
            final_series[series_idx][i] = errors[series_idx][i - p - DISCARDED_SEGMENT_LENGTH] + np.dot(coefs[series_idx][::-1], visible_series)

    # Slice off initial values
    return final_series[:, DISCARDED_SEGMENT_LENGTH + p:]
    # return ar

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(max(len(ar_phis), P * period))
    phis[0:len(ar_phis)] = ar_phis

    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

# Returns array of stationary coefs in a_1, a_2, ..., a_p order
def generate_cluster_coefs(num_series, p, window_size, seed=333, enforce_stationarity=True, seasonal=False, P=0):
    np.random.seed(seed)

    all_coefs = np.empty((num_series, max(p, window_size * P)))

    for x in range(num_series):
        ar_coefs = np.array([(x - 0.5) * 2 for x in np.random.rand(p)])

        if seasonal:
            seasonal_coefs = np.array([(x - 0.5) * 2 for x in np.random.rand(P)])
            sar_coefs = generate_sar_phis(ar_coefs, seasonal_coefs, P, window_size)

        char_polynomial = np.append(1, -1 * ar_coefs)[::-1]

        if enforce_stationarity:
            while np.min(np.abs(np.roots(char_polynomial))) < 1:
                print(char_polynomial)
                print("Resampling")
                ar_coefs = np.array([(x - 0.5) * 2 for x in np.random.rand(p)])
                # [a1, a2] -> [-a2 -a1 1]
                char_polynomial = np.append(1, -1 * ar_coefs)[::-1]

        all_coefs[x] = ar_coefs

    return all_coefs

def build_psl_data(generated_series, noisy_series, coefs_and_biases, cluster_oracle_noise_sigma, oracle_noise_sigma, lags, num_windows, experiment_dir, forecast_window_dirs, window_size,
                   initial_segment_size, series_length, cluster_hierarchy=True, cluster_size=5, batch=True):
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
        predicate_constructors.lag_n_predicate(lag, 0, series_length - 1,
                                               os.path.join(initial_window_dir, "Lag" + str(lag) + "_obs.txt"))


    predicate_constructors.lags_predicate(lags, initial_segment_size - window_size, series_length - 1, os.path.join(initial_window_dir, "Lags_obs.txt"))

    # Series block for specialization
    # SeriesBlock predicate
    predicate_constructors.series_block_predicate(series_ids, os.path.join(initial_window_dir, "SeriesBlock_obs.txt"))

    # Assign times to windows for use in hierarchical rule, IsInWindow predicate
    # IsInWindow predicate
    predicate_constructors.time_in_aggregate_window_predicate(0, series_length - 1, window_size,
                                                              os.path.join(initial_window_dir,
                                                                           "IsInWindow_obs.txt"))

    predicate_constructors.series_mean_predicate(noisy_series, series_ids, initial_segment_size, os.path.join(initial_window_dir, "SeriesMean_obs.txt"))

    # Series predicate
    # Initial segment series values
    predicate_constructors.series_predicate(noisy_series, series_ids, initial_segment_size - (window_size + 1), initial_segment_size - 1,
                                            os.path.join(initial_window_dir, "Series_obs.txt"),
                                            include_values=True)

    # Truth & targets for initial forecast window
    predicate_constructors.series_predicate(generated_series, series_ids, initial_segment_size, initial_segment_size + window_size - 1,
                                            os.path.join(initial_window_dir, "Series_target.txt"),
                                            include_values=False)

    predicate_constructors.series_predicate(generated_series, series_ids, initial_segment_size, initial_segment_size + window_size - 1,
                                            os.path.join(initial_window_dir, "Series_truth.txt"),
                                            include_values=True)


    if cluster_hierarchy:
        # Cluster stations, then sum series within clusters to generate oracle series, and create targets for the ClusterMean predicate
        # SeriesCluster predicate and ClusterOracle predicate
        cluster_series_map, cluster_agg_series_list = predicate_constructors.series_cluster_predicate(generated_series, series_ids, cluster_size, os.path.join(initial_window_dir, "SeriesCluster_obs.txt"))
        cluster_forecasts = predicate_constructors.cluster_oracle_predicate(generated_series, cluster_series_map, cluster_oracle_noise_sigma, initial_segment_size, initial_segment_size, series_length - 1, window_size, os.path.join(initial_window_dir, "ClusterOracle_obs.txt"))
        # ClusterMean predicate

        predicate_constructors.cluster_mean_predicate(cluster_series_map, initial_segment_size, series_length - 1, os.path.join(initial_window_dir, "ClusterMean_target.txt"))

    # Compute aggregate series used to generate oracle values
    agg_series = [predicate_constructors.generate_aggregate_series(series, 0, series_length - 1, window_size) for series in generated_series]

    # OracleSeries predicate.
    predicate_constructors.oracle_series_predicate(agg_series, series_ids, 0, int(round(series_length/window_size)) - 1, oracle_noise_sigma,
                                                   os.path.join(initial_window_dir, "OracleSeries_obs.txt"))

    # Naive prediction predicate (WIP, need to make it based on historical means)
    # predicate_constructors.naive_prediction_predicate(agg_series, series_ids, int(initial_segment_size / window_size), int(initial_segment_size / window_size) + 1, window_size, os.path.join(initial_window_dir, "NaiveBaseline_obs.txt"))

    # AR Baseline; not used in model, but used for evaluation.
    # ARBaseline and ARBaselineAdj predicates.
    base_forecast_series_list = predicate_constructors.ar_baseline_predicate(noisy_series, coefs_and_biases, series_ids, [series[int((initial_segment_size + window_size)/window_size - 1)] for series in agg_series], 0, initial_segment_size - 1, window_size,
                                                 os.path.join(initial_window_dir,  "ARBaseline_obs.txt"), os.path.join(initial_window_dir,  "ARBaselineAdj_obs.txt"))

    predicate_constructors.cluster_equal_bias_ar_forecasts_predicate(noisy_series, coefs_and_biases, cluster_forecasts, cluster_size,
                                              0, initial_segment_size - 1, initial_segment_size, window_size, os.path.join(initial_window_dir, "ARBaselineNaiveTD_obs.txt"))
    if cluster_hierarchy:
        #predicate_constructors.cluster_ar_baseline_predicate(generated_series, coefs_and_biases, cluster_agg_series_list, initial_segment_size, window_size, cluster_size, os.path.join(initial_window_dir,  "ARBaselinec_obs.txt"))
        #exit(1)

        # TODO@Alex: re-implement top-down FP

        predicate_constructors.fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map, cluster_forecasts, cluster_size, initial_segment_size, initial_segment_size, initial_segment_size + window_size - 1,
                             os.path.join(initial_window_dir, "ARBaselineFP_obs.txt"))

    # AggSeries predicate
    predicate_constructors.agg_series_predicate(series_ids, 0, initial_segment_size + window_size - 1, window_size,
                                                os.path.join(initial_window_dir, "AggSeries_target.txt"))

    # First forecast window commands.
    if batch:
        open(os.path.join(initial_window_dir, "commands.txt"), "w").write(
            command_constructor.create_forecast_window_commands(generated_series, noisy_series, series_ids, cluster_ids, initial_segment_size,
                                                                initial_segment_size + window_size - 1, window_size, 0, int(np.rint(initial_segment_size / window_size)),
                                                                cluster=cluster_hierarchy))
    else:
        open(os.path.join(initial_window_dir, "commands.txt"), "w").write(
            command_constructor.iterative_forecast_commands(generated_series, series_ids, cluster_ids, initial_segment_size,
                                                                initial_segment_size + window_size - 1, window_size, 0, int(np.rint(initial_segment_size / window_size))))

    for window_idx in range(1, num_windows):
        forecast_window_dir = os.path.join(experiment_dir, forecast_window_dirs[window_idx])

        if not os.path.exists(forecast_window_dir):
            os.makedirs(forecast_window_dir)

        start_time_step = initial_segment_size + (window_idx * window_size)
        end_time_step = start_time_step + window_size - 1

        # Write series target/truth for current forecast window. Currently only being used for evaluation.
        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_target.txt"),
                                                include_values=False)

        predicate_constructors.series_predicate(generated_series, series_ids, start_time_step, end_time_step,
                                                os.path.join(forecast_window_dir, "Series_truth.txt"),
                                                include_values=True)

        # Naive prediction for current forecast window, WIP
        predicate_constructors.naive_prediction_predicate(agg_series, series_ids, window_idx + int(initial_segment_size/window_size), window_idx + int(initial_segment_size/window_size) + 1, window_size, os.path.join(forecast_window_dir, "NaiveBaseline_obs.txt"))


        # AR Baseline; not used in model, but used for evaluation.
        predicate_constructors.ar_baseline_predicate(generated_series, coefs_and_biases, series_ids, [series[int((initial_segment_size + window_size)/window_size - 1 + window_idx)] for series in agg_series], 0, start_time_step - 1, window_size, os.path.join(forecast_window_dir, "ARBaseline_obs.txt"), os.path.join(forecast_window_dir, "ARBaselineAdj_obs.txt"))
        predicate_constructors.fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map,
                                                        cluster_forecasts, cluster_size, initial_segment_size,
                                                        initial_segment_size + (window_idx * window_size), initial_segment_size + ((window_idx + 1) * window_size) - 1,
                                                        os.path.join(forecast_window_dir, "ARBaselineFP_obs.txt"))

        predicate_constructors.cluster_equal_bias_ar_forecasts_predicate(generated_series, coefs_and_biases,
                                                                         cluster_forecasts, cluster_size,
                                                                         0, start_time_step - 1, initial_segment_size,
                                                                         window_size,
                                                                         os.path.join(forecast_window_dir,
                                                                                      "ARBaselineNaiveTD_obs.txt"))
        if batch:
            open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
                command_constructor.create_forecast_window_commands(generated_series, noisy_series, series_ids, cluster_ids, start_time_step, end_time_step, window_size, window_idx,
                                                                    int(np.rint(initial_segment_size / window_size)) + window_idx, cluster=cluster_hierarchy))
        else:
            open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
                command_constructor.iterative_forecast_commands(generated_series, series_ids, cluster_ids,
                                                                    start_time_step, end_time_step, window_size,
                                                                    window_idx,
                                                                    int(np.rint(
                                                                        initial_segment_size / window_size)) + window_idx))

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
def gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, model_name, lags, temporal_hierarchical_rule_weight=1.0, cluster_hierarchical_rule_weight=1.0,
                  temporal_rules=True, cluster_rules=False, cluster_hard=False, mean_hard=False, series_mean_prior=False, series_mean_prior_weight=1.0, series_mean_prior_squared=True):
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
            hts_model_lines += str(coef) + " * Series(S, T_Lag" + str(idx + 1) + ") + "

        hts_model_lines += "0.0 * Lags(T, " + ", ".join(["T_Lag" + str(lag) for lag in lags]) + ") + "
        hts_model_lines += str(bias) + "\n"

    if series_mean_prior:
        hts_model_lines += str(series_mean_prior_weight) + ": Series(S, T) = SeriesMean(S)"

        if series_mean_prior_squared:
            hts_model_lines += " ^2"

        hts_model_lines += "\n"

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
   SeriesMean/1: closed
   Lags/"""

    hts_data_lines += str(len(lags) + 1) + ": closed\n"

    hts_data_lines += """observations:\n"""
    for lag in lags:
         hts_data_lines += "   Lag" + str(lag) + ":   ../data/hts/eval/000/Lag" + str(lag) + "_obs.txt \n"

    hts_data_lines +=  """   Series: ../data/hts/eval/000/Series_obs.txt
   IsInWindow: ../data/hts/eval/000/IsInWindow_obs.txt
   SeriesBlock: ../data/hts/eval/000/SeriesBlock_obs.txt
   OracleSeries: ../data/hts/eval/000/OracleSeries_obs.txt
   SeriesCluster: ../data/hts/eval/000/SeriesCluster_obs.txt
   ClusterOracle: ../data/hts/eval/000/ClusterOracle_obs.txt
   SeriesMean: ../data/hts/eval/000/SeriesMean_obs.txt
   Lags: ../data/hts/eval/000/Lags_obs.txt

truth:
   Series: ../data/hts/eval/000/Series_truth.txt
    
targets:
   Series: ../data/hts/eval/000/Series_target.txt
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

def set_up_experiment(exp_name, series_count, p, cluster_size, post_noise_var, err_means, err_cov_matrix, forecast_variance_scale, init_segment_size, num_windows, experiment_name_dir, temp_oracle_variance, window_size, initial_segment_size, series_length, batch=False):
    lags = [l + 1 for l in np.arange(p)]
    generated_series, coefs = generate_dataset(series_count, cluster_size, p, init_segment_size + (num_windows*window_size) + DISCARDED_SEGMENT_LENGTH, err_means, err_cov_matrix, window_size, seed=1234)

    # All off-diagonal elements (cross-series covariances) are equal
    cross_cov = err_cov_matrix[0][1]

    noisy_series = copy.deepcopy(generated_series)

    for series_idx in range(series_count):
        noisy_series[series_idx] = noisy_series[series_idx] + np.random.normal(0, post_noise_var, noisy_series.shape[1])

    for cluster_idx in range(int(series_count / cluster_size)):
        noisy_and_original_series = np.concatenate(((generated_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size]), (noisy_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size])), axis=0)
        noisy_and_original_series_norm = normalize_multiple(noisy_and_original_series)

        generated_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size] = noisy_and_original_series_norm[:cluster_size]
        noisy_series[cluster_idx * cluster_size:(cluster_idx + 1) * cluster_size] = noisy_and_original_series_norm[cluster_size:]

    mase_scale_factor_lines = ""

    for idx, series in enumerate(generated_series):
        mase_scale_factor_lines += str(idx) + "\t" + str(np.abs(np.diff( series[:initial_segment_size])).sum()/(initial_segment_size - 1)) + "\n"

    open("mase_scale.txt", "w").write(mase_scale_factor_lines)

    coefs_and_biases = fit_ar_models(noisy_series, 0, initial_segment_size-1, p)

    #print(coefs_and_biases)

    experiment_dir = os.path.join(DATA_PATH, experiment_name_dir, "eval")
    forecast_window_dirs = [str(window_idx).zfill(3) for window_idx in range(num_windows)]

    build_psl_data(generated_series, noisy_series, coefs_and_biases, forecast_variance_scale, temp_oracle_variance, lags,
                   num_windows, experiment_dir, forecast_window_dirs, window_size, initial_segment_size, series_length,
                   cluster_hierarchy=True, cluster_size=cluster_size, batch=batch)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_10", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_5", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=5.0,
                  cluster_rules=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "temporal_hard", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=False, mean_hard=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "temporal_hard_meanprior0.1_nsq", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=False, mean_hard=True, series_mean_prior=True, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior1", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=0.1)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior0.1_nsq", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=0.1, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_100_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=100.0,
                  cluster_rules=True, cluster_hard=False, series_mean_prior=True, series_mean_prior_weight=0.1)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_10_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=False, series_mean_prior=True, series_mean_prior_weight=0.1)


    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_5_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=5.0,
                  cluster_rules=True, cluster_hard=False, series_mean_prior=True, series_mean_prior_weight=0.1)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior0.1_nonsq", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=0.1, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior0.2", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=0.2)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior0.01", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=0.01)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_tw_hard_ns_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, mean_hard=True, series_mean_prior=True, series_mean_prior_weight=0.1, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_tw_10_meanprior0.1", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, mean_hard=False, series_mean_prior=True, series_mean_prior_weight=0.1, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_meanprior10", lags,
                  temporal_hierarchical_rule_weight=0.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, series_mean_prior=True, series_mean_prior_weight=10)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_hard_combined", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=True, mean_hard=True)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "tw_hard_l1_prior", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=0.0,
                  cluster_rules=False, cluster_hard=False, mean_hard=True, series_mean_prior=True, series_mean_prior_weight=0.1, series_mean_prior_squared=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_10_tw_10", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=False, mean_hard=False)

    gen_hts_model(generated_series, coefs_and_biases, experiment_name_dir, "cw_10_tw_10_meanprior", lags,
                  temporal_hierarchical_rule_weight=10.0,
                  cluster_hierarchical_rule_weight=10.0,
                  cluster_rules=True, cluster_hard=False, mean_hard=False, series_mean_prior=True)

    options_file_handle = open(os.path.join(DATA_PATH, experiment_name_dir, "options.txt"), "w")
    options_file_lines = "p\t" + str(p) + "\nwindow_size\t" + str(window_size) + \
                         "\nSeed\t" + str(SEED) + "\ncluster_forecast_noise_variance\t" + str(forecast_variance_scale) + "\nCluster_size\t" + str(cluster_size) + "\n"
    options_file_handle.write(options_file_lines)

def generate_dataset(series_count, cluster_size, p, n, means, e_cov_matrix, window_size, seed=1234):
    coefs = generate_cluster_coefs(cluster_size, p, window_size, enforce_stationarity=True)
    series_list = generate_ar_cluster(cluster_size, n, coefs, means, e_cov_matrix, seed)

    all_coefs = coefs

    while series_list.shape[0] < series_count:
        seed += 1

        coefs = generate_cluster_coefs(cluster_size, p, window_size, seed=seed)
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
    series_count = 120
    cluster_size = 4

    p = 4


    post_noise_vars = [0, 0.5, 1]
    cross_covs = [0]
    forecast_variance_scales = [1]
    temp_or_variances = [0]
    forecast_window_sizes = [p]
    initial_segment_size = 1000

    num_forecast_windows = 30

    #exp_name = "E1_p" + str(p)
    exp_name = "E1"

    exp1_dirs = ""

    for forecast_variance_scale in forecast_variance_scales:
        for cross_cov in cross_covs:
            for temp_or_variance in temp_or_variances:
                for forecast_window_size in forecast_window_sizes:
                        for post_noise_var in post_noise_vars:
                            err_means = np.zeros(cluster_size)
                            err_cov_matrix = np.full((cluster_size, cluster_size), cross_cov)
                            np.fill_diagonal(err_cov_matrix, 1)

                            initial_segment_size = 1000
                            initial_segment_size += (forecast_window_size - (initial_segment_size % forecast_window_size))

                            series_length = initial_segment_size + (num_forecast_windows) * forecast_window_size

                            exp_dir = os.path.join(exp_name, "base_noise_" + str(post_noise_var), "clus_or_variance_" + str(forecast_variance_scale), "cross_cov_" + str(cross_cov), "temp_or_variance_" + str(temp_or_variance), "window_size_" + str(forecast_window_size))
                            exp1_dirs += exp_dir + " "
                            set_up_experiment(exp_name, series_count, p, cluster_size, post_noise_var, err_means, err_cov_matrix, forecast_variance_scale,
                                      initial_segment_size, num_forecast_windows, exp_dir, temp_or_variance, forecast_window_size, initial_segment_size, series_length, batch=True)

    open(os.path.join(DATA_PATH, "e1_data_dirs.txt"), "w").write(exp1_dirs)
    exit(1)


    p = 4

    cross_covs = [0]
    forecast_variance_scales = [1]
    temp_or_variances = [0, 0.25, 0.5, 0.75, 1.0]
    forecast_window_sizes = [p, 3 * p]
    initial_segment_size = 1000

    exp_name = "E2_p" + str(p)

    for forecast_variance_scale in forecast_variance_scales:
        for cross_cov in cross_covs:
            for temp_or_variance in temp_or_variances:
                for forecast_window_size in forecast_window_sizes:
                    # Experiment 2 setup,
                    err_means = np.zeros(cluster_size)
                    err_cov_matrix = np.full((cluster_size, cluster_size), cross_cov)
                    np.fill_diagonal(err_cov_matrix, 1)

                    initial_segment_size += (forecast_window_size - (initial_segment_size % forecast_window_size))

                    series_length = initial_segment_size + (num_forecast_windows) * forecast_window_size

                    exp_dir = os.path.join(exp_name, "clus_or_variance_" + str(forecast_variance_scale), "cross_cov_" + str(cross_cov), "temp_or_variance_" + str(temp_or_variance), "window_size_" + str(forecast_window_size))
                    set_up_experiment(exp_name, series_count, p, cluster_size, err_means, err_cov_matrix, forecast_variance_scale,
                              initial_segment_size, num_forecast_windows, exp_dir, temp_or_variance, forecast_window_size, initial_segment_size, series_length, batch=True)

if __name__ == '__main__':
    main()
