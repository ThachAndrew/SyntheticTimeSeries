import command_constructor
import predicate_constructors
import estimate_AR_coefs
import time_series_noise

import numpy as np
import os

DATA_PATH = "data"
MODEL_PATH = "timeseries_models"
EXPERIMENT_NAME = "test_experiment_t"

NUM_SERIES_GENERATED = 25
VARIANCE_LOWER_PERCENTILE = 40
VARIANCE_UPPER_PERCENTILE = 60
WINDOW_SIZE = 10

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

def generate_sar_phis(ar_phis, sar_phis, P, period):
    phis = np.zeros(max(len(ar_phis), P * period))
    phis[0:len(ar_phis)] = ar_phis
    for x in range(P):
        phis[((x + 1) * period) - 1] = sar_phis[x]

    return phis

def generate_multiple_series(num_series, length, p, P, period, seed, enforce_nonstationarity=True):
    series = np.empty((num_series, length))
    coefs = np.empty((num_series, max(p, P * period)))

    np.random.seed(seed)

    for x in range(num_series):
        var = np.NAN
        while np.isnan(var) or np.isinf(var) or var < MIN_VARIANCE:
            ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
            sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

            all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
            char_polynomial = np.append(1, -1 * all_coefs)[::-1]

            if enforce_nonstationarity:
                while np.max(np.abs(np.roots(char_polynomial))) < 1:
                    ar_coefs = [(x - 0.5) * 2 for x in np.random.rand(p)]
                    sar_coefs = [(x - 0.5) * 2 for x in np.random.rand(P)]

                    all_coefs = generate_sar_phis(ar_coefs, sar_coefs, P, period)
                    char_polynomial = np.append(1, -1 * all_coefs)[::-1]

            coefs[x] = all_coefs
            series[x] = np.array([generate_ar(length, all_coefs, sigma=2)])

            var = np.var(series[x])

    return series, coefs

def build_psl_data(generated_series, coefs_and_biases, cluster_oracle_noise_sigma, oracle_noise_sigma, lags, num_windows, experiment_dir, forecast_window_dirs,
                   cluster_hierarchy=False, cluster_size=5):
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Construct series ID constants
    series_ids = np.arange(len(generated_series))

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
        predicate_constructors.cluster_oracle_predicate(generated_series, cluster_series_map, cluster_oracle_noise_sigma, 0, SERIES_LENGTH - 1, os.path.join(initial_window_dir, "ClusterOracle_obs.txt"))

        # ClusterMean predicate
        predicate_constructors.cluster_mean_predicate(cluster_series_map, INITIAL_SEGMENT_SIZE - (WINDOW_SIZE + 1), SERIES_LENGTH - 1, os.path.join(initial_window_dir, "ClusterMean_target.txt"))

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
    if cluster_hierarchy:
        predicate_constructors.fp_ar_baseline_predicate(base_forecast_series_list, cluster_series_map, cluster_agg_series_list, INITIAL_SEGMENT_SIZE, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1,
                             os.path.join(initial_window_dir, "ARBaselineFP_obs.txt"))

    # AggSeries predicate
    predicate_constructors.agg_series_predicate(series_ids, 0, INITIAL_SEGMENT_SIZE + WINDOW_SIZE - 1, WINDOW_SIZE,
                                                os.path.join(initial_window_dir, "AggSeries_target.txt"))

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
        open(os.path.join(forecast_window_dir, "commands.txt"), "w").write(
            command_constructor.create_forecast_window_commands(generated_series, series_ids, start_time_step, end_time_step, WINDOW_SIZE, window_idx,
                                                                int(np.rint(INITIAL_SEGMENT_SIZE / WINDOW_SIZE)) + window_idx))

# Fits AR models to a list of series
def fit_ar_models(generated_series, start_idx, end_idx, lags):
    coefs_and_biases = dict()

    for series_idx in range(len(generated_series)):
        estimated_ar_coefs, bias = estimate_AR_coefs.fit_AR_model(generated_series[series_idx], start_idx, end_idx,
                                                              lags)
        coefs_and_biases[series_idx] = [estimated_ar_coefs, bias]

    return coefs_and_biases

# Generate hierarchical time series PSL models and data files.
# TODO @Alex: Update with ClusterMean and OracleCluster
def gen_hts_model(generated_series, coefs_and_biases, model_name, lags, temporal_hierarchical_rule_weight=1.0, cluster_hierarchical_rule_weight=1.0, cluster_rules=False):
    if not os.path.exists(os.path.join(MODEL_PATH, str(model_name))):
        os.makedirs(os.path.join(MODEL_PATH, str(model_name)))

    # Generate model with hierarchical and AR rules
    hts_model_file = open(os.path.join(MODEL_PATH, str(model_name), "hts.psl"), "w")
    hts_model_lines = "Series(S, +T) / |T| = AggSeries(S, P). {T: IsInWindow(T, P)}\n" + str(temporal_hierarchical_rule_weight) + ": AggSeries(S, P) = OracleSeries(S, P) ^2\n\n"   

    if cluster_rules:
        hts_model_lines += "Series(+S, T) / |T| = ClusterMean(C, T). {S: SeriesCluster(S, C)} \n"
        hts_model_lines += str(cluster_hierarchical_rule_weight) + ": ClusterMean(C, T) = ClusterOracle(C, T) ^2\n" 

    # Add AR rules
    for series_idx in range(len(generated_series)):
        estimated_ar_coefs, bias = coefs_and_biases[series_idx]

        hts_model_lines += "1.0: Series(S, T) + 0.0 * SeriesBlock(S, '"+ str(series_idx) + "') = "

        for idx, coef in enumerate(estimated_ar_coefs):
            hts_model_lines += str(coef) + " * Series(S, T_Lag" + str(lags[idx]) + ") + 0.0 * Lag" + str(lags[idx]) + "(T, T_Lag" + str(lags[idx]) + ")  + "

        hts_model_lines += str(bias) + "\n"

    hts_model_file.write(hts_model_lines)

    # Generate data file
    hts_data_file = open(os.path.join(MODEL_PATH, str(model_name), "hts-eval.data"), "w")
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

#def set_up_experiment(num_series, series_length, forecast_window_width, order, ar_process_noise_sigma, added_noise_sigma, oracle_noise_sigma, experiment_dir):

def set_up_experiment(p, P, period, add_noise, base_noise_scale, cluster_noise_scale, oracle_noise_scale, experiment_data_dir, experiment_model_name,
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
                                                            seed=SEED, enforce_nonstationarity=False)

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

def main():
    # Experiment 1: Vary noise added to base-level series.
    E1_base_noise_scales = [0.25, 0.5, 0.75, 1.0, 1.25]

    p = 2
    P = 0
    period = 10
    add_noise = True

    E1_cluster_noise_scale = 0
    E1_oracle_noise_scale = 0
    experiment_data_dir = "E1_base_noise"
    experiment_model_name = "E1_base_noise"

    for base_noise_scale in E1_base_noise_scales:
        set_up_experiment(p, P, period, add_noise, base_noise_scale, E1_cluster_noise_scale, E1_oracle_noise_scale,
                          experiment_data_dir + "_" + str(round(base_noise_scale, 3)),
                          experiment_model_name + "_" + str(round(base_noise_scale, 3)),
                          temporal_hierarchy_rule_weight=100.0,
                          cluster_hierarchy_rule_weight=100.0,
                          cluster_hierarchy=True, cluster_size=5)

    p = 2
    P = 0
    period = 10
    add_noise = True

    E2_oracle_noise_scales = [0.25, 0.5, 0.75, 1.0, 1.25]
    E2_base_noise_scale = 0.5
    E2_cluster_oracle_noise_scale = 0

    # Experiment 2: Vary noise added to aggregate series
    experiment_data_dir = "E2_oracle_noise"
    experiment_model_name = "E2_oracle_noise"

    for oracle_noise_scale in E2_oracle_noise_scales:
        set_up_experiment(p, P, period, add_noise, E2_base_noise_scale, E2_cluster_oracle_noise_scale, oracle_noise_scale,
                          experiment_data_dir + "_" + str(round(oracle_noise_scale, 3)),
                          experiment_model_name + "_" + str(round(oracle_noise_scale, 3)),
                          temporal_hierarchy_rule_weight=100.0,
                          cluster_hierarchy_rule_weight=100.0,
                          cluster_hierarchy=False, cluster_size=5)

    p = 2
    P = 0
    period = 10
    add_noise = True

    E3_base_noise_scale = 0.5
    E3_oracle_noise_scale = 0
    E3_cluster_oracle_noise_scale = 0

    experiment_data_dir = "E3_temporal"
    experiment_model_name = "E3_temporal"

    set_up_experiment(p, P, period, add_noise, E3_base_noise_scale, E3_cluster_oracle_noise_scale, E3_oracle_noise_scale,
                      experiment_data_dir + "_" + str(round(E3_base_noise_scale, 3)),
                      experiment_model_name + "_" + str(round(E3_base_noise_scale, 3)),
                      temporal_hierarchy_rule_weight=100.0,
                      cluster_hierarchy_rule_weight=100.0,
                      cluster_hierarchy=False, cluster_size=5)

    experiment_data_dir = "E3_temporal_cluster"
    experiment_model_name = "E3_temporal_cluster"

    set_up_experiment(p, P, period, add_noise, E3_base_noise_scale, E3_cluster_oracle_noise_scale, E3_oracle_noise_scale,
                      experiment_data_dir + "_" + str(round(E3_base_noise_scale, 3)),
                      experiment_model_name + "_" + str(round(E3_base_noise_scale, 3)),
                      temporal_hierarchy_rule_weight=100.0,
                      cluster_hierarchy_rule_weight=100.0,
                      cluster_hierarchy=True, cluster_size=5)

if __name__ == '__main__':
    main()
