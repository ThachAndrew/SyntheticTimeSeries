import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

INFERRED_PREDICATE_FILE_NAME = "SERIES.txt"
TRUTH_PREDICATE_FILE_NAME = "Series_truth.txt"
AR_BASELINE_FILE_NAME = "ARBaseline_obs.txt"
CLUSTER_BASELINE_NAME = "ARBaselineNaiveTD_obs.txt"

OUT_FILE_NAME = "ar_vs_psl_metrics.tsv"

METRICS = ["MAE", "MedAE", "MSE", "Corr", "R2", "SMAPE", "MASE"]

def absolute_error(x, y):
    return abs(x - y)

def means_and_sig_test(results_df, metric, mean=True):
    psl_window_vals = []
    ar_window_vals = []

    if mean:
        skipped = []
        for name, group in results_df[results_df["Method"] == "PSL"].groupby(by=["Forecast_Window"]):
            psl_window_vals += [np.mean(group[metric].values)]
        for name, group in results_df[results_df["Method"] == "AR"].groupby(by=["Forecast_Window"]):
            ar_window_vals += [np.mean(group[metric].values)]
    else:
        skipped = []
        for name, group in results_df[results_df["Method"] == "PSL"].groupby(by=["Forecast_Window"]):
            psl_window_vals += [np.median(group[metric].values)]
        for name, group in results_df[results_df["Method"] == "AR"].groupby(by=["Forecast_Window"]):
            ar_window_vals += [np.median(group[metric].values)]

    return np.mean(psl_window_vals), np.std(psl_window_vals), np.mean(ar_window_vals), np.std(ar_window_vals), ttest_rel(psl_window_vals, ar_window_vals)[1]

def mase(a, f, scale):
    return np.mean(np.abs(a - f)) / scale

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def main():
    truth_dir = sys.argv[1]
    res_dir = sys.argv[2]
    out_file = sys.argv[3]

    cluster_baseline = False
    iterative = False

    if len(sys.argv) > 4:
        if "cluster" in sys.argv[4]:
            cluster_baseline = True

        if "iterative" in sys.argv[5]:
            iterative = True

    mase_scale_factors = dict()

    for line in open("mase_scale.txt", "r").readlines():
        tokens = line.rstrip().split("\t")
        mase_scale_factors[tokens[0]] = float(tokens[1])

    abs_error_psl = 0
    abs_error_ar = 0

    abs_error_count = 0

    results_df = pd.DataFrame(columns=["Series_ID", "Forecast_Window", "Method", "MAE", "MedAE", "Corr", "R2"])

    for i in range(15):
        fold_dir = str(i).zfill(3)
        if not os.path.isdir(os.path.join(res_dir, fold_dir)):
            continue

        result_fold_dir = os.path.join(res_dir, fold_dir)
        truth_fold_dir = os.path.join(truth_dir, fold_dir)

        truth_lines = open(os.path.join(truth_fold_dir, TRUTH_PREDICATE_FILE_NAME), "r").readlines()

        if iterative:
            result_lines = []
            timestep_dirs = os.listdir(result_fold_dir)

            for timestep_dir in timestep_dirs:
                if len(result_lines) == 0:
                    result_lines = open(os.path.join(result_fold_dir, timestep_dir, INFERRED_PREDICATE_FILE_NAME)).readlines()
                else:
                    result_lines += open(os.path.join(result_fold_dir, timestep_dir, INFERRED_PREDICATE_FILE_NAME)).readlines()
        else:
            result_lines = open(os.path.join(result_fold_dir, INFERRED_PREDICATE_FILE_NAME), "r").readlines()

        if cluster_baseline:
            ar_baseline_lines = open(os.path.join(truth_fold_dir, CLUSTER_BASELINE_NAME), "r").readlines()
        else:
            ar_baseline_lines = open(os.path.join(truth_fold_dir, AR_BASELINE_FILE_NAME), "r").readlines()


        truth_dict = dict()
        result_dict = dict()
        ar_baseline_dict = dict()

        for line in truth_lines:
            tokens = line.split("\t")
            series_id = tokens[0]
            timestep = tokens[1]
            val = tokens[2].rstrip()

            if series_id not in truth_dict:
                truth_dict[series_id] = dict()

            truth_dict[series_id][timestep] = round(float(val), 5)

        for line in result_lines:
            tokens = line.split("\t")
            series_id = tokens[0]
            timestep = tokens[1]
            val = tokens[2].rstrip()

            if series_id not in result_dict:
                result_dict[series_id] = dict()

            result_dict[series_id][timestep] = round(float(val), 5)

        for line in ar_baseline_lines:
            tokens = line.split("\t")
            series_id = tokens[0]
            timestep = tokens[1]
            val = tokens[2].rstrip()

            if series_id not in ar_baseline_dict:
                ar_baseline_dict[series_id] = dict()

            ar_baseline_dict[series_id][timestep] = round(float(val), 5)

        for series in truth_dict.keys():
            abs_errors_psl = []
            abs_errors_ar = []

            sq_errors_psl = []
            sq_errors_ar = []

            truth_values = []
            predicted_values_psl = []
            predicted_values_ar = []

            for timestep in sorted(list(truth_dict[series].keys())):
                truth_values += [truth_dict[series][timestep]]
                predicted_values_psl += [result_dict[series][timestep]]
                predicted_values_ar += [ar_baseline_dict[series][timestep]]

                ts_abs_error_psl = absolute_error(truth_dict[series][timestep], result_dict[series][timestep])
                ts_abs_error_ar = absolute_error(truth_dict[series][timestep], ar_baseline_dict[series][timestep])

                abs_errors_psl += [ts_abs_error_psl]
                abs_errors_ar += [ts_abs_error_ar]

                sq_errors_psl += [ts_abs_error_psl ** 2]
                sq_errors_ar += [ts_abs_error_ar ** 2]

                abs_error_count += 1



            if np.var(predicted_values_ar) == 0 or np.var(predicted_values_psl) == 0:
                continue

            smape_psl = smape(np.array(truth_values), np.array(predicted_values_psl))
            smape_ar = smape(np.array(truth_values), np.array(predicted_values_ar))

            mase_psl = mase(np.array(truth_values), np.array(predicted_values_psl), mase_scale_factors[series])
            mase_ar = mase(np.array(truth_values), np.array(predicted_values_ar), mase_scale_factors[series])

            corr_psl = np.corrcoef(truth_values, predicted_values_psl)[0][1]
            corr_ar = np.corrcoef(truth_values, predicted_values_ar)[0][1]

            r2_psl = r2_score(truth_values, predicted_values_psl)
            r2_ar = r2_score(truth_values, predicted_values_ar)

            results_df = pd.concat([results_df, pd.DataFrame({"Series_ID": series, "Forecast_Window": fold_dir, "Method": "PSL",
                                                 "MAE": np.mean(abs_errors_psl), "MedAE": np.median(abs_errors_psl), "MSE": np.mean(sq_errors_psl),
                                                              "SMAPE": np.mean(smape_psl), "MASE": np.mean(mase_psl),
                                                              "Corr": corr_psl, "R2": r2_psl}, index=[0])])

            results_df = pd.concat([results_df, pd.DataFrame({"Series_ID": series, "Forecast_Window": fold_dir, "Method": "AR",
                                                 "MAE": np.mean(abs_errors_ar), "MedAE": np.median(abs_errors_ar), "MSE": np.mean(sq_errors_ar),
                                                              "SMAPE": np.mean(smape_ar), "MASE": np.mean(mase_ar),
                                                              "Corr": corr_ar, "R2": r2_ar}, index=[0])])

    metric_cols = []

    for metric in METRICS:
        metric_cols += [metric]
        metric_cols += [metric + "_std"]

    out_file_handle = open(out_file, "w")
    out_file_lines = "Method\t" + "\t".join(metric_cols) + "\n"

    psl_metric_line = "PSL\t"
    ar_metric_line = "AR\t"

    pvals = dict()

    for metric in METRICS:
        mean_psl, std_psl, mean_ar, std_ar, pvalue = means_and_sig_test(results_df, metric)
        pvals[metric] = pvalue
        psl_metric_line += str(mean_psl) + "\t" + str(std_psl) + "\t"
        ar_metric_line += str(mean_ar) + "\t" + str(std_ar) + "\t"

    psl_metric_line += "\n"
    ar_metric_line += "\n"

    out_file_lines += psl_metric_line + ar_metric_line

    out_file_handle.write(out_file_lines)

    pvals_out_file_handle = open(out_file + "_pvals.txt", "w")
    pvals_lines = ""
    for metric in pvals:
        pvals_lines += metric + "\t" + str(pvals[metric]) + "\n"

    pvals_out_file_handle.write(pvals_lines)

if __name__ == '__main__':
    main()
