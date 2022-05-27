import os
import sys

INFERRED_PREDICATE_FILE_NAME = "SERIES.txt"
TRUTH_PREDICATE_FILE_NAME = "Series_truth.txt"
AR_BASELINE_FILE_NAME = "ARBaseline_obs.txt"

def absolute_error(x, y):
    return abs(x - y)

def main():
    truth_dir = sys.argv[1]
    res_dir = sys.argv[2]

    abs_error_psl = 0
    abs_error_ar = 0
    abs_error_count = 0

    for fold_dir in os.listdir(res_dir):
        if not os.path.isdir(os.path.join(res_dir, fold_dir)):
            continue

        result_fold_dir = os.path.join(res_dir, fold_dir)
        truth_fold_dir = os.path.join(truth_dir, fold_dir)

        truth_lines = open(os.path.join(truth_fold_dir, TRUTH_PREDICATE_FILE_NAME), "r").readlines()
        result_lines = open(os.path.join(result_fold_dir, INFERRED_PREDICATE_FILE_NAME), "r").readlines()
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

            truth_dict[series_id][timestep] = float(val)

        for line in result_lines:
            tokens = line.split("\t")
            series_id = tokens[0]
            timestep = tokens[1]
            val = tokens[2].rstrip()

            if series_id not in result_dict:
                result_dict[series_id] = dict()

            result_dict[series_id][timestep] = float(val)

        for line in ar_baseline_lines:
            tokens = line.split("\t")
            series_id = tokens[0]
            timestep = tokens[1]
            val = tokens[2].rstrip()

            if series_id not in ar_baseline_dict:
                ar_baseline_dict[series_id] = dict()

            ar_baseline_dict[series_id][timestep] = float(val)

        for series in truth_dict.keys():
            for timestep in truth_dict[series].keys():
                abs_error_psl += absolute_error(truth_dict[series_id][timestep], result_dict[series_id][timestep])
                abs_error_ar += absolute_error(truth_dict[series_id][timestep], ar_baseline_dict[series_id][timestep])
                abs_error_count += 1

    print("MAE PSL: " + str(float(abs_error_psl / abs_error_count)))

    print("MAE AR: " + str(float(abs_error_ar / abs_error_count)))

if __name__ == '__main__':
    main()
