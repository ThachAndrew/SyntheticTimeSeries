#!/usr/bin/env bash

# Run all the experiments.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly MODELS_DIR="${BASE_DIR}/timeseries_models"
readonly RESULTS_DIR="${BASE_DIR}/results"

readonly TIMESERIES_MODELS='temporal_hard cw_hard'

readonly EXPERIMENTS='Online'
readonly DATASETS='E2_basenoise/base_noise_0/clus_or_variance_0/cross_cov_0/temp_or_variance_0/window_size_4 E2_basenoise/base_noise_0.5/clus_or_variance_0/cross_cov_0/temp_or_variance_0/window_size_4 E2_basenoise/base_noise_1/clus_or_variance_0/cross_cov_0/temp_or_variance_0/window_size_4 E2_basenoise/base_noise_1.5/clus_or_variance_0/cross_cov_0/temp_or_variance_0/window_size_4 E2_basenoise/base_noise_2/clus_or_variance_0/cross_cov_0/temp_or_variance_0/window_size_4'

declare -A MODEL_OPTIONS
MODEL_OPTIONS[test_experiment]='-D sgd.learningrate=1.0 -D sgd.maxiterations=4000'

readonly INFERENCE_OPTIONS='-D sgd.extension=ADAM -D sgd.inversescaleexp=1.5 -D inference.initialvalue=ATOM -D partialgrounding.powerset=true -D reasoner.tolerance=1e-9f'
readonly STANDARD_OPTIONS=''

#todo@alex: iterate over dataset - model pairs
# E1: Vary cluster oracle noise over n \in [0, 0.25, 0.5, 0.75, 1]
# p=4, h \in {4, 12}
# cluster_size = 6
# num_series = 120
# PSL rules:
#   no temporal rule
#   hard cluster rules,
#   all autoregressive rules
#   non-squared mean prior (weight 0.1)
# PSL model name E1_p4/clus_or_variance_[n]/cross_cov_0/cw_hard_meanprior0.1
#
# Baselines:
# Compare to naive top-down (oracle aggregates on top layer) AR baseline that equally biases all predictions in a cluster to sum to the given aggregate value

# E2: Just rules that enforce summing to the temporal oracle, varying its added noise over [0, 0.25, 0.5, 0.75, 1]
# p=4, h \in {4, 12}
# PSL Rules:
#   hard temporal constraint
#   no cluster rules
#   all autoregressive rules
#   non-squared mean prior (weight 0.1)
# PSL model name E1_p4/clus_or_variance_0/cross_cov_0/
#
# Baseline:
# Compare to naive top-down-like AR baseline that equally biases all forecasted values to sum to a temporal aggregate given by the oracle.


function run() {
  local model_name=$1

  # Declare paths to output files.
  local out_directory=""
  local out_path=""
  local err_path=""
  local experiment_options=""

  for experiment in ${EXPERIMENTS}; do
    for dataset in ${DATASETS}; do
        local data_dir="data/${dataset}/eval"
        echo "Running PSL ${model_name} for experiment ${experiment} on dataset ${dataset}."

        # Declare paths to output files.
        out_directory="${RESULTS_DIR}/${experiment}/${dataset}/${model_name}"
        out_path="${out_directory}/out.txt"
        err_path="${out_directory}/out.err"
        experiment_options="${MODEL_OPTIONS[${model_name}]} ${STANDARD_OPTIONS} ${INFERENCE_OPTIONS}"

        # cp model and data files to cli directory
        cp "${MODELS_DIR}/${dataset}/${model_name}/hts.psl" "${BASE_DIR}/cli/hts.psl"
        cp "${MODELS_DIR}/${dataset}/${model_name}/hts-eval.data" "${BASE_DIR}/cli/hts-eval.data"

        out_directory="${RESULTS_DIR}/${experiment}/${dataset}/${model_name}"
        if [[ -e "${out_directory}" ]]; then
          echo "Output file already exists, skipping: ${out_directory}"
        else
          mkdir -p ${out_directory}
          pushd . > /dev/null
             cd "${BASE_DIR}/cli"
             sed -i "s@hts/eval/[0-9]\+@${dataset}/eval/000@g" "hts-eval.data"
             cp "./hts.psl" "${out_directory}/hts.psl"
             
             for fold_dir in "${BASE_DIR}/${data_dir}"/*/; do           
                 # Declare paths to output files.
                 out_path="${out_directory}/$(basename ${fold_dir})/out.txt"
                 err_path="${out_directory}/$(basename ${fold_dir})/out.err"

                 mkdir -p "${out_directory}/$(basename ${fold_dir})"

                 # Set the data split.
                 sed -i "s@eval/[0-9]\+/Series_target.txt@eval/$(basename ${fold_dir})/Series_target.txt@g" "hts-eval.data"
                 sed -i "s@eval/[0-9]\+/Series_obs.txt@eval/$(basename ${fold_dir})/Series_obs.txt@g" "hts-eval.data"

                 cp "./hts-eval.data" "${out_directory}/$(basename ${fold_dir})/hts-eval.data"

                 ./run.sh > "${out_path}" 2> "${err_path}"

                 mkdir -p ${out_directory}/inferred-predicates/$(basename ${fold_dir})

                 mv inferred-predicates/*.txt "${out_directory}/inferred-predicates/$(basename ${fold_dir})"
             done
          popd > /dev/null
        fi
      done
  done
}

function default_models() {
  local model_name=$1

  run ${model_name}
}

function main() {
  trap cleanup SIGINT

  for model_name in ${TIMESERIES_MODELS} ; do
      default_models "${model_name}"
  done
}

function cleanup() {
  for pid in $(jobs -p); do
    pkill -P ${pid}
    kill ${pid}
  done
  pkill -P $$
  exit
}

main "$@"
