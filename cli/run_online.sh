#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG
readonly DATA_DIR_ROOT='../data/'
readonly STOP_COMMAND_FILE='./stop_command.txt'

function main() {
   trap cleanup SIGINT SIGTERM

   # Run PSL
   runAll "$@"
}

function runAll() {
   local experiment=$1
   local dataset=$2
   
   local data_dir="${DATA_DIR_ROOT}/${dataset}/eval/"
   # Make sure previous run of server didn't leave tmp file.
   # Get default tmp directory.
   if [ "$(echo /tmp/onlinePSLServer*)" != "/tmp/onlinePSLServer*" ]; then
     rm -r /tmp/onlinePSLServer*
   fi

   echo "Running Online Server"
   ./run_server.sh "${@:3}" > out_server.txt 2> out_server.err &
   local server_pid=$!

   echo "Running Online Client"
   for fold_dir in "${data_dir}"/*; do
     run_client "${fold_dir}/commands.txt" "./client_output/$(basename ${fold_dir})"
   done

   # Stop the server.
   run_client "${STOP_COMMAND_FILE}" "./client_output/stop"

   # The server takes time to write inferred predicates and shut down after stopping.
   echo "Waiting on Online Server: $(date)"
   wait ${server_pid}
   echo "Finished Waiting on Online Server $(date)"
   date
}

function run_client() {
    local command_file=$1
    local output_dir=$2

    mkdir -p "${output_dir}"

    # Set the commands location in run_client.sh.
    sed -i "s@^readonly COMMAND_FILE=.*'\$@readonly COMMAND_FILE='${command_file}'@g" run_client.sh

    # Set the server response location in run_client.sh.
    sed -i "s@^readonly SERVER_RESPONSE_OUTPUT=.*'\$@readonly SERVER_RESPONSE_OUTPUT='${output_dir}/serverResponses.txt'@g" run_client.sh

    # Run the client. The client will wait until the server closes the socket.
    # This happens after the EXIT or STOP action is executed.
    ./run_client.sh "$@" > "${output_dir}"/out_client.txt 2> "${output_dir}"/out_client.err
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
