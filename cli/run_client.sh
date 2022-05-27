#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

readonly PSL_VERSION="2.3.0-SNAPSHOT"
readonly JAR_PATH="./psl-client-cli-${PSL_VERSION}.jar"
readonly COMMAND_FILE='./stop_command.txt'
readonly SERVER_RESPONSE_OUTPUT='./client_output/stop/serverResponses.txt'

readonly ADDITIONAL_PSL_OPTIONS='--int-ids --onlineClient -D runtimestats.collect=true'
readonly ADDITIONAL_CLIENT_OPTIONS='-D log4j.threshold=TRACE'
readonly MAX_CONNECTION_ATTEMPTS=100
readonly CONNECTION_ATTEMPT_SLEEP_SEC=1

function main() {
   trap exit SIGINT SIGTERM

   # Make sure we can run PSL.
   check_requirements
   fetch_psl

   # Wait for server to drop tmp file
   waitForServer

   # Run PSL
   runClientCommand "$@"
}

function waitForServer() {
  i=0
  while [ $i -le $MAX_CONNECTION_ATTEMPTS ]; do
    if [ "$(echo /tmp/onlinePSLServer*)" != "/tmp/onlinePSLServer*" ]; then
      break
    fi
    sleep $CONNECTION_ATTEMPT_SLEEP_SEC
    ((i++))
  done
}


function runClientCommand() {
   echo "Issueing PSL Client Commands"
   java -jar "${JAR_PATH}" -onlineServerOutput ${SERVER_RESPONSE_OUTPUT} ${ADDITIONAL_CLIENT_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@" < ${COMMAND_FILE}
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run infernce'
      exit 70
   fi
}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download psl jar.'
      exit 10
   fi

   type java > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: java required to run project'
      exit 13
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file located at: '${url}'."
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

# Fetch the jar from a remote or local location and put it in this directory.
# Snapshots are fetched from the local maven repo and other builds are fetched remotely.
function fetch_psl() {
   if [[ $PSL_VERSION == *'SNAPSHOT'* ]]; then
      local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      cp "${snapshotJARPath}" "${JAR_PATH}"
   else
      local remoteJARURL="https://repo1.maven.org/maven2/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      fetch_file "${remoteJARURL}" "${JAR_PATH}" 'psl-jar'
   fi
}

main "$@"

