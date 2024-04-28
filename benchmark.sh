#!/bin/sh

RELATIVE_FILE_PATH=$1
NUM_TRIALS=$(($2))
# NUM_PROCESSORS=$(($(nproc --all)/2))
# NUM_PROCESSORS=$(($(sysctl -n hw.ncpu)/2))
NUM_PROCESSORS=4
TRIALS_PER_PROCESS=$((NUM_TRIALS/NUM_PROCESSORS))
TRIALS_REMAINDER=$((NUM_TRIALS%NUM_PROCESSORS))
CURRENT_TIME=$(date | sed 's/ //g' | sed 's/:/_/g')

# WORKING_DIR=$(PWD)

echo "Total trials    : $NUM_TRIALS"
echo "Processors      : $NUM_PROCESSORS"
echo "Trials Per Proc : $TRIALS_PER_PROCESS"
echo "Trials remainder: $TRIALS_REMAINDER"
echo "Date            : $CURRENT_TIME"
echo "Python file     : $RELATIVE_FILE_PATH"

SEED_VALUE=$(date +%s)
SEED_VALUE_I=1000

PYTHONPATH="$(pwd)"
HOST_NAME="$(hostname)"

echo "STARTING_SEED: $SEED_VALUE"
echo "PYTHONPATH: $PYTHONPATH"
echo "HOST_NAME: $HOST_NAME"

export PYTHONPATH

# Start first process
echo "Starting process 1"

nohup python $RELATIVE_FILE_PATH \
  --seed "$SEED_VALUE$HOST_NAME" \
  --trials $(( $TRIALS_PER_PROCESS + $TRIALS_REMAINDER)) \
  > "./log/Process_${HOST_NAME}_1_${CURRENT_TIME}.out" \
  2> "./log/Process_${HOST_NAME}_1_${CURRENT_TIME}.err" &

# Start other process if processor count > 1
if [ $NUM_PROCESSORS -gt 1 ] && [ $TRIALS_PER_PROCESS -gt 0 ]
  then
    p=2

    while [ $p -le $NUM_PROCESSORS ]; do
      echo "Starting process $p"

      nohup python $RELATIVE_FILE_PATH \
        --seed "$(( $SEED_VALUE + $SEED_VALUE_I * $p ))$HOST_NAME" \
        --trials $TRIALS_PER_PROCESS \
        > "./log/Process_${HOST_NAME}_${p}_${CURRENT_TIME}.out" \
        2> "./log/Process_${HOST_NAME}_${p}_${CURRENT_TIME}.err" &

      p=$(( $p + 1 ))
    done
  else
    echo "Not starting other process"
fi
