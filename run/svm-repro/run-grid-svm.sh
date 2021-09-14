#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}/c/Users/Ben/Uni/BA-Thesis"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/graphgym"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/run"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/git-repos/deepsnap"

echo "setting PYTHONPATH to ${PYTHONPATH}"

CONFIG=config
GRID=grid-svm
REPEAT=1
MAX_JOBS=5  # from looking at task manager
SLEEP=1

# echo "invoking baseline algo"
# python main.py --cfg configs/example.yaml --repeat $REPEAT

echo "generating configs..."
# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python ../configs_gen.py --config ${CONFIG}.yaml \
  --grid ${GRID}.txt \
  --out_dir generated-configs
#  --config_budget configs/${CONFIG}.yaml \
# run batch of configs
echo "running configs..."
# Args: config_dir, num of repeats, max jobs running, sleep time
bash ../parallel.sh generated-configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
# bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
# bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

# aggregate results for the batch
# python agg_batch.py --dir results_modularity-base/${CONFIG}_grid_${GRID}
