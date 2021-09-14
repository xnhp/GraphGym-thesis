#!/usr/bin/env bash

# perform run based on hyperparameters from config file (not grid file)

export PYTHONPATH="${PYTHONPATH}/c/Users/Ben/Uni/BA-Thesis"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/graphgym"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/Uni/BA-Thesis/GraphGym/run"
export PYTHONPATH="${PYTHONPATH}:/c/Users/Ben/git-repos/deepsnap"

EXPERIMENT=svm-repro
CONFIG=config-svm
REPEAT=1

# echo "invoking baseline algo"
python ../main.py --cfg ${CONFIG}.yaml  --repeat $REPEAT

# write plots etc for results
python ../compare_models.py --experiment ${EXPERIMENT} --config ${CONFIG}  --from_config