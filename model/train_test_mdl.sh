#!/bin/bash 

dest=$1 # ../output/evaluation/reveal_pred/v3
feature_config=$2 # all, wo_mutop, only_mutop
logdir="logs"
mkdir -p "${logdir}"

for seed in {0..9}
do 
    echo $seed
    python3.9 rf_model.py -p all -d $dest -s $seed -c $feature_config 2> ../$logdir/rf.$seed.$feature_config.err 1> ../$logdir/rf.$seed.$feature_config.out 
    if [[ $feature_config == 'all' ]]; then
      python3.9 rf_model.py -p all -d $dest -s $seed -rd 2> ../$logdir/rd.$seed.err 1> ../$logdir/rd.$seed.out
    fi 
done
