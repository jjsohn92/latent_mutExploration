#!/bin/bash 

dest=$1 # ../output/evaluation/reveal_pred/v3
for seed in {0..9}
do 
    echo $seed
    python3.9 rf_model.py -p all -d $dest -s $seed -rd 2> ../logs/mdl/new.rd.$seed.err 1> ../logs/mdl/new.rd.$seed.out
done
