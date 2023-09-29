#!/bin/bash

project=$1
outputdir="../output/evaluation/combined_v3"
dest="../output/evaluation/features_v3/all_muts"

if [[ "$project" == "all" ]]; then
  #for project in 'Lang' 'Math' 'Time' 'Closure' 'Cli' 'Collections' 'Codec' 'Compress' 'Csv' 'JacksonCore' 'JacksonDatabind' 'JacksonXml' 'Jsoup' 'JxPath'
  for project in 'Lang' 'Math' 'Time' 'Closure' 'Cli' 'Collections' 'Codec' 'Compress' 'Csv' 'JacksonCore' 'JacksonXml' 'Jsoup' 'JxPath'
  do
    #targetFile="../data/targets/toFocus/final/$project.csv"
    targetFile="../data/targets/toFocus/all/$project.csv"
    echo $targetFile
    if [[ -f $targetFile ]]; then
      cat $targetFile | while read target; do
        python3.9 change_based_all.py -p $project -b $target -d $dest -o $outputdir
      done
    else
      echo $targetFile does not exist
    fi
  done
else
  #targetFile="../data/targets/toFocus/final/$project.csv"
  targetFile="../data/targets/toFocus/all/$project.csv"
  echo $targetFile
  if [[ -f $targetFile ]]; then
    cat $targetFile | while read target; do
      if [[ ( $project == 'Math' ) && ( $target -eq 59 ) ]]; then 
        continue 
      fi 
      python3.9 change_based_all.py -p $project -b $target -d $dest -o $outputdir
    done
  else
    echo $targetFile does not exist
  fi
fi
