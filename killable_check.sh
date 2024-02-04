#!/bin/bash

export D4J_HOME=/Users/jeongju.sohn/workdir/tools/defects4j

projects=('Lang' 'Math' 'Time' 'Compress' 'Cli' 'Codec' 'Gson' 'Closure' 'Collections' 'Jsoup' 'Csv' 'Mockito' 'JxPath' 'JacksonXml' 'JacksonCore' 'JacksonDatabind')
for project in ${projects[@]}; do 
    python3.9 check_killable_atIntro.py $project -2> logs/killable/$project.err 1> logs/killable/$project.out 
done 
