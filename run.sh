#!/bin/bash

scriptdir=$(pwd)
export GUMTREE_HOME="${scriptdir}/lib/gumtree-3.0.0/bin"
export RMINER_HOME="${scriptdir}/lib/RefactoringMiner-2.4.0"
export D4J_HOME="${scriptdir}/lib/defects4j"
export ANT_HOME="${scriptdir}/lib/defects4j/major/bin"

#cp "${scriptdir}/lib/d4j_ext/defects4j.build.ext.xml" $D4J_HOME/framework/projects/
project=$1 
bid=$2
dest=$3
workdir=$4 # checkout under this directory
propagate=$5 

if [ ! -z $propagate ]; then 
    python3 main.py -p $project -b $bid -w $workdir -dst $dest  -propagate # only run mutation testing
else
    python3 main.py -p $project -b $bid -w $workdir -dst $dest
fi
