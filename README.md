# latent_mutExploration

- D4J_HOME: path to the local defects4j repository: https://github.com/rjust/defects4j
- main.py (main execution script)
- result_analysis: RQ1-5 evaluation scripts 
- results directory: contain obtained data 
    ```
    data/*.bids: contain processed bids
    output/evaluation
        └───processed: formatted propagation output 
        │      file111.txt
        │      file112.txt
        └───raw: propagation output 
        |      file021.txt
        |      file022.txt
        └───features: change features
        │      all_muts: change features for all mutants (live, killed, latent, non-latent, discard)
        │      project_bid.chg_features.json: change features of propagated mutants
        └───pred: latent mutant prediction 
               debt_time_thr365/all|rd|wo_mutop: prediction results

* Execution  

1. Install Python 3.8.6 and required modules 
    ```
    python3 -m pip install --user virtualenv
    python3 -m venv env
    . ./env/bin/activate
    python3 -m pip install -r requirements.txt
    ```

2. Clone Defects4J (https://github.com/rjust/defects4j.git) under lib directory and copy the extended build.xml 
    ```
    cp lib/d4j_ext/defects4j.build.ext.xml $D4J_HOME/framework/projects/ 
    ```

3. run mutation testing and propagte
    e.g., run mutation testing on the fixed commit of the bug (Lang-7) and propagte the surviving mutants 
    ```
    sh ./run.sh Lang 7 output temp 1 
    ```
    - output: where the mutation testing (mutation.xml, uniq_mutLRPair_pfile.pkl) 
        and propagation results (appliedAt*, diffMap*, refactorings*, revealedAt*) wil be stored
    - temp: a temporary directory where a given bug-fix commit will be checked out and be investigated for mutant propagation 
    - 1: run propagation (if not given, run only mutation testing) 
