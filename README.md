# latent_mutExploration

- D4J_HOME: path to the local defects4j repository: https://github.com/rjust/defects4j

- result_analysis: RQ1-5 evaluation scripts 
- results directory: contain obtained data 
    ```
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
