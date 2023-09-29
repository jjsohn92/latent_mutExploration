import os, sys
import pandas as pd 
from typing import List, Dict

def get_feature_per_project(projects:List[str], featuredir:str, 
  DEV_FEATURE_TYPES:List[str], norm = True, all_muts:bool = False) -> Dict[str, pd.DataFrame]:
  import glob 
  if all_muts:
    pre_columns = ['k', 'mutOp', 'index', 'block', 'description', 'status', 'fpath', 'lno', 'mth_or_cons_k']
  else:
    pre_columns = ['mutOp', 'status', 'fpath', 'lno', 'mth_or_cons_k']
  feature_df_pproj = {}
  for project in projects:
    feature_files = glob.glob(os.path.join(featuredir, f"{project}_*.chg_features.json"))
    feature_dfs = []
    for feature_file in feature_files:
      bid = int(os.path.basename(feature_file).split("_")[1].split(".")[0])
      feature_df = pd.read_json(feature_file)
      if len(feature_df) == 0: continue # empty file
      feature_df['bid'] = [bid] * len(feature_df)
      feature_df['i_status'] = feature_df.status.apply(lambda v:int(v == 'KILLED')).values
      feature_dfs.append(feature_df)
    pdf = pd.concat(feature_dfs, ignore_index=True)
    set(pdf.columns.values.tolist()) - set(DEV_FEATURE_TYPES)
    pdf = pdf[pre_columns + DEV_FEATURE_TYPES]
    # normalise
    if norm:
      pdf[DEV_FEATURE_TYPES] = (
        pdf[DEV_FEATURE_TYPES] - pdf[DEV_FEATURE_TYPES].min()
       )/(pdf[DEV_FEATURE_TYPES].max() - pdf[DEV_FEATURE_TYPES].min())
    pdf['project'] =[project]*len(pdf)
    feature_df_pproj[project] = pdf 
  feature_df_pproj[project] = pdf 
  return feature_df_pproj

def test_noramlity(df:pd.DataFrame, DEV_FEATURE_TYPES:List[str], STATUSES:List[str]) -> Dict[str, List[str]]:
  from scipy.stats import kstest 
  failed_to_reject = {status:[] for status in STATUSES}
  for feature_type in DEV_FEATURE_TYPES:
    for status in STATUSES:
      vs = df.loc[df.status == status][feature_type].values 
      if len(vs) == 0:
        #failed_to_reject[status].append(feature_type)
        continue
      norm_reject = kstest(vs, 'norm').pvalue <= 0.05
      if not norm_reject:
        failed_to_reject[status].append(feature_type)
  
  print (f"Out of {len(DEV_FEATURE_TYPES)} features")
  for status in STATUSES:
    print (f"\t{status}: {len(failed_to_reject[status])} are from normal distribution")
  return failed_to_reject

def compare_distrt(df:pd.DataFrame, DEV_FEATURE_TYPES:List[str], STATUSES:List[str]):
  pass 


def plot_pval_map(pval_arr, xlabels, ylabels, figsize = (8,6), decimal_points = 3, title:str = None, destfile:str = None):
  import matplotlib.pyplot as plt 
  import numpy as np 

  fig, ax = plt.subplots(figsize = figsize)
  im = ax.imshow(pval_arr)

  # Show all ticks and label them with the respective list entries
  ax.set_xticks(np.arange(len(xlabels)))
  ax.set_yticks(np.arange(len(ylabels)))
  ax.set_xticklabels(xlabels)
  ax.set_yticklabels(ylabels)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  for i in range(len(ylabels)):
      for j in range(len(xlabels)):
          text = ax.text(j, i, np.round(pval_arr[i, j], decimals=decimal_points), ha="center", va="center", color="w")

  if title is not None:
    ax.set_title(title)
  fig.tight_layout()
  if destfile is not None:
    fig.savefig(destfile)
  
