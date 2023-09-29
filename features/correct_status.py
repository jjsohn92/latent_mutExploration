import os, sys 
import glob 
import pandas as pd 

if __name__ == "__main__":
  import argparse 
  #parser = argparse.ArgumentParser()
  #parser = 
  feature_dir = "../output/evaluation/features_v3"
  gt_dir = "../output/evaluation/combined_v3"

  projects = ['Lang', 'Math', 'Time', 'Closure', 'Cli', 'Compress', 'Codec', 'Collections', 'Csv', 
    'JacksonCore', 'JacksonXml', 'JxPath', 'Jsoup']
  
  for project in projects:
    print (project)
    files = glob.glob(os.path.join(feature_dir, f"{project}_*.chg_features.json"))
    gt_df = pd.read_csv(os.path.join(gt_dir, f"{project}.indv_mut_propagation_status.csv"))
    for file in files:
      bid = int(os.path.basename(file).split(".")[0].split("_")[1])
      feature_df = pd.read_json(file)
      a_gt_df = gt_df.loc[gt_df.bid == bid]
      rows = []
      for _, row in feature_df.iterrows():
        gt_row = a_gt_df.loc[a_gt_df.mutK == row.mutK]
        if (len(gt_row)) == 0: continue
        assert len(gt_row) == 1, gt_row
        row['status'] = gt_row.status.values[0]
        rows.append(row)
      new_feature_df = pd.DataFrame(rows)
      new_feature_df.to_json(file)