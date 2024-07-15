import os
import pandas as pd 
import sys 
from typing import List, Tuple
sys.path.append("../")

MUTOPS = ['MATH', 'CONDITIONALS_BOUNDARY', 'INCREMENTS', 'INVERT_NEGS', 'NEGATE_CONDITIONALS',
    'OBBN1',
    'INLINE_CONSTS',
    'CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6',
    'ROR1', 'ROR2', 'ROR3', 'ROR4', 'ROR5',
    'AOR1', 'AOR2', 'AOR3', 'AOR4',
    'EMPTY_RETURNS', 'FALSE_RETURNS', 'TRUE_RETURNS', 'NULL_RETURNS', 'PRIMITIVE_RETURNS',
    'VOID_METHOD_CALLS', 'CONSTRUCTOR_CALLS', 'REMOVE_INCREMENTS',
    'AOD_1', 'AOD_2', 'OBBN2', 'OBBN3']

D4J_HOME = os.getenv("D4J_HOME")
PROJECT_REPO_DIR = os.path.join(D4J_HOME, "project_repos")
PROJECT_REPOS = {
    'Lang':os.path.join(PROJECT_REPO_DIR, 'commons-lang.git'),
    'Math':os.path.join(PROJECT_REPO_DIR, 'commons-math.git'),
    'Time':os.path.join(PROJECT_REPO_DIR, 'joda-time.git'),
    'Closure':os.path.join(PROJECT_REPO_DIR, 'closure-compiler.git'),
    'Chart':os.path.join(PROJECT_REPO_DIR, 'jfreechart'),
    'Mockito':os.path.join(PROJECT_REPO_DIR, 'mockito.git'),
    'Cli':os.path.join(PROJECT_REPO_DIR, 'commons-cli.git'),
    'Codec':os.path.join(PROJECT_REPO_DIR, 'commons-codec.git'),
    'Collections':os.path.join(PROJECT_REPO_DIR, 'commons-collections.git'),
    'Compress':os.path.join(PROJECT_REPO_DIR, 'commons-compress.git'),
    'Csv':os.path.join(PROJECT_REPO_DIR, 'commons-csv.git'),
    'Gson':os.path.join(PROJECT_REPO_DIR, 'gson.git'),
    'JacksonCore':os.path.join(PROJECT_REPO_DIR, 'jackson-core.git'),
    'JacksonDatabind':os.path.join(PROJECT_REPO_DIR, 'jackson-databind.git'),
    'JacksonXml':os.path.join(PROJECT_REPO_DIR, 'jackson-dataformat-xml.git'),
    'Jsoup':os.path.join(PROJECT_REPO_DIR, 'jsoup.git'),
    'JxPath':os.path.join(PROJECT_REPO_DIR, 'commons-jxpath.git')
}

def get_feature_cols(feature_col_file:str) -> List[str]:
  with open(feature_col_file) as f:
    features = [l.strip() for l in f.readlines() if bool(l.strip())]
  return features 

def get_data(feature_dir:str, project:str) -> pd.DataFrame:
  import glob 
  dfs = []
  for feature_file in glob.glob(os.path.join(feature_dir,f"{project}_*.chg_features.json")):
    bid = int(os.path.basename(feature_file).split(".")[0].split("_")[1])
    df = pd.read_json(feature_file)
    if len(df) == 0: print (feature_file); continue
    df['bid'] = [int(bid)] * len(df)
    df['mutOp'] = df.mutOp.apply(lambda v:MUTOPS.index(v)).values  # for later 
    dfs.append(df)
  merged_feature_df = pd.concat(dfs, ignore_index=True)
  return merged_feature_df

def get_gts(gtdir:str, project:str, gt_col:str, thr:int) -> pd.DataFrame:
  gt_file = os.path.join(gtdir, f"{project}.indv_mut_propagation_status_and_debt.csv")
  gts = pd.read_csv(gt_file)
  gts = gts[~(gts.status == 'nowhere')]
  gts['mutOp'] = gts.mutOp.apply(lambda v:MUTOPS.index(v)).values
  # set 
  gts['label'] = gts[gt_col].values
  return gts 

def get_gts_for_reveal(gtdir:str, project:str, gt_col:str, thr:int) -> pd.DataFrame:
  gt_file = os.path.join(gtdir, f"{project}.indv_mut_propagation_status_and_debt.csv")
  gts = pd.read_csv(gt_file)
  gts = gts[~(gts.status == 'nowhere')]
  gts['mutOp'] = gts.mutOp.apply(lambda v:MUTOPS.index(v)).values # for later 
  
  # set ground truth
  ## revealed: take only thos revealed within the threshold 
  reveal_df = gts.loc[gts.status == 'reveal'] 
  reveal_df = reveal_df[reveal_df[gt_col] <= thr]
  reveal_df['label'] = 0

  ## survivied: take only those surived more than the threshold
  surv_df = gts.loc[gts.status == 'surv']
  surv_df = surv_df[surv_df[gt_col] > thr] # 
  surv_df['label'] = 1 
  
  ## dead: take all
  dead_df = gts.loc[gts.status == 'dead'][:]
  dead_df['label'] = 2

  ## combine 
  combined_df = pd.concat([surv_df, reveal_df, dead_df], ignore_index=True)
  #print (len(surv_df) + len(reveal_df) + len(dead_df))
  return combined_df
