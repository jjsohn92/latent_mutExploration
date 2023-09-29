import os, sys 
import pandas as pd 
from typing import List, Dict, Tuple, Union
from tqdm import tqdm 
import javalang
from collections import OrderedDict
sys.path.append("../")
from utils import git_utils, analysis_utils
import numpy as np

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
#CHGDIR = "../output/temp/forMutBug"
EXCLUDE_REFACTORING = False

def get_final_line(target:javalang.tree.Node) -> int:
  max_line = 0
  for _, node in target.filter(javalang.tree.Node):
    import contextlib
    with contextlib.suppress(TypeError):
      max_line = max(max_line, node.position[0])
  return max_line 

def get_full_mth_or_cons_id(
  decl_node:Union[javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration]) -> str:
  params = ",".join([p.type.name for p in decl_node.parameters])
  return f"{decl_node.name}({params})"

def get_lnos_dict(
  decl_nodes:List[Union[javalang.tree.MethodDeclaration, 
  javalang.tree.ConstructorDeclaration]], cls_name:str = None) -> Dict[str, List[int]]:
  lnos_dict = {}
  for _, decl_node in decl_nodes:
    start_lno = decl_node.position[0]
    end_lno = get_final_line(decl_node)
    k = get_full_mth_or_cons_id(decl_node)
    if cls_name is not None:
      k = f"{cls_name}#{k}"
    lnos_dict[k] = np.arange(start_lno, end_lno + 1, 1)
  return lnos_dict

def parse_file(file_content:str) -> Tuple[Dict, Dict]:
  """
  return a method dictonary and constructor dictionary
  """
  tree = javalang.parse.parse(file_content)
  pkg_name = tree.package.name
  cls_lnos_dict = OrderedDict()
  cons_lnos_dict, mth_lnos_dict = {}, {}
  for _, cls_node in tree.filter(javalang.tree.ClassDeclaration):
    cls_name = cls_node.name 
    cls_start_lno = cls_node.position[0]
    cls_end_lno = get_final_line(cls_node)
    outer_cls_name = ""
    for prev_cls_name, prev_cls_info in cls_lnos_dict.items():
      prev_cls_start_lno, prev_cls_end_lno = prev_cls_info['pos']
      if (prev_cls_end_lno >= cls_end_lno) and (prev_cls_start_lno <= cls_start_lno):  
        outer_cls_name = prev_cls_name # the last one 
    if bool(outer_cls_name):
      cls_name = f"{outer_cls_name}${cls_name}"
    cls_lnos_dict[cls_name] = {'pos':(cls_start_lno, cls_end_lno)}
    if bool(pkg_name):
      full_cls_name = f"{pkg_name}.{cls_name}"
    else:
      full_cls_name = cls_name
    # method 
    mth_nodes = cls_node.filter(javalang.tree.MethodDeclaration)
    _mth_lnos_dict = get_lnos_dict(mth_nodes, cls_name = full_cls_name)
    mth_lnos_dict.update(_mth_lnos_dict)
    # constructor 
    cons_nodes = cls_node.filter(javalang.tree.ConstructorDeclaration)
    _cons_lnos_dict = get_lnos_dict(cons_nodes, cls_name = full_cls_name)
    cons_lnos_dict.update(_cons_lnos_dict)
  return mth_lnos_dict, cons_lnos_dict

def revert_k_to_lnos_dict(lnos_dict) -> Dict[int, str]:
  reverted_dict = {}
  for k, lnos in lnos_dict.items():
    for lno in lnos:
      reverted_dict[lno] = k 
  return reverted_dict

def get_mut_status_loc_infos(outputdir:str, project:str, bid:int) -> pd.DataFrame:
  mutfile = os.path.join(outputdir, f"{project}.indv_mut_propagation_status.csv")
  df = pd.read_csv(mutfile) # bid,mutK,status,mutOp,lno 
  df = df[df.bid == bid]
  # file 
  mutKs, fpaths, mutNos = [], [],[]
  for mutK in df.mutK.values: 
    mutKs.append(mutK)
    mutK = analysis_utils.reverse_format_mutK(mutK)
    fpath, mutNo = mutK.split("-")
    mutNo = int(mutNo)
    fpaths.append(fpath)
    mutNos.append(mutNo)
  df['mutK'] = mutKs
  df['fpath'] = fpaths
  df['mutNo'] = mutNos
  return df

def get_chg_features(repo, repo_path:str, rev:str, fpath:str, start_lno:int, end_lno:int) -> Dict[str,float]:
  modifiedAts = git_utils.getModifiedAts_v2(repo_path, rev, fpath, start_lno, end_lno)
  modifiedAts = list(modifiedAts.keys())
  # churn 
  churn = len(modifiedAts)
  # age
  curr_dtime = git_utils.getCommitedDateTime(repo, rev) 
  last_modifiedAt = modifiedAts[0]
  last_mod_dtime = git_utils.getCommitedDateTime(repo, last_modifiedAt)
  first_modifiedAt = modifiedAts[-1]
  first_mod_dtime = git_utils.getCommitedDateTime(repo, first_modifiedAt)
  min_age = (curr_dtime - last_mod_dtime).total_seconds()/(60*60*24)
  max_age = (curr_dtime - first_mod_dtime).total_seconds()/(60*60*24)
  # number of different developers
  n_authors = len(set([git_utils.getAuthor(repo, modifiedAt) for modifiedAt in modifiedAts]))
  # gather 
  chg_features = {'churn':churn, 'min_age':min_age, 'max_age':max_age, 'n_authors':n_authors}
  return chg_features

def main(project:str, bid:int, outputdir:str) -> pd.DataFrame:
  bidToRev, _ = analysis_utils.getBidRevDict(D4J_HOME, project)
  fixedRev = bidToRev[bid]
  repo_path = PROJECT_REPOS[project]
  repo = git_utils.get_repo(repo_path)
  init_mut_info = pd.read_csv(os.path.join(outputdir, f'{project}.init_pit_indv_mut_status.csv'))
  init_mut_info_pbid = init_mut_info.loc[init_mut_info.bid == bid]
  init_mut_info_pbid = init_mut_info_pbid.loc[init_mut_info_pbid.status.isin(['KILLED', 'SURVIVED'])]
  #mutinfo_df = get_mut_status_loc_infos(outputdir, project, bid)
  #if len(mutinfo_df) == 0:
  #  print (f"No mut for {project} {bid}")
  #  return None
  # filter those already processed 
  #if prev_feature_df is not None:
  #  processed_mutKs = prev_feature_df.mutK.values.tolist()
  #  mutinfo_df = init_mut_info.loc[~mutinfo_df.mutK.isin(processed_mutKs)]
  #
  rows = []
  for fpath, df in init_mut_info_pbid.groupby('full_sourceFile'):
    file_content = git_utils.show_file(fixedRev, fpath, repo_path)
    mth_lnos_dict, cons_lnos_dict = parse_file(file_content)
    rvt_mth_lnos_dict = revert_k_to_lnos_dict(mth_lnos_dict)
    rvt_cons_lnos_dict = revert_k_to_lnos_dict(cons_lnos_dict)
    
    d_mth_cons_chg_features = {}
    for _, row in tqdm(list(df.iterrows())): # per-lno
      lno = row.lineNumber
      # get lno age and frequecny  
      lno_chg_features = get_chg_features(repo, repo_path, fixedRev, fpath, lno, lno)
      # get the age of the element that lno belongs
      try:
        mth_or_cons_k = rvt_mth_lnos_dict[lno]
        mth_or_cons_lnos_d = mth_lnos_dict[mth_or_cons_k]
      except KeyError:
        try: 
          mth_or_cons_k = rvt_cons_lnos_dict[lno]
          mth_or_cons_lnos_d = cons_lnos_dict[mth_or_cons_k]
        except KeyError: # lno in none of methods and constructors 
          mth_or_cons_k = None 
          mth_or_cons_lnos_d = None 
      # 
      if mth_or_cons_k is not None:
        try:
          mth_or_cons_chg_features = d_mth_cons_chg_features[mth_or_cons_k]
        except KeyError: 
          start_lno, end_lno = min(mth_or_cons_lnos_d), max(mth_or_cons_lnos_d)
          mth_or_cons_chg_features = get_chg_features(repo, repo_path, fixedRev, fpath, start_lno, end_lno)
          d_mth_cons_chg_features[mth_or_cons_k] = mth_or_cons_chg_features
      else:
        mth_or_cons_chg_features = lno_chg_features # the same =
    
      # save
      rows.append([
        row.k, row.mutOp,
        row['index'], row.block,
        row.description, 
        row.status, 
        fpath, lno, mth_or_cons_k, 
        lno_chg_features['churn'], lno_chg_features['min_age'], lno_chg_features['max_age'], 
        lno_chg_features['n_authors'], 
        mth_or_cons_chg_features['churn'], mth_or_cons_chg_features['min_age'], mth_or_cons_chg_features['max_age'], 
        mth_or_cons_chg_features['n_authors']
      ])
  # convert to dataframe 
  ret_df = pd.DataFrame(rows, columns = [
    #'mutK', 'mutOp', 'status', 
    'k', 'mutOp', 'index', 'block', 'description', 'status', 
    'fpath', 'lno', 'mth_or_cons_k', 
    'l_churn', 'l_min_age', 'l_max_age', 'l_n_authors',
    'e_churn', 'e_min_age', 'e_max_age', 'e_n_authors'])
  return ret_df 


if __name__ == "__main__":
  import argparse 
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--project", type = str)
  parser.add_argument("-b", "--bid", type = int)
  parser.add_argument("-d", "--dest", type = str)
  parser.add_argument("-o", "--outputdir", type = str, 
    help = "path to the directory of the final combined data: e.g.,output/evaluation/combined_v2")

  args = parser.parse_args() 
  
  dest = args.dest 
  os.makedirs(dest, exist_ok=True)
  outputdir = args.outputdir 
  project = args.project  
  bid = args.bid 

  destfile = os.path.join(dest, f"{project}_{bid}.chg_features.json")
  if os.path.exists(destfile):
    prev_feature_df = pd.read_json(destfile)
  else:
    prev_feature_df = None
  
  chg_feature_df = main(project, bid, outputdir)
  if chg_feature_df is not None:
    chg_feature_df.to_json(destfile)
    print (f"Save to {destfile}")
  






