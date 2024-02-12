"""
Generate combine data 

For new revealed info directory, we can simply give different output directory and run 

"""
import os, sys
from tqdm import tqdm  
import pandas as pd 
from utils import analysis_utils, parser_utils, git_utils, analysis_utils 
from typing import List, Dict, Tuple
#
from process.tempAddLnoToCovg import getMutLoc_atRevealed, getLnoAtCommit

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

#FILTERED_INCORRECT = True # filter the cases where incorrectly gathered
#if FILTERED_INCORRECT:
  #NEED_TO_FILTER_OPS = ['CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 'INLINE_CONSTS']
  #NEED_TO_FOUCS_OPS = None
#else:
  #NEED_TO_FILTER_OPS = None
  #NEED_TO_FOUCS_OPS = ['CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 'INLINE_CONSTS']
NEED_TO_FILTER_OPS = None 
NEED_TO_FOUCS_OPS = ['MATH','CONDITIONALS_BOUNDARY',
    'INCREMENTS','INVERT_NEGS','NEGATE_CONDITIONALS','VOID_METHOD_CALLS','PRIMITIVE_RETURNS','EMPTY_RETURNS',
    'FALSE_RETURNS','TRUE_RETURNS','NULL_RETURNS',]

MUTOPS = [
  'org.pitest.mutationtest.engine.gregor.mutators.MathMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.IncrementsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.InvertNegsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.VoidMethodCallMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.returns.PrimitiveReturnsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.returns.EmptyObjectReturnValsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.returns.BooleanFalseReturnValsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.returns.BooleanTrueReturnValsMutator',
  'org.pitest.mutationtest.engine.gregor.mutators.returns.NullReturnValsMutator',
  ]    

# record 
def record_mut_propagation_stauts(outputdir:str, project:str,
  targeted_bids:List[int], bidToRev_dict:Dict[str,str], dest:str, check_dir:str = None, 
  regen_uniq:bool = False) -> pd.DataFrame:
  """
  targeted_bids -> a commit 
  This is computed for unique mutants (uniq_mutK)
  """
  to_save_n_mut_file = os.path.join(dest, f"{project}.n_mut_propagation_status.csv")
  to_save_indv_mut_file = os.path.join(dest, f"{project}.indv_mut_propagation_status.csv")
  if os.path.exists(to_save_n_mut_file):
    nmut_status_df = pd.read_csv(to_save_n_mut_file)
  else:
    nmut_status_df = None 
  if os.path.exists(to_save_indv_mut_file):
    indv_mut_status_df = pd.read_csv(to_save_indv_mut_file)
  else:
    indv_mut_status_df = None

  indv_mut_status = []
  nmut_status = [] # columns = [n_all, n_surv, n_revealed, n_dead]
  for bid in targeted_bids:
    fixedRev = bidToRev_dict[bid]
    #if bid != 7: continue
    #print (bid, fixedRev)
    if check_dir is not None:
      check_file = get_valid_invalid_record_file(check_dir, project, bid)
      if os.path.exists(check_file):
        formatted_invalid_mutKs = analysis_utils.get_formatted_muKs_to_invalid(check_file)
        # convert to within target
        _, _, mapped_to_uniq_mutK = analysis_utils.group_same_mutants(outputdir, project, fixedRev, 
          needToFilterOps = NEED_TO_FILTER_OPS, 
          needToFocusOps = NEED_TO_FOUCS_OPS, 
          regen_uniq=regen_uniq) 
        formatted_invalid_mutKs = analysis_utils.get_only_target_uniq_muts(formatted_invalid_mutKs, mapped_to_uniq_mutK)
      else:
        formatted_invalid_mutKs = set()
    else:
      formatted_invalid_mutKs = set()
    ##
    # filter invalid in the previous
    if indv_mut_status_df is not None:
      indv_mut_status_df = indv_mut_status_df.loc[
        ~((indv_mut_status_df.bid == bid) & (indv_mut_status_df.mutK.isin(formatted_invalid_mutKs)))]
    ##
    # to add mutOp and lno information 
    comp_mut_infos = analysis_utils.getCompletedMutInfos(outputdir, project, fixedRev, uniq_muts=False)#True)
    comp_mut_infos = {analysis_utils.format_mutK(k):v for k,v in comp_mut_infos.items()}
    
    indv_mut_status_pbid = analysis_utils.get_mut_refactor_status(
      outputdir,
      project, 
      fixedRev, 
      needToFilterOps = NEED_TO_FILTER_OPS, 
      needToFocusOps = NEED_TO_FOUCS_OPS, 
      formatted_invalid_mutKs = formatted_invalid_mutKs, 
      regen_uniq=regen_uniq
    )
    #
    if indv_mut_status_pbid is None: # failed to be processed to the end
      nmut_status.append([bid, -1, -1, -1, -1, -1]) # in, when not all targeted mutants are processed thus, saved in the final fiels 
    else: # proessed to the end 
      n_status_row = [bid]
      for status_k in ['surv', 'reveal', 'dead', 'nowhere']:
        muts = indv_mut_status_pbid[status_k]
        n_status_row.append(len(muts))
        for mutK in muts:
          mut_info = comp_mut_infos[mutK] 
          mutOp = mut_info['mutOp'][0]
          lno = mut_info['pos'][0]
          indv_mut_status.append([bid, mutK, status_k, mutOp, lno]) 
      n_status_row.append(sum(n_status_row[1:])) # add total muts-> the sum of len(muts -> surv, reveal, dead, nowehre)
      nmut_status.append(n_status_row)
  nmut_status = pd.DataFrame(nmut_status, 
    columns=['bid', 'n_surv', 'n_reveal', 'n_dead', 'n_nowhere', 'n_all'])
  print (nmut_status)
  print (project, nmut_status.n_reveal.sum(), nmut_status.n_all.sum(), nmut_status.bid.unique().shape[0])
  if nmut_status_df is not None:
    nmut_status = nmut_status.set_index('bid')
    nmut_status_df = nmut_status_df.set_index('bid')
    for bid in nmut_status_df.index.values:
      if nmut_status.loc[bid].n_surv >= 0: # not -1 -> if either orignal or ramain is not processed yet, then currenlty skipped
        init_v = nmut_status_df.loc[bid].n_surv
        if init_v >= 0:
          nmut_status_df.loc[bid].n_surv += nmut_status.loc[bid].n_surv
          nmut_status_df.loc[bid].n_reveal += nmut_status.loc[bid].n_reveal
          nmut_status_df.loc[bid].n_dead += nmut_status.loc[bid].n_dead
          nmut_status_df.loc[bid].n_nowhere += nmut_status.loc[bid].n_nowhere,
          nmut_status_df.loc[bid].n_all += nmut_status.loc[bid].n_all
        else:
          nmut_status_df.loc[bid][['n_surv', 'n_reveal', 'n_dead', 'n_nowhere', 'n_all']] = (
            nmut_status.loc[bid].n_surv, nmut_status.loc[bid].n_reveal, 
            nmut_status.loc[bid].n_dead, nmut_status.loc[bid].n_nowhere,
            nmut_status.loc[bid].n_all
          )
    nmut_status_df = nmut_status_df.reset_index()
    nmut_status = nmut_status_df
  # save 
  nmut_status.to_csv(to_save_n_mut_file, index = False)
  
  indv_mut_status = pd.DataFrame(indv_mut_status, columns=['bid', 'mutK', 'status', 'mutOp', 'lno'])
  if indv_mut_status_df is not None:
    indv_mut_status = pd.concat([indv_mut_status_df, indv_mut_status], ignore_index=True) # add
  indv_mut_status = indv_mut_status.drop_duplicates()
  ## checking 
  grouped_indv_muts = indv_mut_status.groupby(['bid', 'mutK'])
  for k,df in grouped_indv_muts:
    if len(df) > 1:
      print (k)
      for _, row in df.iterrows():
        print (row, row.mutK)
      sys.exit()
  assert len(grouped_indv_muts) == len(indv_mut_status), f"{len(grouped_indv_muts)} vs {len(indv_mut_status)}"
  ##
  # save
  indv_mut_status.to_csv(to_save_indv_mut_file, index = False)
  print (len(indv_mut_status))
  return nmut_status, indv_mut_status

# record 
def record_mut_debt_statistics(outputdir:str, project:str,bidToRev_dict:Dict[str,str], dest:str, 
  check_dir:str = None, regen_uniq:bool = False) -> pd.DataFrame:
  """
  """
  import numpy as np
  import glob 
  repo = git_utils.get_repo(PROJECT_REPOS[project])
  def compute_tech_debt(repo, commit_hash_a, commit_hash_b) -> float:
    dt_a = git_utils.getCommitedDateTime(repo, commit_hash_a)
    dt_b = git_utils.getCommitedDateTime(repo, commit_hash_b)
    time_dist = np.abs((dt_a - dt_b).total_seconds()/(60*60*24))
    return time_dist 
  
  def compute_tech_debt_ncommits(all_commits, commit_hash_a, commit_hash_b) -> float:
    def get_c_idx(all_commits, commit_hash) -> int:
      try:
        return all_commits.index(commit_hash)
      except ValueError: 
        for idx,c in enumerate(all_commits):
          if c.startswith(commit_hash) or commit_hash.startswith(c):
            return idx
          
    idx_to_commit_a = get_c_idx(all_commits, commit_hash_a)
    idx_to_commit_b = get_c_idx(all_commits, commit_hash_b)
    return np.abs(idx_to_commit_b - idx_to_commit_a)
  
  mut_propgate_status_and_debt_file = os.path.join(dest, f"{project}.indv_mut_propagation_status_and_debt.csv")
  if os.path.exists(mut_propgate_status_and_debt_file):
    prev_mut_status_debt_df = pd.read_csv(mut_propgate_status_and_debt_file)
  else:
    prev_mut_status_debt_df = None

  mut_propgate_status_file = os.path.join(dest, f"{project}.indv_mut_propagation_status.csv")
  if not os.path.exists(mut_propgate_status_file): return None 
  mut_status_df = pd.read_csv(mut_propgate_status_file) 
  #n_org_muts = len(mut_status_df)
  ## filter those already processed 
  if prev_mut_status_debt_df is not None:
    n_prev = len(mut_status_df)
    processed = set([f"{bid}-{mutK}" for bid,mutK in prev_mut_status_debt_df[['bid', 'mutK']].values])
    mut_status_df = mut_status_df.loc[~(mut_status_df.apply(lambda v: f"{v.bid}-{v.mutK}" in processed, axis = 1))]
    print (f'Changed from {n_prev} to {len(mut_status_df)}, {len(processed)}')
    if len(mut_status_df) == 0: # all processed:
      return True 
  # also, further focus on specifc mutants b/c the data mined here will be constrained to those in outputdir
  if NEED_TO_FOUCS_OPS is not None: 
    mut_status_df = mut_status_df[mut_status_df.mutOp.isin(NEED_TO_FOUCS_OPS)]
  elif NEED_TO_FILTER_OPS is not None:
    mut_status_df = mut_status_df[~mut_status_df.mutOp.isin(NEED_TO_FILTER_OPS)]
  ###
  # compute debt and add 
  mut_status_df['debt_time'] = [-1] * len(mut_status_df) # the one that we target to update 
  mut_status_df['debt_nc'] = [-1] * len(mut_status_df) # the one that we target to update 
  bidToRev_dict, _ = analysis_utils.getBidRevDict(D4J_HOME, project)
  all_commits = git_utils.getAllCommits(PROJECT_REPOS[project], branch = "trunk" if project in ['Lang', 'Math'] else 'master')
  latest_commit = all_commits[0]
  #for bid in mut_status_df.bid.unique(): # for each 
  for bid, df in mut_status_df.groupby('bid'): # for each 
    fixedRev = bidToRev_dict[bid]
    print (bid, fixedRev)
    # survived 
    days_survived = compute_tech_debt(repo, latest_commit, fixedRev)
    nc_survived = compute_tech_debt_ncommits(all_commits, latest_commit, fixedRev)
    #
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'surv'), 'debt_time'] = days_survived
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'surv'), 'debt_nc'] = nc_survived
    # revealed 
    _, _, mapped_to_uniq_mutK = analysis_utils.group_same_mutants(outputdir, project, fixedRev, 
      needToFilterOps = NEED_TO_FILTER_OPS, needToFocusOps = NEED_TO_FOUCS_OPS, 
      regen_uniq=regen_uniq
    ) # 
    should_target_revealeds = set(df[df.status == 'reveal'].mutK.values.tolist())
    should_curr_target_mutKs = set()
    for k, uniq_k in mapped_to_uniq_mutK.items():
      if uniq_k in should_target_revealeds:
        should_curr_target_mutKs.add(k)

    reveal_mut_files = glob.glob(os.path.join(outputdir, project, "inter", fixedRev, "revealedAt.*.json"))
    days_survived_l, nc_survived_l = {}, {}
    for reveal_mut_file in reveal_mut_files:
      with open(reveal_mut_file) as f:
        import json 
        data = json.load(f)
      revealedAt = os.path.basename(reveal_mut_file).split(".")[1]
      days_survived = compute_tech_debt(repo, revealedAt, fixedRev)
      nc_survived = compute_tech_debt_ncommits(all_commits, revealedAt, fixedRev)
      #
      revealed_mutKs = [analysis_utils.format_mutK(f"{fpath}-{mutNo}") for fpath,vs in data.items() for mutNo in vs]
      for mutK in revealed_mutKs:
        # temporary filter: due to data saved in two places: 
        if mutK not in should_curr_target_mutKs: continue 
        uniq_mutK = mapped_to_uniq_mutK[mutK]
        try:
          days_survived_l[uniq_mutK] = min(days_survived_l[uniq_mutK], days_survived) # take the shorter one
          nc_survived_l[uniq_mutK] = min(nc_survived_l[uniq_mutK], nc_survived) # take the shorter one
        except KeyError:
          days_survived_l[uniq_mutK] = days_survived
          nc_survived_l[uniq_mutK] = nc_survived
    rdf = df[df.status == 'reveal']
    # here, key error because, df containts more, 
    vs = [days_survived_l[mutK] for mutK in rdf.mutK.values] # RELATED TO FILTERED? 
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'reveal'), 'debt_time'] = vs 
    vs = [nc_survived_l[mutK] for mutK in rdf.mutK.values] # RELATED TO FILTERED? 
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'reveal'), 'debt_nc'] = vs 

    # deadAt 
    should_target_revealeds = set(df[df.status == 'dead'].mutK.values.tolist())
    should_curr_target_mutKs = set()
    for k, uniq_k in mapped_to_uniq_mutK.items():
      if uniq_k in should_target_revealeds:
        should_curr_target_mutKs.add(k)

    _, _, deadmuts_file, _ =  analysis_utils.getToSaveFiles(
      os.path.join(outputdir, "processed", project), project, fixedRev)
    with open(deadmuts_file, 'rb') as f:
      import pickle 
      dead_muts = pickle.load(f)
    days_survived_l, nc_survived_l = {}, {}
    for (fpath, mutNo), deadAt in dead_muts.items():
      mutK = analysis_utils.format_mutK(f"{fpath}-{mutNo}")
      if mutK not in should_curr_target_mutKs: continue 
      uniq_mutK = mapped_to_uniq_mutK[mutK]
      if deadAt is not None:
        days_survived = compute_tech_debt(repo, deadAt, fixedRev)
        #days_survived_l[analysis_utils.format_mutK(f"{fpath}-{mutNo}")] = days_survived 
        days_survived_l[uniq_mutK] = days_survived 
        #
        nc_survived = compute_tech_debt_ncommits(all_commits, deadAt, fixedRev)
        #nc_survived_l[analysis_utils.format_mutK(f"{fpath}-{mutNo}")] = nc_survived 
        nc_survived_l[uniq_mutK] = nc_survived 
    df = mut_status_df[(mut_status_df.bid == bid) & (mut_status_df.status == 'dead')] 
    vs = [days_survived_l[mutK] for mutK in df.mutK.values]
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'dead'), 'debt_time'] = vs 
    vs = [nc_survived_l[mutK] for mutK in df.mutK.values]
    mut_status_df.loc[(mut_status_df.bid == bid) & (mut_status_df.status == 'dead'), 'debt_nc'] = vs 

    #print (mut_status_df.loc[mut_status_df.bid == bid])
    #sys.exit()
  #
  if prev_mut_status_debt_df is not None:
    mut_status_df = pd.concat([prev_mut_status_debt_df, mut_status_df], ignore_index = True)
    mut_status_df = mut_status_df.drop_duplicates()  
  ## checking 
  grouped_indv_muts = mut_status_df.groupby(['bid', 'mutK'])
  assert len(grouped_indv_muts) == len(mut_status_df), f"{len(grouped_indv_muts)} vs {len(mut_status_df)}"
  ##
  mut_status_df.to_csv(mut_propgate_status_and_debt_file, index = False)
  print (f"Save to {mut_propgate_status_and_debt_file}")
  return mut_status_df


def record_init_indv_mt_statistics(outputdir:str, project:str, 
  bidToRev_dict:Dict[str,str], dest:str) -> pd.DataFrame:
  """
  """
  to_save_file = os.path.join(dest, f"{project}.init_pit_indv_mut_status.csv") 
  prev_init_indv_mut_status = None
  if False:#os.path.exists(to_save_file):
    #return pd.read_csv(to_save_file)
    prev_init_indv_mut_status = pd.read_csv(to_save_file)
  init_indv_mut_status = analysis_utils.get_init_indv_mut_status(outputdir, project, list(bidToRev_dict.items()), MUTOPS = MUTOPS)
  ##
  # for "sourceFile"
  rows = []
  for bid, df in tqdm(init_indv_mut_status.groupby('bid')):
    fixedRev = bidToRev_dict[bid]
    repo_path = PROJECT_REPOS[project]
    full_fpath_d = {}
    for _, row in df.iterrows():
      partial_file_path = row.sourceFile
      class_id = row.mutatedClass
      try:
        full_fpath = full_fpath_d[class_id]
      except KeyError:
        #print (partial_file_path, class_id)
        full_fpath = git_utils.get_full_fpath(fixedRev, partial_file_path, repo_path, class_id)
        assert full_fpath is not None, f"{bid}, {fixedRev}, {partial_file_path}, {class_id}, {row.k}"
        full_fpath_d[class_id] = full_fpath
      rows.append(row.values.tolist() + [full_fpath])
  init_indv_mut_status = pd.DataFrame(rows, columns = init_indv_mut_status.columns.values.tolist() + ['full_sourceFile'])
  #
  ##
  init_indv_mut_status.mutOp = init_indv_mut_status.mutOp.apply(lambda v:'AOD' if v.startswith("AOD") else v)
  init_indv_mut_status.mutOp = init_indv_mut_status.mutOp.apply(lambda v:'CR' if v.startswith("CR") else v)
  init_indv_mut_status.mutOp = init_indv_mut_status.mutOp.apply(lambda v:'ROR' if v.startswith("ROR") else v)
  init_indv_mut_status.mutOp = init_indv_mut_status.mutOp.apply(lambda v:'OBBN' if v.startswith("OBBN") else v)
  init_indv_mut_status.mutOp = init_indv_mut_status.mutOp.apply(lambda v:'AOR' if v.startswith("AOR") else v)
  ##
  if prev_init_indv_mut_status is not None:
    init_indv_mut_status = pd.concat([prev_init_indv_mut_status, init_indv_mut_status], ignore_index = True)
    init_indv_mut_status = init_indv_mut_status.drop_duplicates()
  init_indv_mut_status.to_csv(to_save_file, index = False)
  print (f"Save to {to_save_file}")
  return init_indv_mut_status


def record_init_mt_statistics(outputdir:str, project:str, 
  bidToRev_dict:Dict[str,str], dest:str) -> pd.DataFrame:
  """
  """
  to_save_file = os.path.join(dest, f"{project}.init_pit_mut_status.csv") 
  prev_initStatusPBug = None
  if os.path.exists(to_save_file):
    prev_initStatusPBug = pd.read_csv(to_save_file)

  initStatusPBug = analysis_utils.getInitMutStatus(
    outputdir, project, list(bidToRev_dict.items()), logging = False, MUTOPS=MUTOPS)  
  
  init_mutOpFreq = analysis_utils.get_mutator_freq(
    outputdir, project, list(bidToRev_dict.items()), only_unique = False)
  n_org_init = len(initStatusPBug)
  initStatusPBug = initStatusPBug.merge(
    init_mutOpFreq, on = ['bid', 'k'], how = 'inner')
  assert n_org_init == len(initStatusPBug), f"{len(initStatusPBug)} vs {n_org_init}"
  assert len(init_mutOpFreq) == len(initStatusPBug), f"{len(initStatusPBug)} vs {len(init_mutOpFreq)}"
  print ("init collected")
  unique_initStatusPBug = analysis_utils.getInitMutStatus_of_unique(
    outputdir, project, list(bidToRev_dict.items()), logging = False, MUTOPS=MUTOPS)  
  unique_init_mutOpFreq = analysis_utils.get_mutator_freq(
    outputdir, project, list(bidToRev_dict.items()), only_unique = True)
  u_n_org_init = len(unique_initStatusPBug)
  unique_initStatusPBug = unique_initStatusPBug.merge(
    unique_init_mutOpFreq, on = ['bid', 'k'], how = 'inner')
  assert u_n_org_init == len(unique_initStatusPBug), f"{len(unique_initStatusPBug)} vs {u_n_org_init}"
  assert len(unique_init_mutOpFreq) == len(unique_initStatusPBug), f"{len(unique_initStatusPBug)} vs {len(unique_init_mutOpFreq)}"
  print ("init regarding unique mutants collected")
  # combine
  combiend_statusPBug = initStatusPBug.merge(
    unique_initStatusPBug, on = ['bid', 'k'], how = 'inner')
  assert len(combiend_statusPBug) == len(initStatusPBug), f"{len(combiend_statusPBug)} vs {len(initStatusPBug)}"
  assert len(combiend_statusPBug) == len(unique_initStatusPBug), f"{len(combiend_statusPBug)} vs {len(unique_initStatusPBug)}"

  if prev_initStatusPBug is not None:
    initStatusPBug = pd.concat([prev_initStatusPBug, initStatusPBug], ignore_index = True)
  initStatusPBug = initStatusPBug.drop_duplicates()
  #to_save_file = os.path.join(dest, f"{project}.init_pit_mut_status.csv")
  combiend_statusPBug.to_csv(to_save_file, index = False)
  print (f"Save to {to_save_file}")
  return initStatusPBug

def get_init_mut_infos(outputdir:str, project:str, fixedRev:str, uniq_muts:bool = True) -> Dict:
  rets = analysis_utils.getCompletedMutInfos(outputdir, project, fixedRev, uniq_muts = uniq_muts)
  rets = {analysis_utils.format_mutK(k):v for k,v in rets.items()}
  return rets 

def get_applied_at_mut_infos(outputdir:str, 
  project:str, fixedRev:str, appliedAt:str, target_mutKs:List[str] = None) -> Dict[str,List]:
  """
  target_mutKs: from revealedAt info 
    -> appliedAt info always maintain the most recent version -> if not here, then ignore (likely the targets being randomly revealed)
  """
  ## curently applieAt will take revealedAt for this generaiton 
  dirpath = os.path.join(outputdir, project, "inter", fixedRev)
  appliedAtInfo_file = os.path.join(dirpath, f"appliedAt.{appliedAt}.json")
  with open(appliedAtInfo_file) as f:
    import json  
    appliedAtInfo = json.load(f)
  rets = {}
  for mutated_fpath, pfileInfo in appliedAtInfo.items():
    parsed_pos_node_dict, parsed, d_file_contents = {}, {}, {}
    for mutNo, pmutInfo in pfileInfo.items():
      mutNo = int(mutNo)
      mutK = analysis_utils.format_mutK(f"{mutated_fpath}-{mutNo}")
      if target_mutKs is not None and mutK not in target_mutKs: continue
      rets[mutK] = []
      for revealed_fpath, mutLocs in pmutInfo.items():
        for _, mutLoc in mutLocs:
          #print (revealed_fpath, appliedAt, mutLoc)
          #try:
          lno_revealedAt = min(getLnoAtCommit(PROJECT_REPOS[project], appliedAt, 
            revealed_fpath, mutLoc[-2], mutLoc[-1]))
          #except Exception as e:
          #  print(e)
          #  print (mutLoc, revealed_fpath, appliedAt, fixedRev)
          #  sys.exit()
          try:
            build_pos_dict, lno_to_node_dict = parsed_pos_node_dict[revealed_fpath]
          except KeyError:
            try:
              tree = parsed[revealed_fpath]
            except KeyError:
              try:
                file_content = d_file_contents[revealed_fpath]
              except KeyError:
                file_content = git_utils.show_file(
                  appliedAt, revealed_fpath, PROJECT_REPOS[project])
                d_file_contents[revealed_fpath] = file_content
              tree = parser_utils.parse(file_content)
              parsed[revealed_fpath] = tree 
            build_pos_dict, lno_to_node_dict = parser_utils.build_positon_dict(tree)
            parsed_pos_node_dict[revealed_fpath] = (build_pos_dict, lno_to_node_dict)
        
          new_all_chgd_lnos = getMutLoc_atRevealed(
              lno_revealedAt, build_pos_dict, lno_to_node_dict)
          min_new_lno, max_new_lno = min(new_all_chgd_lnos), max(new_all_chgd_lnos)
          rets[mutK].append(
            [revealed_fpath, f"{min_new_lno}:{max_new_lno}", mutLoc[-2], mutLoc[-1]])
  return rets 

def get_target_revealed_info(revealed_mut_infos:Dict, lno_at_revealed:int):
  import numpy as np 
  diffs = []
  for info in revealed_mut_infos:
    start_lno, end_lno = info[1].split(":")
    start_lno = int(start_lno)
    end_lno = int(end_lno)
    diffs.append(
      min([np.abs(start_lno - lno_at_revealed), np.abs(end_lno - lno_at_revealed)])
    )
    if lno_at_revealed in range(start_lno, end_lno+1):
      return info, True 

  idx_to_min = diffs.index(np.min(diffs))
  return revealed_mut_infos[idx_to_min], False

def get_valid_invalid_record_file(check_dir:str, project:str, bid:int) -> str:
  check_file = os.path.join(check_dir, project, f"{project}_{bid}_mut_valid_invalid.json")
  return check_file 

def compute_final_chg_state(revealedAt:str, allCommits:List[str], 
  isNotRefactoring_d:Dict, isSemanticChange_d:Dict, isChanged_d:Dict) -> Tuple[bool,bool,bool]:
  """
  isNotRefactoring, isSemanticChange, isChanged
  """
  def get_flags_within(d:Dict, revealedAt:str, allCommits:List[str]) -> List[bool]:
    idx_to_revealedAt = analysis_utils.getCommitIndex(allCommits, revealedAt)
    ret_flags = []
    for rev, flag in d.items():
      if rev is not None:
        idx_to_rev = analysis_utils.getCommitIndex(allCommits, rev)
        if idx_to_revealedAt <= idx_to_rev:
          ret_flags.append(flag)
      else: # not changed at all
        continue
    return ret_flags
  # with emty list -> the value will be False
  isNotRefactoring = any(get_flags_within(isNotRefactoring_d, revealedAt, allCommits))
  isSemanticChange = any(get_flags_within(isSemanticChange_d, revealedAt, allCommits))
  isChanged = any(get_flags_within(isChanged_d, revealedAt, allCommits))
  return (isNotRefactoring, isSemanticChange, isChanged)


def record_revealed_mut_statistics(statdir:str, outputdir:str, 
  project:str, bidToRev_dict:Dict[str,str], 
  dest:str, uniq_muts:bool = True, 
  check_dir:str = None, 
  regen_uniq:bool = False, 
  chg_stat_dir:str = None) -> pd.DataFrame:
  """
  """
  def checkProcessed(_mutK, revealedAt, df):
    if df is None: return False
    processed = df.loc[(df.mutK == _mutK) & (df.revealedAt == revealedAt)]
    return len(processed) > 0
  
  # here, all_XXX_True.pkl is accessed 
  revealedInfos = analysis_utils.getMutantsRevealInfos(statdir, project)
  #
  if chg_stat_dir is not None:
    chgStateInfos = analysis_utils.getMutantsRevealInfos(chg_stat_dir, project)
  else:
    chgStateInfos = None
  #
  allCommits = git_utils.getAllCommits(PROJECT_REPOS[project], 
    'master' if project not in ['Lang', 'Math'] else 'trunk')
  repo = git_utils.get_repo(PROJECT_REPOS[project])
  
  for bid, revealedInfoPbug in tqdm(revealedInfos.items()):
    #print (bid)
    #if bid != 20: continue
    #if bid !=31: continue
    cnt_approximated = 0
    to_save_file = os.path.join(dest, f"{project}.{bid}.revealed_comp_mutinfo.pkl")
    if os.path.exists(to_save_file):
      prev_mut_stats = pd.read_pickle(to_save_file) # per-bid 
    else:
      prev_mut_stats = None
    fixedRev = bidToRev_dict[bid]
    #print (bid, fixedRev)
    #all_init_mut_infos = get_init_mut_infos(outputdir, project, fixedRev, uniq_muts=False)
    init_mut_infos = get_init_mut_infos(outputdir, project, fixedRev, uniq_muts=False)#uniq_muts)
    _, _, mapped_to_uniq_mutK = analysis_utils.group_same_mutants(
      outputdir, project, fixedRev, 
      needToFocusOps=NEED_TO_FOUCS_OPS,
      needToFilterOps=NEED_TO_FILTER_OPS, 
      regen_uniq=regen_uniq)
    
    if check_dir is not None:
      check_file = get_valid_invalid_record_file(check_dir, project, bid)
      if os.path.exists(check_file):
        formatted_invalid_mutKs = analysis_utils.get_formatted_muKs_to_invalid(check_file)
        # convert to 
        formatted_invalid_mutKs = analysis_utils.get_only_target_uniq_muts(
          formatted_invalid_mutKs, mapped_to_uniq_mutK)
      else:
        formatted_invalid_mutKs = set()
    else:
      formatted_invalid_mutKs = set()
    # 
    # filter prev_mut_stats 
    if prev_mut_stats is not None:
      prev_n = len(prev_mut_stats)
      prev_mut_stats = prev_mut_stats[~(prev_mut_stats.mutK.isin(formatted_invalid_mutKs))]
      nothing_in_prev_filtered = prev_n == len(prev_mut_stats)
    else:
      nothing_in_prev_filtered = True 
    # 
    mut_status = {
      'mutK':[], 'mutOp':[], 
      'level':[], 
      'notRefAndSemChg':[],'notRefactoring':[], 'semanticChange':[], 'changed':[], 
      'init_mut_loc':[], 'revealed_mut_loc':[], 
      'mutatedAt':[], 'revealedAt':[], 
      'debt_commit':[], 'debt_time':[]
    }

    uniq_revealed_mut_infos_d = {}
    checked = set()
    for (fpath, mutNo), mutInfoLst in tqdm(revealedInfoPbug.items()):
      _mutK = analysis_utils.format_mutK(f"{fpath}-{mutNo}")
      #if _mutK != 'src.main.java.org.apache.commons.lang3.StringUtils-105': continue
      try:
        mutK_to_uniq = mapped_to_uniq_mutK[_mutK]
      except KeyError:
        continue # since NEED_TO_FILTER_OPS and NEED_TO_FOCUS_OPS are reflected in mapped_to_uniq_mutK
      #if mutK_to_uniq != 'src.main.java.org.apache.commons.math3.optimization.direct.CMAESOptimizer-173': continue
      #_mutOp = all_init_mut_infos[_mutK]['mutOp'][0] ### original code 
      #if NEED_TO_FILTER_OPS is not None and _mutOp in NEED_TO_FOUCS_OPS: continue 
      #mutK_to_uniq = mapped_to_uniq_mutK[_mutK]
      # filter 
      if mutK_to_uniq in checked: continue # currently checked (b/c here, we will drop duplicated results)
      if formatted_invalid_mutKs is not None and mutK_to_uniq in formatted_invalid_mutKs: continue
      checked.add(mutK_to_uniq)
      #print ('targeting...', mutK_to_uniq)
      mutOp = init_mut_infos[mutK_to_uniq]['mutOp'][0] # -> revealed
      ##
      # if any of the changes is not refactoring then True
      isNotRefactoring_d, isSemanticChange_d, isChanged_d = {}, {}, {}
      revealedAts, modifedAts = [], []
      revealed_mut_loc_d = {}
      for mutInfo in mutInfoLst: # include per-change info
        #print (mutInfo[0])
        (_isNotRefactoring, _isSemanticChange, _isChanged), others = analysis_utils.processInfos(mutInfo)
        #print ("--", _isNotRefactoring, _isSemanticChange, _isChanged)
        #mutLevel = analysis_utils.computeMutLevel(isNotRefactoring, isSemanticChange, isChanged)
        #
        revealedAt = others[0][1] 
        modifedAt = others[0][2] # this does not mean that the mutant is revealed here: simply meaning this mutant was changed at this commit
        idxToRevealedAt = analysis_utils.getCommitIndex(allCommits, revealedAt)
        revealedAts.append([revealedAt, idxToRevealedAt])
        #idxToModifiedAt = analysis_utils.getCommitIndex(allCommits, modifedAt)
        modifedAts.append(modifedAt)
        #
        isNotRefactoring_d[modifedAt] = _isNotRefactoring
        isSemanticChange_d[modifedAt] = _isSemanticChange
        isChanged_d[modifedAt] = _isChanged
        #
        lno_at_revealed = others[2][1]
        if checkProcessed(mutK_to_uniq, revealedAt, prev_mut_stats): continue # prev-chcecked  ## DISABLE FOR DEBGGING
        if (revealedAt in uniq_revealed_mut_infos_d.keys()) and (
          mutK_to_uniq in uniq_revealed_mut_infos_d[revealedAt].keys()):
          uniq_revealed_mut_infos = uniq_revealed_mut_infos_d[revealedAt]
        else:
          uniq_revealed_mut_infos = get_applied_at_mut_infos(outputdir, project, 
            fixedRev, revealedAt, #target_mutKs = [mutK_to_uniq])
            target_mutKs = [_mutK]) # use the key in revealedAt
          if len(uniq_revealed_mut_infos) == 0: continue
          try:
            uniq_revealed_mut_infos_d[revealedAt].update(uniq_revealed_mut_infos)
          except KeyError:
            uniq_revealed_mut_infos_d[revealedAt] = uniq_revealed_mut_infos
        #
        #revealed_mut_loc, flag = get_target_revealed_info(uniq_revealed_mut_infos[mutK_to_uniq], lno_at_revealed)
        revealed_mut_loc, flag = get_target_revealed_info(uniq_revealed_mut_infos[_mutK], lno_at_revealed)
        if not flag: cnt_approximated += 1
        revealed_mut_loc = tuple(revealed_mut_loc)
        revealed_mut_loc_d[revealedAt] = revealed_mut_loc

      if len(revealed_mut_loc_d) == 0: # mostly those random: 
        continue
      #print ('here???')
      # take the oldest revealedAt: by finding here, we .. 
      revealedAt = sorted(revealedAts, key = lambda v:v[1])[-1][0]
      #isNotRefactoring, isSemanticChange, _ = compute_final_chg_state(revealedAt, allCommits, isNotRefactoring_d, isSemanticChange_d, isChanged_d)
      #print ('final', isNotRefactoring, isSemanticChange)
      #print (isNotRefactoring_d)
      #print (isSemanticChange_d)
      #try:
      #print (chgStateInfos)
      if chgStateInfos is not None:
        try:
          #print ('here?')
          isNotRefactoring_l, isSemanticChange_l= [], []
          _fpath, _mutNo = analysis_utils.reverse_format_mutK(mutK_to_uniq).split("-")
          _mutNo = int(_mutNo)
          #print ((_fpath, _mutNo))
          mutInfoLst = chgStateInfos[bid][(_fpath,_mutNo)]
          #print (mutInfoLst)
          #print ('-----')
          for mutInfo in mutInfoLst:
            (_isNotRefactoring, _isSemanticChange, _isChanged), others = analysis_utils.processInfos(mutInfo, only_chg=True)
            isNotRefactoring_l.append(_isNotRefactoring)
            isSemanticChange_l.append(_isSemanticChange)

        except KeyError:
          #print ("here????????")
          isNotRefactoring_l = list(isNotRefactoring_d.values())
          isSemanticChange_l = list(isSemanticChange_d.values())
      else:
        isNotRefactoring_l = list(isNotRefactoring_d.values())
        isSemanticChange_l = list(isSemanticChange_d.values())

      mutLevel, notRefAndSemChg, notRefactoring, semanticChange, changed = analysis_utils.computeRevealedMutType(
        #list(isNotRefactoring_d.values()), list(isSemanticChange_d.values()), 
        isNotRefactoring_l, isSemanticChange_l, 
        revealedAt, modifedAts) # modifiedAt -> should be the most recent one
      #print (mutLevel, notRefAndSemChg, notRefactoring, semanticChange, changed)
      #sys.exit()
      #except Exception:
      #  print (bid, mutK_to_uniq)
      #  sys.exit()
      #print ('Level', mutLevel)
      ##
      mut_status['mutK'].append(mutK_to_uniq)
      mut_status['mutOp'].append(mutOp)
      mut_status['level'].append(mutLevel)
      # 
      mut_status['notRefAndSemChg'].append(notRefAndSemChg)#isNotRefactoring & isSemanticChange)
      mut_status['notRefactoring'].append(notRefactoring)#isNotRefactoring)
      mut_status['semanticChange'].append(semanticChange)#isSemanticChange)
      mut_status['changed'].append(semanticChange)#isSemanticChange)
      # 
      init_mut_loc = init_mut_infos[mutK_to_uniq]['pos']
      mut_status['init_mut_loc'].append(init_mut_loc) # mutated_lno, start_p, end_p 
      #mut_status['revealed_mut_loc'].append(revealed_mut_loc) # revealed_fpath, f"{min_new_lno}:{max_new_lno}", mutLoc[-2], mutLoc[-1]
      mut_status['revealed_mut_loc'].append(str(revealed_mut_loc_d)) # revealed_fpath, f"{min_new_lno}:{max_new_lno}", mutLoc[-2], mutLoc[-1]
      mut_status['mutatedAt'].append(fixedRev)
      mut_status['revealedAt'].append(revealedAt)
      #
      # compute debt 
      idxToMutatedAt = analysis_utils.getCommitIndex(allCommits, fixedRev)
      idxToRevealedAt = analysis_utils.getCommitIndex(allCommits, revealedAt)
      n_btwn_commits = idxToMutatedAt - idxToRevealedAt
      datetime_mutatedAt = analysis_utils.getCommitedDateTime(repo, fixedRev)
      datetime_revealedAt = analysis_utils.getCommitedDateTime(repo, revealedAt)
      sec_btwn_commits = (datetime_revealedAt - datetime_mutatedAt).total_seconds()/(60*60*24)
      #
      mut_status['debt_commit'].append(n_btwn_commits)
      mut_status['debt_time'].append(sec_btwn_commits)
    
    mut_status = pd.DataFrame(mut_status)
    #sys.exit()
    if cnt_approximated > 0:
      print (f"\tout of {len(mut_status)}, {cnt_approximated} approximatd")
    if len(mut_status) > 0 or (not nothing_in_prev_filtered): # only if there are somethin gto add
      #if prev_mut_stats is not None: #len(mut_status) > 0:
        #mut_status = pd.concat([prev_mut_stats, mut_status]).drop_duplicates()
      #elif 
      if prev_mut_stats is not None and len(mut_status) > 0:
        mut_status = pd.concat([prev_mut_stats, mut_status]).drop_duplicates()
      elif prev_mut_stats is not None: # and len(mut_status) == 0
        mut_status = prev_mut_stats # as something were filtered out 
      # len(mut_status) > 0 and prev_mut_stats = None
      mut_status.to_pickle(to_save_file)
      print (f"Save to {to_save_file}")  
  

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  #parser.add_argument("-o", "--outputdir", type = str, default = "output/pit/remore/final_latest", 
  #  help = "a path to the output directory or a list of paths separated by ,")
  #parser.add_argument("-s", "--statdir", type = str, default = "output/stat/pit/final_latest", 
  #  help = "a path to the stat directory or a list of paths separated by ,")
  parser.add_argument("-d", "--dest", type = str, default = "output/evaluation/combined_v2")
  parser.add_argument("-p", "--project", type = str)

  args = parser.parse_args()

  project = args.project
  #outputdir = args.outputdir
  #statdir = args.statdir 
  dest = args.dest
  os.makedirs(dest, exist_ok = True)

  outputdirs = ['output/pit/remote/final_latest']#, 'output/pit/remote/final_latest_remain']
  statdirs = ['output/stat/pit/final_latest']#, 'output/stat/pit/final_latest_remain']
  bidToRev, revToBid = analysis_utils.getBidRevDict(os.getenv("D4J_HOME"), project)
  targetdir = "data/targets/toFocus/all"
  #
  regen_uniq = True
  #
  #
  to_save_n_mut_file = os.path.join(dest, f"{project}.n_mut_propagation_status.csv")
  if os.path.exists(to_save_n_mut_file):
    os.remove(to_save_n_mut_file)
  
  # save init mutant status  => these two are about raw pit results 
  #_ = record_init_mt_statistics(outputdirs[0], project, bidToRev, dest)
  _ = record_init_indv_mt_statistics(outputdirs[0], project, bidToRev, dest) 
  sys.exit()
  check_dir = "output/evaluation/mut_val_check" # since this process the raw result that contain everythig
  #
  chg_stat_dir = "output/stat/pit/v3/final_latest_combined"
  #
  for outputdir, statdir in zip(outputdirs, statdirs):
    #if "remain" in outputdir:
      #NEED_TO_FILTER_OPS = None
      #NEED_TO_FOUCS_OPS = ['CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 'INLINE_CONSTS']
    #else:
      #NEED_TO_FILTER_OPS = ['CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 'INLINE_CONSTS']
      #NEED_TO_FOUCS_OPS = None
    proj_outputdir = os.path.join(outputdir, project)

    # save revealed mutants refactoring status 
    record_revealed_mut_statistics(statdir, outputdir, project, 
      bidToRev, dest, uniq_muts = True, check_dir = check_dir, regen_uniq=regen_uniq, 
      chg_stat_dir=chg_stat_dir)

    # svae init mut refactor status 
    #target_file = os.path.join(targetdir, f"{project}.csv")
    #with open(target_file) as f:
    #  targeted_bids = [int(bid.strip()) for bid in f.readlines() if bool(bid.strip())]
    
    #record_mut_propagation_stauts(outputdir, project, targeted_bids, bidToRev, dest, check_dir = check_dir, 
    #                              regen_uniq=regen_uniq)
    #record_mut_debt_statistics(outputdir, project, bidToRev, dest, regen_uniq = regen_uniq)
    #sys.exit()
  # test 
  #to_save_indv_mut_file = os.path.join(dest, f"{project}.indv_mut_propagation_status.csv")
  #to_save_indv_mut_and_debt_file = os.path.join(dest, f"{project}.indv_mut_propagation_status_and_debt.csv")
  #df_indv_mut = pd.read_csv(to_save_indv_mut_file)
  #df_indv_mut_debt = pd.read_csv(to_save_indv_mut_and_debt_file)
  ##vs1 = set([f"{bid}-{mutK}" for bid, mutK in df_indv_mut[['bid', 'mutK']].values])
  ##vs2 = set([f"{bid}-{mutK}" for bid, mutK in df_indv_mut_debt[['bid', 'mutK']].values])
  #assert len(df_indv_mut) == len(df_indv_mut_debt), f"{len(df_indv_mut)} vs {len(df_indv_mut_debt)}"

