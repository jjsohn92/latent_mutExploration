"""
By propgating a failing test that killed a mutant, check whether we can 

- Currently (31.01.2024), the dependency to a test method has not been considered yet, for both candidate selection and the test-injection to a test file 
  at the time of the latent mutant introduction 
  We will simply look at whether a developer can find an injected mutant at the time of introduciton assuming that a developer has the test  

"""
import os, sys 
import pandas as pd 
import pickle 
import json
import numpy as np 
import glob 
from utils import analysis_utils, git_utils, file_utils, mvn_utils, java_utils, ant_d4jbased_utils, ant_mvn_utils
import pydriller 
from tqdm import tqdm 
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List, Dict, Tuple
import process.test_modifier as test_modifier 
from mutants import pit_mutant
import propagator

import logging 
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = None

D4J_HOME = os.getenv('D4J_HOME')
random_seed = 0
old_outputdir = "/Users/jeongju.sohn/workdir/mutBugInducing/output/pit/remote/final_latest"

REPOS = {
    'Lang':os.path.join(D4J_HOME, "project_repos/commons-lang.git"),
    'Math':os.path.join(D4J_HOME, "project_repos/commons-math.git"),
    'Time':os.path.join(D4J_HOME, "project_repos/joda-time.git"), 
    'Compress':os.path.join(D4J_HOME, "project_repos/commons-compress.git"), 
    'Cli':os.path.join(D4J_HOME, "project_repos/commons-cli.git"),
    'Codec':os.path.join(D4J_HOME, "project_repos/commons-codec.git"), 
    'Gson':os.path.join(D4J_HOME, "project_repos/gson.git"),
    'Closure':os.path.join(D4J_HOME, "project_repos/closure-compiler.git"),
    'Collections':os.path.join(D4J_HOME, "project_repos/commons-collections.git"), 
    'Jsoup':os.path.join(D4J_HOME, "project_repos/jsoup.git"), 
    'Csv':os.path.join(D4J_HOME, "project_repos/commons-csv.git"), 
    'Mockito':os.path.join(D4J_HOME, "project_repos/mockito.git"),
    'JxPath':os.path.join(D4J_HOME, "project_repos/commons-jxpath.git"),
    'JacksonXml':os.path.join(D4J_HOME, "project_repos/jackson-dataformat-xml.git"),
    'JacksonCore':os.path.join(D4J_HOME, "project_repos/jackson-core.git"),
    'JacksonDatabind':os.path.join(D4J_HOME, "project_repos/jackson-databind.git"),
}
from main import commonGIDs, prepare_workdir

def set_default_logger():
  global logger
  if logger is None: 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def get_latest_commit(project:str) -> str:
  """
  """
  import git 
  repo = git.Repo(REPOS[project])
  for c in repo.iter_commits():
    return c.hexsha[:8]
  return None 

def prepare_temp_refdir(project:str, base_refdir:str, checkout_commit:str):
  """
  generate a temporary reference directory or repo with the most recent commit 
  
  :param project 
  :param base_refdir: a base for the reference directories
  :param commit  
  :param commit
  """
  active_bugfile = os.path.join(D4J_HOME, f"framework/projects/{project}/active-bugs.csv")
  df = pd.read_csv(active_bugfile)
  random_bid = df['bug.id'].values[0]

  os.makedirs(base_refdir, exist_ok=True)
  temp_refdir = prepare_workdir(D4J_HOME, project, random_bid, checkout_commit, base_refdir)
  return temp_refdir

#def get_full_fpath(ftest:str) -> str:
  #"""
  #:param repo_path: a repository path 
  #:param ftest: a test identifier - ftestclass#ftestname
  #"""
  #import pathlib 
  #sub_fpath = ftest.split("#")[0].replace(".", "/") + ".java"
  #found = glob.glob(os.path.join("**", sub_fpath), recursive = True)
  #assert len(found) == 1, sub_fpath 
  #target_fpath = found[0]
  #return target_fpath
def get_ftest_chghist(repo_path:str, ftest:str, from_commit:str, to_commit:str, all_commits:List[str]) -> Tuple[str, List[List]]:
  """
  Cover both changed and newly introduced 
  if ftest exist when a mutant is introduced, then check the history 
  -> (we can go step by step?)

  :param ftest: e.g., org.apache.commons.lang3.time.FastDateFormatTest#testFormat
  :param target_fpath: 

  return (target_ftest_path[str], ftest_change_history[List])
  """
  set_default_logger()
  ftest_shortform = ftest.split(".")[-1].replace('#', "::")
  sub_fpath = ftest.split("#")[0].replace(".", "/") + ".java"
  target_fpath = git_utils.get_full_fpath(to_commit, sub_fpath, repo_path)
  if target_fpath is None:
    logger.error(f"{target_fpath} does not exist in {to_commit}")
    return None 
  ### !!!!! here, we need to also SET TO_COMMIT -> THE INJECTION POINT (ALSO, FOR FROM_COMMIT -> THIS SHOULD 
  # BE THE COMMIT AT REVEALING POINT!)
  #repo = pydriller.Repository(
  #  repo_path, filepath = target_fpath, order = 'reverse', to_commit = from_commit) 
  repo = pydriller.Repository(
    repo_path, 
    filepath = target_fpath, 
    order = 'reverse', 
    to_commit = to_commit)
  idx_to_from_commit = git_utils.getCommitIdx(all_commits, from_commit) 
  # limit this range and also to a specific file & include the first commit
  change_hist = []
  updated_target_fpath = target_fpath
  #is_newly_introduced_at = []
  with logging_redirect_tqdm(): 
    for commit in tqdm(list(repo.traverse_commits())): 
      idx_to_commit = git_utils.getCommitIdx(all_commits, commit.hash[:8])
      if idx_to_commit > idx_to_from_commit: # older than from_commit:
        break 

      # also ... 
      for mf in commit.modified_files:
        if (mf.new_path == updated_target_fpath) and (
          mf.change_type not in [pydriller.ModificationType.RENAME, pydriller.ModificationType.COPY]
        ): # think about a file being entirely new ...etc. 
          # ... maybe better to filter out first here? 
          if ftest_shortform in [mth.name for mth in mf.changed_methods]: # we will simply check whether it is one of changed methods 
            # update target_fpath 
            ## whether newly introduced (i.e., only in new methods and not in methods before changes)
            is_newly_introduced = (
              ftest_shortform in [mth.name for mth in mf.methods]) and not (
              ftest_shortform in [mth.name for mth in mf.methods_before]
            )
            #if is_newly_introduced:
            #  is_newly_introduced_at.append(commit.hash[:8])
            change_hist.append([commit.hash[:8], updated_target_fpath, is_newly_introduced])
            updated_target_fpath = mf.old_path # update with the file path of the next commit 
  
  #assert len(is_newly_introduced_at) == 1 or len(is_newly_introduced_at) == 0, is_newly_introduced_at
  if len(change_hist) == 0:
    logger.warning(f"ftest {ftest_shortform} failed to be changed")
    return None 
  else:
    logger.info(f"failing test {ftest_shortform} changed {len(change_hist)} times")
    return target_fpath, change_hist 

def filter_muts_nochange_in_ftest(repo_path:str, latent_mut_data:pd.DataFrame)  -> pd.DataFrame:
  """
  :param repo_path 
  :param latent_mut_data: a target mutant dataframe 
  """
  set_default_logger()
  logger.info('Start filtering mutants with failing tests that remained unchanged')
  
  #ftest_changed = latent_mut_data.loc[latent_mut_data.ftests.apply(lambda v:bool(v)).values]
  all_commits = git_utils.getAllCommits(repo_path, branch=None)
  indices_to_ftest_changed = []
  all_ftest_paths, all_ftest_chgdhist = [], []
  # check the change history of ftests of latent_mut_data 
  
  bidToRev, _ = analysis_utils.getBidRevDict(D4J_HOME, project)
  with logging_redirect_tqdm(): # 
    for index, latent_mut in tqdm(list(latent_mut_data.iterrows())):
      bid = latent_mut.bid 
      from_commit = bidToRev[bid] # a mutant introduced this commit
      to_commit = latent_mut.revealedAt # a mutant revealed at this commit
      # repo_path = prepare_workdir(D4J_HOME, project, bid, fixedRev, basedir)
      ftests = latent_mut.ftests.split(",")
      # check ftest change history:
      ## 1. either changed at least once (... this may cover the case 2.)
      ## 2. or did not exist at the time of mutant generation 
      ftest_paths, chgdhist_pftest = {}, {}
      for ftest in ftests:
        # check whether this test identifier exists (this may not be complete (e.g., there can be cases
        # where only the test name is changed, but will be sufficient for the case study))
        ftest_path_and_chghist = get_ftest_chghist(repo_path, ftest, from_commit, to_commit, all_commits)
        if ftest_path_and_chghist is not None: # if any, then keep it 
          # save 
          ftest_path, ftest_chghist = ftest_path_and_chghist
          chgdhist_pftest[ftest] = ftest_chghist
          ftest_paths[ftest] = ftest_path
          #break # we will try to get all the possible information 
      if len(chgdhist_pftest) > 0:
        indices_to_ftest_changed.append(index)
        all_ftest_chgdhist.append(chgdhist_pftest)
        all_ftest_paths.append(ftest_paths)
  # 
  ftest_changed = latent_mut_data.loc[indices_to_ftest_changed]
  if ftest_changed.shape[0] == 0:
    logger.warning(f"all latent mutans of {project}, failing tests have never been changed")
    return None 
  else:
    # add ftest chg hist info 
    ftest_changed['ftesthist'] = all_ftest_chgdhist
    ftest_changed['ftestpath'] = all_ftest_paths
    logger.info(f"out of {latent_mut_data.shape[0]}, {ftest_changed.shape[0]} remain after filtering")
  return ftest_changed

def select_target_mutants(base_refdir:str, latent_mut_data:pd.DataFrame, 
  percentage:int, colname:str, numSel:int, min_numSel:int, processed:bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  select target mutants to inspect based on the test history, selecting only those where a test has been at least changed once or newly introduced
  
  :param base_refdir: 
  :param latent_mut_data: 
  :param percentage: 
  :param colname:
  :param numSel: 
  :param min_numSel: 
  :param processed: if True, assume that mutants with no-ftest changes are already filtered. Otherwise, filter
  """
  set_default_logger()
  if processed:
    logger.info(f"latent mutant data is already filtered")
    df = latent_mut_data
  else:
    logger.info(f'start selecting latent mutants (total: {latent_mut_data.shape[0]}) to investigate')
    # check whether a failing test was changed at least once or newly introduced 
    checkout_commit = get_latest_commit(project)
    repo_path = prepare_temp_refdir(project, base_refdir, checkout_commit)
    df = filter_muts_nochange_in_ftest(repo_path, latent_mut_data)
  
  if df is None: # none of the failing tests of latent mutants were changed 
    logger.warning(f"nothing to select")
    return None 
  # randomly sample XX percentage 
  logger.info(f"start selection")
  if df.shape[0] * percentage/100 < min_numSel: 
    return df, df 
  else:
    df = df.loc[df[colname].values <= np.percentile(df[colname].values, percentage)]
    numSel = min([df.shape[0], numSel]) 
    logger.info(f"Out of {df.shape[0]}, {numSel} are randomly sampled")
    return (df, df.sample(n = numSel, random_state = random_seed))

def get_latent_mut_data(outputdir:str, project:str) -> pd.DataFrame:
  """
  """
  set_default_logger()
  logger.info('Start collecting latent mutant information')
  mut_propa_stat_debt_file = os.path.join(
    outputdir, 
    f"processed/{project}.indv_mut_propagation_status_and_debt.csv") 
  df = pd.read_csv(mut_propa_stat_debt_file)
  df = df.loc[df.status == 'latent'] # 

  bidToRev, _ = analysis_utils.getBidRevDict(D4J_HOME, project)
  #common_pkg = commonGIDs[project]
  mut_kill_tests = []
  for bid in df.bid.unique():
    # get revealing info 
    #mut_rev_file = os.path.join(outputdir, f"processed/{project}.{bid}.revealed_comp_mutinfo.pkl") 
    #mut_reveal_df = pd.read_pickle(mut_rev_file)
    rev = bidToRev[bid]
    uniq_mut_file = os.path.join(outputdir, f"raw/{project}/{rev}/uniq_mutLRPair_pfile.pkl")
    with open(uniq_mut_file, 'rb') as f:
      uniq_muts = pickle.load(f)
    #mutatedFiles_pmut = {}
    #for mutated_fpath, muts in uniq_muts.items():
    #  for mutNo in muts.keys():
    #    mutatedFiles_pmut[mutNo] = mutated_fpath
    # generate which mutant (identified by mutK) is killed by which tests (list) 
    revealed_mutFiles = glob.glob(
      os.path.join(
      #outputdir, f"raw/{project}/{rev}/revealedAt.*.json"))
      old_outputdir, f"{project}/inter/{rev}/revealedAt.*.json"))
    #
    for revealed_mutFile in revealed_mutFiles:
      with open(revealed_mutFile) as f:
        rev_muts = json.load(f) # the bid (key) is 
      revealed_at = os.path.basename(revealed_mutFile).split(".")[1]
      for targetFile, rev_muts_in_file in rev_muts.items():
        for mutNo, rev_mut in rev_muts_in_file.items():
          #mutNo = int(mutNo)
          #try:
          #  targetFile = mutatedFiles_pmut[mutNo] # error here ... 
          #except KeyError: # due to redudant mut info in rev. skip this rev_mut
          #  continue 
          #targetFile_dir = os.path.dirname(targetFile).replace("/", ".")
          #start_idx = targetFile_dir.index(common_pkg)
          #targetFile = targetFile[start_idx:]
          mutK = analysis_utils.format_mutK(f"{targetFile}-{mutNo}")
          # get mut loc 
          #(init_lno, init_left, init_right) = mut_reveal_df.loc[mut_reveal_df.mutK == mutK].init_mut_loc # init_left
          try:
            uniq_mut = uniq_muts[targetFile][int(mutNo)]
          except KeyError:
            continue 
          (_, init_left, init_right) = uniq_mut['pos']
          idx_init_left = init_left - 1
          idx_init_right = init_right - 1
          mutant_cont = uniq_mut['right']
          
          (idx_revl_left, idx_revl_right) = rev_mut['mutLoc']
          # save
          mut_kill_tests.append([bid, mutK, ",".join(rev_mut['ftests']), revealed_at, 
                                 (idx_init_left, idx_init_right),
                                 (idx_revl_left, idx_revl_right), 
                                 mutant_cont
                                 ]) 

  logger.info("latent mutant data and respective failing test data collected")
  ## currently, since due to the missing info of duplicated mutant, e.g., src.main.java.org.apache.commons.lang3.StringUtils-13 and src.main.java.org.apache.commons.lang3.StringUtils-23
  ## for now, (though can be done using  analysis_utils.group_same_mutants (in check.ipynb), we will simply discard them (will handle later to match the mutK for the final)
  mut_kill_tests = pd.DataFrame(mut_kill_tests, columns = ['bid', 'mutK', 'ftests', 'revealedAt', 'init_mut_loc_idx', 'revl_mut_loc_idx', 'mut_content'])
  df_w_ftest = pd.merge(df, mut_kill_tests, how = 'left', on = ['bid', 'mutK'])
  #null_ftests = df_w_ftest.ftests.isnull().sum()
  df_w_ftest = df_w_ftest[~df_w_ftest.ftests.isnull()]
  #assert null_ftests == 0, df_w_ftest[df_w_ftest.ftests.isnull()]
  # then now, df_w_ftest has killing test information
  logger.info("latent mutant data and respective failing test data merged")
  return df_w_ftest


#def run_a_test_on_latent 
def get_file_path_at_commit(repo_path:str, target_fpath:str, from_commit:str, to_commit:str, all_commits:List) -> str:
  """
  visit commits between from_commit and to_commit in a reverse order to find the old file path at from_commit

  :param from_commit: a commit of our interest 
  :param to_commit: a commit that we start checking  
  """
  repo = pydriller.Repository(repo_path, 
    filepath = target_fpath, 
    order = 'reverse', 
    to_commit = to_commit)

  updated_target_fpath = target_fpath
  idx_to_from_commit = git_utils.getCommitIdx(all_commits, from_commit) 
  for commit in tqdm(list(repo.traverse_commits())): 
    idx_to_commit = git_utils.getCommitIdx(all_commits, commit.hash[:8])
    if idx_to_commit > idx_to_from_commit: # older than from_commit:
      break 
    for mf in commit.modified_files:
      if mf.new_path == updated_target_fpath: 
        updated_target_fpath = mf.old_path # 
  return updated_target_fpath 

def get_ftestcls_in_content(file_content:str, test_path:str, ftestcls:str) -> str:
  """
  :param file_content: a file content to look for 
  :param test_path: a test path of file_content
  :param ftest: a target ftest to look for 

  return updated ftest in file_content
  """
  pkg_name = test_modifier.get_package_name(file_content)
  assert pkg_name is not None
  parent_cls_name = os.path.dirname(
    test_path[test_path.index(pkg_name.replace(".", "/")):]
  ).replace('/', ".")
  ## work around based on the data -> for this to be complete, we need to have a line number 
  full_cls_name = parent_cls_name + "." + ftestcls
  # cls_name = os.path.basename(test_path).split(".")[0]
  #full_cls_name = parent_cls_name + "." + cls_name
  #_cls_name = ftest.split("#")[0].split(".")[-1]
  #if _cls_name != cls_name:
  #  cs = ftest.split("#")[0].split(".")
  #  full_cls_name += "." + ".".join(cs[cs.index(cls_name)+1:])
  #ftest_name = ftest.split("#")[-1]
  #ftest_in_content = f"{full_cls_name}#{ftest_name}"
  #return ftest_in_content
  return full_cls_name 

#def get_mutant_info(mut_gen_info:pd.DataFrame, mutK:str, bid, rev_at_intro, rev_at_revealed):
  #"""
  #return the mutated content and the location at intro and at reveal
  #"""
  ##revealed_mutFiles = glob.glob(
  ##  os.path.join(
  ##  old_outputdir, f"{project}/inter/{rev_at_revealed}/revealedAt.*.json"))
  #target_mut = mut_gen_info.loc[mut_gen_info.mutK == mutK]
  #target_mut.
  #pass 


def prepare_env(project:str, work_dir:str):
  """
  """
  def preprocess_lang(work_dir:str, target_fpaths:List[str]):
    mvn_utils.preprocess_lang(work_dir) 
    for target_fpath in target_fpaths:
      target_basename = os.path.basename(target_fpath)
      if target_basename == 'TypeUtils.java': return False
    return True 
  java_utils.changeJavaVer(8)
  if 'lang' == project.lower(): 
    ret = preprocess_lang(work_dir)
    if not ret: return False
  ## compile repository 
  #if ant_or_mvn == 'ant_d4j':
    #is_compiled, _compile_cmd = ant_d4jbased_utils.compile(D4J_HOME, project, work_dir)
    #is_test_compiled, _tst_compile_cmd = ant_d4jbased_utils.test_compile(D4J_HOME, project, work_dir)
    ##assert is_compiled and is_test_compiled, f"compile: {is_compiled} {_compile_cmd}, test-compile: {is_test_compiled} {_tst_compile_cmd}"
  #else: 
    #is_compiled, _ant_or_mvn, _compile_cmd = ant_mvn_utils.compile(work_dir, prefer = ant_or_mvn)
    #is_test_compiled, _ant_or_mvn, _tst_compile_cmd = ant_mvn_utils.test_compile(work_dir, prefer = ant_or_mvn)

def inject_mutant():
  pass 

# this should be run for introduced_at and revealed_at (to compare the error message )
def run_and_check_mutant(workdir:str, project:str):
  """
  :param workdir: working directory (already checked out the revision
  :param project: 
  """ 
  # prepare testing environment 
  prepared = prepare_env(project, workdir)
  # inject mutant 
  inject_mutant()
  # replace test 
#
  #revealedMuts, survivedMuts, mutDeadAts, refactoringOccurred = propagator.MutantPropagator.run(
    #workdir, 
    #mutLRPair_pmut_pfile,
    #commonGID, 
    #targetCommits, 
    #os.path.abspath(intermediate_dst),
    #_testClassPatterns = testClassPatterns,
    #ant_or_mvn = ant_or_mvn,
    #**kwargs
  #)
  # inject mutants to the fiel 
  # pitmutnant ... 
  

def run_tests_for_killCheck(project:str, sel_latent_mutdf:pd.DataFrame, repo_path:str):
  """
  run a test that reveals a latent mutant at the time of its introduction. 
  currently, we do not care about the dependency to a test method itself. 
    - on future, we will check the dependency of a test method at the time 

  :param project: 
  :param sel_latent_mutdf: a dataframe of latent mutants  
  """
  all_commits = git_utils.getAllCommits(repo_path, branch=None)
  bidToRev, _ = analysis_utils.getBidRevDict(D4J_HOME, project)
  outputs = [] # mutK, bid, rev_at_intro, rev_at_reveal, ftest, ftestpath at revealing, ftestpath at intro, error message when revealed, error message at the intro
  for _, row in sel_latent_mutdf.iterrows(): 
    bid = row.bid 
    rev_at_intro = bidToRev[bid][:8]
    rev_at_reveal = row.revealedAt
    
    # prepare mutated code 
    ... # Saved at temp and copy to the prepared direector (temp) after 

    ### 
    ftesthists = row.ftesthist 
    # get test path at the intro 
    ftestpath = row.ftestpath 
    for ftest, ftest_hist in ftesthists.items():
      test_path_at_reveal = ftestpath[ftest]
      is_introduced_flags = [is_introduced for _,_,is_introduced in ftest_hist]
      #assert sum(is_introduced_flags) in [0,1], is_introduced_flags ... no, a test method may disappear and appear again
      #for rev, test_path, is_introduced in ftest_hist:
      rev_at_last_mth_chg = ftest_hist[-1][0][:8]
      test_path_at_intro = get_file_path_at_commit(repo_path, test_path_at_reveal, rev_at_intro, rev_at_last_mth_chg, all_commits)
      new_content = None 
      if is_introduced_flags[-1]:
        # if it was newly introduced, then add the test at the end of test_file 
        # 1. first check whether a file exist (because while a test method is newly introduced, a file may exist)
        # check if exists at commit 
        if not git_utils.check_file_exist(rev_at_intro, test_path_at_intro): # if a file didn't exist, then pass 
          logger.warn(f"{test_path_at_intro} did not exist at {rev_at_intro} (bug {bid})") 
          outputs.append(
            [
              row.mutK, bid, rev_at_intro, rev_at_reveal, 
              ftest, ftestpath, None, 
              None, None 
            ]
          )
          continue # skip this one 
        else: 
          # if a test method exists, then add to the end of a file 
          ## get file content 
          content_at_latent = git_utils.show_file(rev_at_reveal, test_path_at_reveal, repo_path)
          content_at_intro = git_utils.show_file(rev_at_intro, test_path_at_intro, repo_path)
          ## append test method 
          ### get the body of a test method at the latent point 
          ftest_body_pos_at_latent = test_modifier.get_body_of_a_test(content_at_latent, ftest)
          if ftest_body_pos_at_latent is None:
            logger.warn(f"failed to find {ftest} in {test_path_at_reveal} at {rev_at_reveal}")
            continue # skip this one 
          else:
            ftest_body_at_latent, _ = ftest_body_pos_at_latent
            if test_path_at_intro == test_path_at_reveal:
              ftest_at_intro = ftest 
            else:
              ftestcls_at_intro = get_ftestcls_in_content(content_at_intro, test_path_at_intro, ftest.split("#")[0].split(".")[-1])
              ftest_name = ftest.split("#")[-1]
              ftest_at_intro =  f"{ftestcls_at_intro}#{ftest_name}"
            # inject at the end of ftest_at_intro class
            new_content = test_modifier.inject_a_new_test(content_at_intro, ftest_at_intro, ftest_body_at_latent)
      else: 
        # if already exist, then replace the content of a test method 
        content_at_latent = git_utils.show_file(rev_at_reveal, test_path_at_reveal, repo_path)
        content_at_intro = git_utils.show_file(rev_at_intro, test_path_at_intro, repo_path)
        ## get the body of a test method at the latent point 
        ftest_body_pos_at_latent = test_modifier.get_body_of_a_test(content_at_latent, ftest)
        if ftest_body_pos_at_latent is None:
          logger.warn(f"failed to find {ftest} in {test_path_at_reveal} at {rev_at_reveal}")
          continue # skip tis one 
        else:
          ftest_body_at_latent, _ = ftest_body_pos_at_latent

          ## get the position of the modification point
          ### get ftest at intro
          if test_path_at_intro == test_path_at_reveal:
            ftest_at_intro = ftest 
          else: # if not, we need to look for detaisl
            ftestcls_at_intro = get_ftestcls_in_content(content_at_intro, test_path_at_intro, ftest.split("#")[0].split(".")[-1])
            ftest_name = ftest.split("#")[-1]
            ftest_at_intro =  f"{ftestcls_at_intro}#{ftest_name}"
          # modify (replace)
          new_content = test_modifier.replace_a_test(content_at_intro, ftest_at_intro, ftest_body_at_latent)
      
      # test file overwrite at the introduction point (at test_path_at_intro)
      ## 1. go to the introduction point 
      temp_repo_path = os.path.join(os.path.dirname(repo_path), f"{project}_{bid}_temp")
      file_utils.copydir(repo_path, temp_repo_path)
      # checkout 
      git_utils.checkout(temp_test_path_at_intro, rev_at_intro)
      logger.info(f"check out to {rev_at_intro[:8]} ({project}_{bid})")

      temp_test_path_at_intro = os.path.join(temp_repo_path, test_path_at_intro)
      assert os.path.exists(temp_test_path_at_intro), temp_test_path_at_intro 
      
      ## copy the mutated code to the prepared temp working directory  
      ... 

      # now testing 
      ## testing of an orignal test (with a mutated code)
      ...

      ## testing of an updated test (with a mutated code)
      assert os.path.exists(temp_test_path_at_intro), temp_test_path_at_intro 
      logger.info(f"Update {temp_test_path_at_intro}")
      file_utils.fileWrite(new_content, temp_test_path_at_intro) # or ... new file path? 
      
    pass 

if __name__ == "__main__":
  project = sys.argv[1]
  percentage = 50 # top 25%
  colname = 'debt_time' # debt_nc
  default_numSel = 20
  default_min_numSel = 10
  
  logger = logging.getLogger(f'{project}.log')
  logger.setLevel(logging.DEBUG)
  logger.info(f"Working with {project}")
  outputdir = "output/evaluation"
  dest = os.path.join(outputdir, "killable_check")
  os.makedirs(dest, exist_ok=True)
  base_refdir = 'workdir/refs'
  
  # check whether all target latent mutant data already exists:
  target_latent_mut_file = os.path.join(dest, f"{project}.target_latentmuts.pkl")
  if False:#os.path.exists(target_latent_mut_file):
    target_latent_mut_data = pd.read_pickle(target_latent_mut_file)
    target_df_and_selected_df = select_target_mutants(
      None, target_latent_mut_data, percentage, colname, default_numSel, default_min_numSel, processed=True)
    if target_df_and_selected_df is not None:
      _, selected_df = target_df_and_selected_df
    else:
      selected_df = None
  else:
    latent_mut_data = get_latent_mut_data(outputdir, project)
    latent_mut_file = os.path.join(dest, f"{project}.all_latentmuts.pkl")
    latent_mut_data.to_pickle(latent_mut_file)
    logger.info(f"{project} {latent_mut_data.shape}")
    # select target mutants to look at 
    target_df_and_selected_df = select_target_mutants(
      base_refdir, latent_mut_data, percentage, colname, default_numSel, default_min_numSel)
    #
    if target_df_and_selected_df is not None:
      target_df, selected_df = target_df_and_selected_df
      target_df.to_pickle(target_latent_mut_file)
      logger.info(f"target latent mut data,")
      logger.info(f"   : saved to {target_latent_mut_file}")
    else:
      selected_df = None

  if selected_df is not None:
    selected_latent_mut_file = os.path.join(dest, f"{project}.{colname}_perc{percentage}.pkl")
    selected_df.to_pickle(selected_latent_mut_file)
    logger.info(f"selected target latent mut data,")
    logger.info(f"   :saved to {selected_latent_mut_file}")
  
  # run tests 
  # ... 
  #run_test_for_killCheck()
  for _, row in selected_df.iterrows():
    #run_tests_for_killCheck(project:str, sel_latent_mutdf:pd.DataFrame, repo_path:str)
    pass 
    