"""
A main script to generate artificial mutation-based bug-inducing commits
    - target: Defects4J 
    - refactoring technique: RefactorMiner (RMiner)

To check whether a mutant is killed or not, we have to check whether there is a difference between 
the status of the set of covered tests (tests that covered the mutated method) with and without the mutant
"""
import os, sys 
import pandas as pd 
import pickle 
from utils import git_utils, java_utils, file_utils, mvn_utils
from utils import ant_d4jbased_utils, ant_mvn_utils
from typing import List, Dict, Tuple
import refactor
import multiprocessing as mp
import shutil 
import traceback, logging
import mutants.pitMutationTool as pitMutationTool

commonGIDs = {
    'Lang':'org.apache.commons',
    'Math':'org.apache.commons',
    'Time':'org.joda.time',  # 
    'Compress':'org.apache.commons', 
    'Cli':'org.apache.common',
    'Codec':'org.apache.commons', 
    'Gson':'com.google.gson',
    'Closure':'com.google', 
    'Collections':'org.apache.commons', 
    'Jsoup':'org.apache.commons', 
    'Csv':'org.apache.commons', 
    'Mockito':'org.mockito', 
    'JxPath':'org.apache.commons',
    'JacksonXml':'com.fasterxml', 
    'JacksonCore':'com.fasterxml', 
    'JacksonDatabind':'com.fasterxml', 
    'Jsoup':'org.jsoup'
}

def getToSaveFiles(dest:str, final_output_key:str) -> Tuple[str]:
    revealed_mut_file = os.path.join(dest, f"{final_output_key}.revealed.pkl")
    surv_mut_file = os.path.join(dest, f"{final_output_key}.survived.pkl")
    mut_deadat_file = os.path.join(dest, f"{final_output_key}.mutDeadAt.pkl")
    refactoring_file = os.path.join(dest, f"{final_output_key}.refactorings.pkl")
    return (revealed_mut_file, surv_mut_file, mut_deadat_file, refactoring_file)

def checkAllProcessd(
    revealed_mut_file:str, 
    surv_mut_file:str, 
    mut_deadat_file:str, 
    refactoring_file:str
) -> bool:
    all_processed = os.path.exists(revealed_mut_file) and (
                os.path.exists(surv_mut_file)) and (
                os.path.exists(mut_deadat_file)) and (
                os.path.exists(refactoring_file))
    return all_processed

def getTargetedLnos(targetDir:str, project:str, k:str = "flex") -> Dict[int, pd.DataFrame]:
    targetFile = os.path.join(targetDir, f"{project}.{k}.targetedLos.json")
    with open(targetFile) as f:
        import json 
        data = json.load(f)
    targetedLnoInfos = {}
    for k,(c,v) in data.items():
        targetedLnoInfos[int(k)] = (c, pd.DataFrame(v))
    return targetedLnoInfos

def isInLno(start_end_chars_cnt_dict, end_chars_cnt) -> int:
    ret_lno = None
    for lno, (_, end) in start_end_chars_cnt_dict.items():
        if end_chars_cnt <= end: 
            ret_lno = lno # first within 
            break 
    assert ret_lno is not None, f"{max([vs[-1] for vs in start_end_chars_cnt_dict.items()])} vs {end_chars_cnt}"
    return ret_lno

def excludeRedundantMutants(mutLRPair_pmut_pfile:Dict[str, Dict]) -> Dict[str, Dict]:
    import numpy as np 
    np.random.seed(0)
    processed = {} # key = (fpath-is_neg-pos[0]-pos[1]-right), value = 
    n_total_muts = 0
    for targetFile, mutInfos in mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            # 'left', 'right', 'pos', 'mutOp', 'targeted', 'text'
            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            try:
                processed[k].append((targetFile, mutNo))
            except KeyError:
                processed[k] = [(targetFile, mutNo)]
            n_total_muts += 1
             
    # drop redundant 
    n_uniq_muts = 0
    uniq_mutants = {}
    for k, the_same_muts in processed.items(): 
        indices = np.arange(len(the_same_muts))
        selected = the_same_muts[np.random.choice(indices, 1)[0]]
        fpath, mutNo = selected
        if fpath not in uniq_mutants.keys():
            uniq_mutants[fpath] = {}
        uniq_mutants[fpath][mutNo] = mutLRPair_pmut_pfile[fpath][mutNo]
        n_uniq_muts += 1
    # logging 
    print (f"From {n_total_muts}, {n_uniq_muts} mutants remained")
    return uniq_mutants 

def run_w_pit(
    work_dir:str, 
    project:str, 
    targeted:Dict, 
    commonGID:str, 
    pit_jar_path:str = "mutants/pit/pitest-command-line-1.7.4-SNAPSHOT-jar-with-dependencies.jar",
    dest:str = "outputs", 
    with_refactor:bool = True, 
    d4j_home:str = None, 
    bid:int = None, 
    ant_or_mvn:str = True, 
    use_strict:bool = False
):
    # Dict: key = target_fpaht, value = test class pattern
    target_fpaths = list(targeted.keys())
    if d4j_home is None:
        testClassPatterns = java_utils.getTestClassPats(target_fpaths, commonGID) 
    else:
        assert bid is not None
        testClassPatterns = java_utils.getTestClassPats_d4j(d4j_home, project, bid)
        
    commit_hexsha = git_utils.get_current_commit(work_dir) # where the mutant is
    final_output_key = f"{project}.{commit_hexsha[:8]}"
    os.makedirs(dest, exist_ok=True)
    intermediate_dst = os.path.join(dest, f"inter/{commit_hexsha[:8]}")
    os.makedirs(intermediate_dst, exist_ok=True)
    mutLRPair_file = os.path.join(intermediate_dst, "mutLRPair_pfile.pkl")
    strict_mutLRPair_file = os.path.join(intermediate_dst, "strict." + os.path.basename(mutLRPair_file))
    uniq_mutLRPair_file = os.path.join(intermediate_dst, "uniq_" + os.path.basename(mutLRPair_file))
    
    if os.path.exists(mutLRPair_file): # already exist, then use it 
        ## temporary
        if not use_strict:
            if os.path.exists(uniq_mutLRPair_file):
                print (f"ML Pair: {uniq_mutLRPair_file}")
                with open(uniq_mutLRPair_file, 'rb') as f:
                    mutLRPair_pmut_pfile = pickle.load(f)
            else:
                print (f"ML Pair: {mutLRPair_file}")
                with open(mutLRPair_file, 'rb') as f:
                    mutLRPair_pmut_pfile = pickle.load(f)
                mutLRPair_pmut_pfile = excludeRedundantMutants(mutLRPair_pmut_pfile)
                with open(uniq_mutLRPair_file, 'wb') as f:
                    pickle.dump(mutLRPair_pmut_pfile, f)
        else:
            print (f"ML Pair: {strict_mutLRPair_file}")
            with open(strict_mutLRPair_file, 'rb') as f:
                mutLRPair_pmut_pfile = pickle.load(f)
    else:
        java_utils.changeJavaVer(8)
        if 'lang' == project.lower(): 
            mvn_utils.preprocess_lang(work_dir) # move TypeUtilTests (for lang) 
            for target_fpath in target_fpaths:
                target_basename = os.path.basename(target_fpath)
                if target_basename == 'TypeUtils.java':
                    print (f"Can't process this one: {bid}")
                    return False
        # compile repository 
        if ant_or_mvn == 'ant_d4j':
            is_compiled, _compile_cmd = ant_d4jbased_utils.compile(d4j_home, project, work_dir)
            is_test_compiled, _tst_compile_cmd = ant_d4jbased_utils.test_compile(d4j_home, project, work_dir)
            assert is_compiled and is_test_compiled, f"compile: {is_compiled} {_compile_cmd}, test-compile: {is_test_compiled} {_tst_compile_cmd}"
        else: 
            is_compiled, _ant_or_mvn, _compile_cmd = ant_mvn_utils.compile(work_dir, prefer = ant_or_mvn)
            assert is_compiled, f"compile: {ant_or_mvn}: {_compile_cmd}"
            assert _ant_or_mvn == ant_or_mvn, f'{_ant_or_mvn} vs {ant_or_mvn} while compiling'
            is_test_compiled, _ant_or_mvn, _tst_compile_cmd = ant_mvn_utils.test_compile(work_dir, prefer = ant_or_mvn)
            assert is_test_compiled, f"test compile: {ant_or_mvn}: {_tst_compile_cmd}"
            assert _ant_or_mvn == ant_or_mvn, f'{_ant_or_mvn} vs {ant_or_mvn} while compiling tests'

        targetFiles = list(targeted.keys())
        if project == 'Lang':
            targetTests = "*Test"
        else:
            targetTests = java_utils.getTestClassPats_d4j(d4j_home, project, bid)
        targetClasses = java_utils.getTargetClasses_d4j(d4j_home, project, bid, targetFiles)
        #
        if ant_or_mvn == 'ant_d4j':
            _classPath = ant_d4jbased_utils.export_compileClassPath_d4jbased(d4j_home, project, work_dir)
            _classPath_ts = set(_classPath.split(","))
            testClassPath = ant_d4jbased_utils.export_compileTestClassPath_d4jbased(d4j_home, project, work_dir)
            testClassPath_ts = set(testClassPath.split(","))
            if _classPath_ts - testClassPath_ts: 
                classPath = testClassPath
            else:
                classPath = ",".join(_classPath_ts.union(testClassPath_ts))
            if ant_d4jbased_utils.check_set_SourceDirs_d4jbased(d4j_home, project, work_dir):
                sourceDirs = ant_d4jbased_utils.export_SourceDirs_d4jbased(d4j_home, project, work_dir)
            else:
                # temporary 
                if project == 'Closure':
                    sourceDirs = "src" 
                else:
                    if os.path.exists(os.path.join(work_dir, "src/java")):
                        sourceDirs = "src/java"
                    elif os.path.exists(os.path.join(work_dir, "src/main/java")):
                        sourceDirs = "src/main/java"
                    elif os.path.exists(os.path.join(work_dir, "src")):
                        sourceDirs = "src"
                    else:
                        print (f"Need to implement ....")
                        sys.exit()
            kwargs = {
                'pit_jar_path':pit_jar_path, 
                'classPath':classPath, 
                'sourceDirs':sourceDirs, 
            }
        else: # mvn 
            kwargs = {}
        #
        kwargs['further'] = True
        # run mutation 
        mutLRPair_pmut_pfile = pitMutationTool.PitMutantProcessor.runMutation(
            work_dir, 
            targetFiles, 
            targetClasses, 
            targetTests, 
            is_mvn = "mvn" == ant_or_mvn, 
            mutators_config ='all', 
            **kwargs, 
        )
        mutation_file = pitMutationTool.PitMutantProcessor.getMutationResultFile(work_dir, is_mvn = "mvn" == ant_or_mvn)
        if os.path.exists(mutation_file):
            to_save_file = os.path.join(intermediate_dst, os.path.basename(mutation_file))
            shutil.move(mutation_file, to_save_file)
            print (f"Move {mutation_file} to {to_save_file} for {project} {bid}")
        
        with open(mutLRPair_file, 'wb') as f:
            pickle.dump(mutLRPair_pmut_pfile, f)

        mutLRPair_pmut_pfile = excludeRedundantMutants(mutLRPair_pmut_pfile)
        ###### #############################
        with open(uniq_mutLRPair_file, 'wb') as f:
            pickle.dump(mutLRPair_pmut_pfile, f)  


    if with_refactor:
        # if no mut, no need to track and process the file 
        mutLRPair_pmut_pfile = {
            targetFile:info for targetFile, info in mutLRPair_pmut_pfile.items() if len(info) > 0}
        target_fpaths = list(mutLRPair_pmut_pfile.keys())
        revealed_mut_file, surv_mut_file, mut_deadat_file, refactoring_file = getToSaveFiles(dest, final_output_key)
        if checkAllProcessd(
            revealed_mut_file, 
            surv_mut_file, 
            mut_deadat_file, 
            refactoring_file
        ):
            print (f"{final_output_key} alread processed")
            return True
        
        targetCommits = git_utils.get_commits_upto_recent(
            work_dir, commit_hexsha, 'trunk' if project in ['Lang', 'Math'] else 'master')
        # copy workdir
        cp_work_dir = os.path.join(
            os.path.dirname(work_dir), 
            os.path.basename(work_dir) + "temp2") # b/c we don't want to tarnish the original one ..
        file_utils.copydir(work_dir, cp_work_dir) 
        # delete mutants directory here
        if os.path.exists(os.path.join(work_dir, "mutants")):
            shutil.rmtree(os.path.join(work_dir, "mutants")) # b/c for refactoring not needed 
        
        print (f"Refactoring ... : {len(targetCommits)} commits")
        print (mutLRPair_pmut_pfile.keys())
        #if project in ['Math', 'Closure', 'Mockito', 'Collections']:
        if project in ['Math', 'Closure', 'Mockito']:#, 'Collections']:
            ant_or_mvn = 'ant_d4j'
            kwargs = {
                'd4j_home':d4j_home, 
                'project':project
            }
        else:
            ant_or_mvn = 'mvn'
            kwargs = {}

        revealedMuts, survivedMuts, mutDeadAts, refactoringOccurred = refactor.RMinerProcesser.run(
            cp_work_dir, 
            mutLRPair_pmut_pfile,
            commonGID, 
            targetCommits, 
            os.path.abspath(intermediate_dst),
            use_sdk = False,
            _testClassPatterns = testClassPatterns,
            ant_or_mvn = ant_or_mvn,
            which_mutant = 'pit', 
            **kwargs
        )
        print (f"Delete temp directory ...: {cp_work_dir}")
        try:
            #import shutil
            if os.path.exists(cp_work_dir):
                shutil.rmtree(cp_work_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        try:
            #import shutil
            if os.path.exists(os.path.abspath(work_dir)):
                shutil.rmtree(os.path.abspath(work_dir))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        #
        print (f"Save refactoring-aware mutation propagation results to {dest} with {final_output_key}")
        revealed_mut_file = os.path.join(dest, f"{final_output_key}.revealed.pkl")
        surv_mut_file = os.path.join(dest, f"{final_output_key}.survived.pkl")
        mut_deadat_file = os.path.join(dest, f"{final_output_key}.mutDeadAt.pkl")
        refactoring_file = os.path.join(dest, f"{final_output_key}.refactorings.pkl")
        with open(revealed_mut_file, 'wb') as f:
            pickle.dump(revealedMuts, f)
        print (f"Save to {revealed_mut_file}")
        #
        with open(surv_mut_file, 'wb') as f:
            pickle.dump(survivedMuts, f)
        print (f"Save to {surv_mut_file}")
        #
        with open(mut_deadat_file, 'wb') as f:
            pickle.dump(mutDeadAts, f)
        print (f"Save to {mut_deadat_file}")
        #
        with open(refactoring_file, 'wb') as f:
            pickle.dump(refactoringOccurred, f) 
        print (f"Save to {refactoring_file}")
    else:
        try:
            if os.path.exists(os.path.abspath(work_dir)):
                shutil.rmtree(os.path.abspath(work_dir))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    return True

def getWorkDir(basedir:str, project:str, bid:int) -> str:
    workdir = os.path.join(basedir, f"{project}{bid}")
    return workdir

def prepare_workdir(
    d4j_home:str, project:str,
    bid:int, fixedRev:str, basedir:str) -> str:
    import subprocess 
    from subprocess import CalledProcessError
    #workdir = os.path.join(basedir, f"{project}{bid}")
    workdir = getWorkDir(basedir, project, bid)
    if os.path.exists(workdir):
        print (f"{workdir} already exists")
        git_utils.checkout(workdir, fixedRev)
        return os.path.abspath(workdir)
    else:
        cmd = f"{d4j_home}/framework/bin/defects4j checkout -p {project} -v {bid}f -w {project}{bid}"
        try:
            out = subprocess.run(cmd, shell = True, cwd = basedir)
        except CalledProcessError as e:
            print (out)
            return None 
        workdir = os.path.join(basedir, f"{project}{bid}")
        git_utils.checkout(workdir, fixedRev) 
        if workdir[-1] == "/": workdir = workdir[:-1]
        return os.path.abspath(workdir)


def run_all(
    d4j_home:str, project:str, targetdir:str, 
    root_work_dir:str, 
    #mml_fpath:str = "mutants/mml/likely_default.mml.bin",
    mutSelMth:str = "all",
    dest:str = "outputs", 
    with_refactor:bool = True, 
    indices:List[int] = None ,
    ant_or_mvn:str = 'mvn', 
    mod:int = None, 
    kIdx:int = None, 
):
    os.makedirs(root_work_dir, exist_ok=True)
    #if project == 'Gson':
    target_info_file = os.path.join(targetdir, f"{project}_all_targets.json")
    with open(target_info_file) as f:
        import json
        targeted = json.load(f)
        targeted = dict(sorted(targeted.items(), key = lambda v:int(v[0].split("-")[0])))

    targetedLnoInfosPbug = getTargetedLnos(targetdir, project, k = 'incl.refactor.incl.mutAt.wo_semcheck')# "flex")
    failed = []
    exclude_bids = []
    to_uses = None
    ## temporary to handle those interesting first 

    for bid, (_, df) in targetedLnoInfosPbug.items():
        if len(df) == 0:
            exclude_bids.append(bid)
    print ('exclude', exclude_bids)
    for i, (bid_fixedRev, target_files) in enumerate(list(targeted.items())):
        bid, fixedRev = bid_fixedRev.split("-")
        bid = int(bid)
        if toFocus is not None and bid not in toFocus:
            print (f"For {project}, {bid} is not in focus")
            continue
        
        if to_uses is not None and bid not in to_uses:
            print (f"For {project}, {bid} can't be the target")
            continue

        if (indices is not None) and (bid not in indices): continue
        if bid in exclude_bids: print (f"Nothing to look at {bid}"); continue
        if mod is not None:
            assert kIdx is not None, kIdx
            if bid % mod != kIdx:
                continue 
        ##
        _, targetedLnoInfos = targetedLnoInfosPbug[bid]
        validTargetFiles = targetedLnoInfos.fpath.unique()
        print (f"Out of {len(target_files)}, {len(validTargetFiles)} are valid targets for {bid}")
        if len(targetedLnoInfos) == 0: 
            print (f"\tThere are no valid lines, i.e., changed lines, in {bid}")
            #continue
        from subprocess import CalledProcessError
        try:
            is_success, bid = prepare_and_run(
                bid_fixedRev,
                #target_files, 
                targetedLnoInfos[['fpath', 'lno']], 
                d4j_home, project,
                root_work_dir,
                #mml_fpath = mml_fpath, 
                mutSelMth = mutSelMth, 
                dest = dest, 
                with_refactor = with_refactor, 
                ant_or_mvn = ant_or_mvn
            )
        except (Exception, CalledProcessError) as e:
            print (f"Error while processing {bid} {fixedRev}")
            print (e)
            logging.error(traceback.format_exc())
            work_dir = getWorkDir(root_work_dir, project, bid)
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            if os.path.exists(work_dir + "temp2"):
                shutil.rmtree(work_dir + "temp2")
            failed.append(bid)
            continue
        #if not is_success: # will cover both the failure in prepare_workdir and run
    n_targeted = len(targeted) if indices is None else len(indices)
    print (f"Out of {n_targeted}, {len(failed)} failed")
    print ("Done!")
    return failed

def prepare_and_run(
    bid_fixedRev:str,
    targeted:pd.DataFrame, 
    d4j_home:str, project:str,
    root_work_dir:str,
    dest:str = "outputs",
    with_refactor:bool = True,
    ant_or_mvn:str = 'mvn'
) -> Tuple[bool, int]:
    bid, fixedRev = bid_fixedRev.split("-")
    print (f'Processing')
    ## prepare the play grund
    work_dir = prepare_workdir(d4j_home, project, bid, fixedRev, root_work_dir)
    if work_dir is None:
        print (f"Failed to process ")
        return (False, bid)
    print (f"Processing {bid}")
    ## formulated targeted to Dict
    targetedDict = {}
    for fpath, lno in targeted[['fpath', 'lno']].values:
        try:
            targetedDict[fpath].append(lno)
        except KeyError:
            targetedDict[fpath] = [lno]
    
    success = run_w_pit(
        work_dir, 
        project, 
        targetedDict, 
        commonGIDs[project],
        pit_jar_path = os.path.abspath(
            "mutants/pit/pitest-command-line-1.7.4-SNAPSHOT-jar-with-dependencies.jar"),
        dest = dest, 
        with_refactor = with_refactor, 
        d4j_home = d4j_home, 
        bid = bid, 
        ant_or_mvn = ant_or_mvn, 
    )

    if success:
        return (True, bid)
    else:
        return (False, bid)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("-dst", "--dest", type = str, 
        help = "a directory to save colleted bug-inducing commits")
    parser.add_argument("-p", "--project", type = str, help = "a targeted project name")
    #parser.add_argument("-data", "--datadir", type = str)
    parser.add_argument("-repo", "--repo_path", type = str, help = "this is infact a repo directory")
    parser.add_argument("-target", "--targetdir", type = str, default="data/targets")
    parser.add_argument("-sdk", "--use_sdk", action="store_true")
    parser.add_argument("-w", "--workdir", type = str, default = "workdir")
    parser.add_argument("-b", "--bid", type =int)
    parser.add_argument("-refactor", "--with_refactor", action = "store_true")
    parser.add_argument("-m", "--mod", type = int, default = None)
    parser.add_argument("-i", "--kIdx", type = int, default = None)
    args = parser.parse_args()

    dest = args.dest 
    dest = os.path.join(dest, args.project)
    os.makedirs(dest, exist_ok=True)
    indices = [args.bid]
    run_all(
        os.getenv('D4J_HOME'),
        args.project,
        args.targetdir,
        args.workdir,
        mutSelMth = "all",
        dest = dest, 
        with_refactor = args.with_refactor,#True,
        indices = indices,
        ant_or_mvn = 'ant_d4j' if args.project in ['Math', 'Closure', 'Mockito'] else 'mvn',
        mod = args.mod, 
        kIdx = args.kIdx
    )

