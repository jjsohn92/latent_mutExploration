import pandas as pd 
import pickle, json  
import os, glob 
import sys 
from main import prepare_workdir 
from typing import List, Dict, Tuple, Union
import utils.git_utils as git_utils
import utils.analysis_utils as analysis_utils
import utils.semantic_checker as semantic_checker
import numpy as np
from tqdm import tqdm 

TEMP_WORKDIR = "temp"
TEMP_K = "temp"
CHGDIR = "output/temp/new"

D4J_HOME = os.getenv('D4J_HOME')
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

NEED_TO_FILTER_MUTOPS = ['CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 'INLINE_CONSTS']

def prepare_toBlameDir(
    d4j_home:str, project:str,
    bid:int, targetRev:str, basedir:str
) -> str:
    os.makedirs(basedir, exist_ok = True)
    forBlameDir = os.path.join(basedir, f"{project}_{targetRev}") # this should be the directory name
    if os.path.exists(forBlameDir):
        return forBlameDir
    else:
        checkoutdir = prepare_workdir(d4j_home, project, bid, targetRev, basedir)
        import shutil
        shutil.move(checkoutdir, forBlameDir)
        return forBlameDir

def getLineNos(
    mut_start_pos, mut_end_pos, 
    work_dir:str, inputfile:str
) -> List[str]:
    """
    """
    from mutants.matcher import initTreeAndRootSRCML
    _, start_end_pos_pline = initTreeAndRootSRCML(inputfile, os.path.abspath(work_dir))
    modified_lnos = []
    for lno, (start_pos, end_pos) in start_end_pos_pline.items():
        if (start_pos <= mut_start_pos) and (mut_start_pos <= end_pos):
            modified_lnos.append(lno)
        if (start_pos <= mut_end_pos) and (mut_end_pos <= end_pos):
            modified_lnos.append(lno)
    modified_lnos = list(set(modified_lnos))
    return modified_lnos

def getAndSaveTempFile(
    repo_path:str, fpath:str, rev:str, k:str
) -> str:
    from utils.git_utils import show_file
    content = show_file(rev, fpath, repo_path)
    temp_file = os.path.join(TEMP_WORKDIR, f"{k}.java")
    with open(temp_file, 'w') as f:
        f.write(content)
    return temp_file 

def getStartEndParseInfo(at:str, repo_path:str, inputfile:str) -> Dict:
    from mutants.matcher import initTreeAndRootSRCML
    temp_file = getAndSaveTempFile(
        repo_path, inputfile, at, f"{TEMP_K}-{at}-{os.path.basename(inputfile)[:-5]}")
    print ('temp file', temp_file)
    _, start_end_pos_pline = initTreeAndRootSRCML(
        os.path.abspath(temp_file), os.path.abspath(TEMP_WORKDIR))  
    return start_end_pos_pline 

def getLineNos_byRepo(
    #at, 
    start_end_pos_pline:Dict, 
    t_start_pos, t_end_pos, 
    #repo_path:str, 
    #inputfile:str, 
) -> List[str]:
    """
    """
    #from mutants.matcher import initTreeAndRootSRCML
    #temp_file = getAndSaveTempFile(
    #    repo_path, inputfile, at, f"{TEMP_K}_{rev}-{os.path.basename(inputfile)[:-5]}")
    #_, start_end_pos_pline = initTreeAndRootSRCML(
    #    os.path.abspath(temp_file), os.path.abspath(TEMP_WORKDIR))
    modified_lnos = []
    for lno, (start_pos, end_pos) in start_end_pos_pline.items():
        if (start_pos <= t_start_pos) and (t_end_pos <= end_pos):
            modified_lnos.append(lno)
        if (start_pos <= t_end_pos) and (t_end_pos <= end_pos):
            modified_lnos.append(lno)
    modified_lnos = list(set(modified_lnos))
    #if len(modified_lnos) == 0:
    #    print (at, t_start_pos, t_end_pos)
    #    with open('temp.json', 'w') as f:
    #        import json
    #        f.write(json.dumps(start_end_pos_pline))
    #print (modified_lnos)
    return modified_lnos

def checkWhetherLnoChgdAft(
    datadir:str, project:str, 
    start_commit_hash:str, end_commit_hash:str, 
    mutatedAtLno:int, fpath:str, 
) -> bool:
    chgdInfoFilePat = os.path.join(datadir, f"{project}/{project}.*.{start_commit_hash}.chgdInfo.pkl")
    chgdInfoFiles = glob.glob(chgdInfoFilePat)
    assert len(chgdInfoFiles) == 1, chgdInfoFilePat
    chgdInfoFile = chgdInfoFiles[0]
    import pickle 
    with open(chgdInfoFile, 'rb') as f:
        chgdInfo = pickle.load(f)
    
    all_commits = git_utils.getAllCommits(PROJECT_REPOS[project], None)
    idx_to_start = all_commits.index(start_commit_hash)
    try:
        idx_to_end = all_commits.index(end_commit_hash)
    except ValueError: # e.g., len(end_commit_has) < 8
        idx_to_end = None
        for idx, c in enumerate(all_commits):
            if c.startswith(end_commit_hash) or end_commit_hash.startswith(c):
                idx_to_end = idx 
                break 
        assert idx_to_end is not None, end_commit_hash 
    chgdAtInfo = chgdInfo[fpath][mutatedAtLno][0]
    if len(chgdAtInfo) > 0:
        if isinstance(chgdAtInfo[0], List): # 
            for chgdAtInfoPcommit in chgdAtInfo:
                _, chgdAt = chgdAtInfoPcommit
                idx_to_chgdAt = all_commits.index(chgdAt)
                if (idx_to_chgdAt >= idx_to_end) and (idx_to_chgdAt < idx_to_start): 
                    return True 
        else:
            _, chgdAt = chgdAtInfoPcommit
            idx_to_chgdAt = all_commits.index(chgdAt)
            if (idx_to_chgdAt >= idx_to_end) and (idx_to_chgdAt < idx_to_start): 
                return True
    return False 


def isInLno(start_end_chars_cnt_dict, end_chars_cnt) -> int:
    #end_chars_idx = end_chars_cnt - 1 
    ret_lno = None
    for lno, (_, end) in start_end_chars_cnt_dict.items():
        if end_chars_cnt <= end: 
            ret_lno = lno # first within 
            break 
    assert ret_lno is not None, f"{np.max([vs[-1] for vs in start_end_chars_cnt_dict.items()])} vs {end_chars_cnt}"
    return ret_lno


def getMatchedFile(repo_path:str, rev:str, targetFpath:str, renamedFiles:Dict[str, str] = None): 
    repo = git_utils.get_repo(repo_path)
    files = git_utils.list_files_in_commit(repo.commit(rev))
    #target_basename = os.path.basename(targetFpath)
    target_basename = "/".join(targetFpath.split("/")[-2:])
    #print ("base", target_basename)
    #print ("files", files)
    ret = None
    for file in files:
        if not file.endswith('.java'):
            continue
        #basename = os.path.basename(file)
        #if basename == target_basename: ## temporary
        if file.endswith(target_basename):
            #print (file, target_basename, file.endswith(target_basename))
            #return file
            ret = file 
    # temporary -> will be replaced with renaming tracker
    if ret is None:
        target_basename = os.path.basename(target_basename)
        for file in files:
            if not file.endswith('.java'):
                continue 
            if file.endswith(target_basename):
                ret = file 
    # temporary endss
    return ret 

def getMatchedFile_v2(repo_path:str, rev:str, targetFpath:str): 
    repo = git_utils.get_repo(repo_path)
    files = git_utils.list_files_in_commit(repo.commit(rev))
    target_basename = os.path.basename(targetFpath) #"/".join(targetFpath.split("/")[-2:])
    ret = None
    matched = []
    for file in files:
        if not file.endswith('.java'):
            continue
        file_basename = os.path.basename(file)
        #if file.endswith(target_basename):
        if file_basename == target_basename:
            matched.append(file)
    if len(matched) == 1:
        return matched[0]
    else:
        import jellyfish
        ret_score = None
        ret = None 
        for cand in matched:
            score = jellyfish.levenshtein_distance(cand, targetFpath)
            if (ret_score is None) or (ret_score < score):
                ret_score = score 
                ret = cand 
        return ret 

#def getFileRenamed(
    #outputdir:str, rev:str, all_commits:List[str]
#) -> Dict[str,List[Tuple[str,str]]]:
    #diffMapFiles = glob.glob(
        #os.path.join(outputdir, f"{rev}/diffMap*.json"))
    #diffMapFiles_w_index = []
    #for diffMapFile in diffMapFiles:
        #rev_at_diff = os.path.basename(diffMapFile).split("_")[1].split(".")[0]
        #idx_to_diff = all_commits.index(rev_at_diff)
        #diffMapFiles_w_index.append([diffMapFile, idx_to_diff])
#    
    ##print ('input', diffMapFiles_w_index)
    #diffMapFiles_w_index = sorted(
        #diffMapFiles_w_index, 
        #key = lambda v:v[1], reverse = True) # oldest to recent 
    #renameds = {}
    #for i, (diffMapFile, idx_to_diff) in enumerate(diffMapFiles_w_index): # oldest ot recent
        #with open(diffMapFile) as f:
            #diffMaps = json.load(f)
        #print (diffMapFile, diffMaps.keys())
        #prevFpath_currFpath_pairs = list(diffMaps.keys())
        #if len(prevFpath_currFpath_pairs) == 0:
            #continue 
        #for prevFpath_currFpath in prevFpath_currFpath_pairs:
            #prevFpath, currFpath = prevFpath_currFpath.split("-")
            #if prevFpath == currFpath:
                #continue # for this case, no need for any recodring
            #if i == 0:
                #renameds[prevFpath] = [(prevFpath, all_commits[idx_to_diff])]
            #else:
                ###
                #for org_fpath in renameds.keys():
                    #renameds_of_fpath = renameds[org_fpath]
                    #latest = renameds_of_fpath[-1]
                    #if latest == prevFpath:
                        #renameds[org_fpath].append((currFpath, all_commits[idx_to_diff]))
                    ### due to 
                    #else:
                        #org_fpath_basename = os.path.basename(org_fpath)
                        #prev_fpath_basename = os.path.basename(prevFpath)
    #return renameds

#def getRenamedFile(
    #renameds_ofFile:List[Tuple[str, str]], commit:str, all_commits:List[str]
#) -> str:
    #idx_to_commit = all_commits.index(commit)
    #renamed_fpath_at_commit = None
    #for renamed_fpath, commit_at_renamed in renameds_ofFile: # insertd in the oldest to recent
        #idx_to_commit_at_renamed = all_commits.index(commit_at_renamed)
        #if idx_to_commit_at_renamed <= idx_to_commit:
            #renamed_fpath_at_commit = renamed_fpath
            #break 
#
    #if renamed_fpath_at_commit is None: # meaning commit is more recent than any of the previous renamed at
        #renamed_fpath_at_commit = renameds_ofFile[-1][0] # take the most recent 
    #return  renamed_fpath_at_commit


def getMatchedFile_v2(
    repo_path:str, rev:str, targetFpath:str, 
    renamedFiles:Dict[str, List[str]] = None
): 
    repo = git_utils.get_repo(repo_path)
    files = git_utils.list_files_in_commit(repo.commit(rev))
    target_basename = "/".join(targetFpath.split("/")[-2:])
    ret = None
    for file in files:
        if not file.endswith('.java'):
            continue
        if file.endswith(target_basename):
            ret = file 
    # temporary -> will be replaced with renaming tracker
    if ret is None:
        target_basename = os.path.basename(target_basename)
        for file in files:
            if not file.endswith('.java'):
                continue 
            if file.endswith(target_basename):
                ret = file 
    # temporary endss
    return ret 

def get_mut_files_at_reveal_from_appliedAts(fpath, mut_revealing_info:Dict, 
    appliedAtInfos_of_fpath:Dict, mapped_to_uniq_mutK:Dict) -> Dict[int, str]: 
    rets = {}
    for mutNo, revealInfos in mut_revealing_info.items():
        matched_fpath_at_appliedAt = None
        mutLoc = revealInfos['mutLoc']
        try:
            mut_appliedAt_infos = appliedAtInfos_of_fpath[str(mutNo)]
        except KeyError:
            uniq_mutK = mapped_to_uniq_mutK[analysis_utils.format_mutK(f"{fpath}-{mutNo}")]
            for k,v in mapped_to_uniq_mutK.items():
                if v == uniq_mutK:
                    try:
                        _ = appliedAtInfos_of_fpath[k.split("-")[1]]
                    except KeyError:
                        continue
                    matched_fpath_at_appliedAt = v.split("-")[0].replace(".", "/") + ".java"
                    break 
        if matched_fpath_at_appliedAt is None:
            for fpath_at_appliedAt, mut_appliedAt_info in mut_appliedAt_infos.items():
                for _, _mutLoc in mut_appliedAt_info:
                    if _mutLoc[0] == mutLoc[0] and _mutLoc[1] == mutLoc[1]:
                        matched_fpath_at_appliedAt = fpath_at_appliedAt
                        break 
                if matched_fpath_at_appliedAt is not None: break 
        # the mutant itself can be propated to mutliple files and loc, however, here the goal is
        # to get the location WHEN it was revealed (once mutant is reveald, it is not further processed)
        rets[int(mutNo)] = matched_fpath_at_appliedAt
    return rets 


#def getRefactedAt(outputdir:str, project:str, rev:str) -> List[str]:
    #refactoring_file = os.path.join(outputdir, f"{project}.{rev}.refactorings.pkl")
    #refactorings = pd.read_pickle(refactoring_file)
    ##for refactoredAt, infos in refactorings.items():
    ##    pass 
    #refactoredAts = list(refactorings.keys())
    #return refactoredAts
def getRefactedAt(outputdir:str, rev:str) -> List[str]:
    refactoredAts = []
    for file in glob.glob(
        os.path.join(outputdir, f"{rev}/refactorings_*.json")
    ):
        rev = os.path.basename(file).split("_")[-1].split(".")[0]
        with open(file) as f:
            data = json.load(f) 
        ds = data['commits']
        for d in ds:
            if len(d['refactorings']) > 0:
                for r in d['refactorings']:
                    if len(r['leftSideLocations']) + len(r['rightSideLocations']) > 0: 
                        refactoredAts.append(rev)
    refactoredAts = list(set(refactoredAts))
    return refactoredAts

def getRefactedAtDeep(outputdir:str, rev:str) -> List[str]:
    """
    further check for the line numbers
    """
    refactoredAts = {}
    for file in glob.glob(
        os.path.join(outputdir, f"{rev}/refactorings_*.json")
    ):
        rev = os.path.basename(file).split("_")[-1].split(".")[0]
        with open(file) as f:
            data = json.load(f) 
        ds = data['commits']
        for d in ds:
            if len(d['refactorings']) > 0:
                for r in d['refactorings']:
                    #if len(r['leftSideLocations']) + len(r['rightSideLocations']) > 0: 
                    if len(r['rightSideLocations']) > 0: 
                        try:
                            _ = refactoredAts[rev]
                        except KeyError:
                            refactoredAts[rev] = {}
                        for ar in r['rightSideLocations']:
                            fpath = ar['filePath']
                            start_lno = ar['startLine']
                            end_lno = ar['endLine']
                            lnos = list(range(start_lno, end_lno+1))
                            try:
                                refactoredAts[rev][fpath].extend(lnos)
                            except KeyError:
                                refactoredAts[rev][fpath] = lnos
    return refactoredAts

def getRefactedAtDeep_v2(combinedRefactoringfile:str) -> Dict[str, Dict[str,List[int]]]:
    """
    further check for the line numbers
    """
    refactoredAts = {}
    with open(combinedRefactoringfile, 'rb') as f:
        import pickle 
        refactoringsPRev = pickle.load(f)
    for rev, refactorings in refactoringsPRev.items():
        refactoredAts[rev] = {}
        rightRefactorings = refactorings.loc[refactorings.isLeft == 0]
        for fpath, df in rightRefactorings.groupby("filePath"):   
            refactoredAts[rev][fpath] = []
            for _, r in df.iterrows():
                start_lno = r.start[0]
                end_lno = r.end[0]
                lnos = list(range(start_lno, end_lno+1))
                try:
                    refactoredAts[rev][fpath].extend(lnos)
                except KeyError:
                    refactoredAts[rev][fpath] = lnos
    return refactoredAts

def checkRefactored(
    refactoredAts:Union[List, Dict], 
    rev:str, 
    fpath:str, lno:int, 
    deeper:bool
) -> bool:
    #
    if not deeper:
        return rev in refactoredAts
    else:
        if rev in refactoredAts.keys():
            refactorings = refactoredAts[rev]
            try:
                rs_at_fpath = refactorings[fpath]
            except KeyError:
                _basenames = [os.path.basename(_fpath) for _fpath in refactorings.keys()]
                _basename_fpath = os.path.basename(fpath)
                if _basename_fpath in _basenames:
                    idx = _basenames.index(_basename_fpath)
                    _lnos = list(refactorings.values())[idx]
                    return lno in _lnos                
                else:
                    return False
        else:
            return False 

def checkRefactored_v2(
    refactoredAts:Union[List, Dict], 
    rev:str, 
    fileAtRev:str, #fpath:str, 
    lno:int, 
    deeper:bool, #fileAtRev:str = None
) -> bool:
    if not deeper:
        return rev in refactoredAts
    else:
        if rev in refactoredAts.keys():
            refactorings = refactoredAts[rev]
            for rfpath, lnos in refactorings.items():
                lnos = set(lnos)
                #if fileAtRev is None:
                #    if os.path.basename(rfpath) == os.path.basename(fpath):
                #        if (lno is not None) and (lno in lnos):
                #            return True 
                #else:
                if rfpath == fileAtRev:
                    if (lno is not None) and (lno in lnos):
                        return True  
            return False 
        else:
            return False 

def get_activebugs(project):
    active_bugfile = f"{D4J_HOME}/framework/projects/{project}/active-bugs.csv"
    active_bugs = pd.read_csv(active_bugfile)
    active_bugs['revision.id.fixed'] = active_bugs['revision.id.fixed'].apply(lambda v:v[:8])
    rev_bids = {rev:bid for bid, rev in active_bugs[['bug.id', 'revision.id.fixed']].values}
    bid_revs = {bid:rev for rev, bid in rev_bids.items()}
    return rev_bids, bid_revs

def getRevealeds(inter_outputdir:str, rev:str) -> List:
    revealeds_files = glob.glob(os.path.join(inter_outputdir, f"{rev}/revealedAt.*"))
    revealeds = []
    for afile in revealeds_files:
        revealed_at_commit = os.path.basename(afile).split(".")[1]
        with open(afile) as f:
            revealedMuts = json.load(f)
        revealeds.append((revealed_at_commit, revealedMuts))
    return revealeds 

def getAppliedAt(inter_outputdir:str, mutatedAt:str, revealedAt:str) -> Dict:
    appliedAt_file = os.path.join(inter_outputdir, mutatedAt, f"appliedAt.{revealedAt}.json")
    with open(appliedAt_file) as f:
        appliedAt_mutInfos = json.load(f)
    return appliedAt_mutInfos 

def genRevealedInfo_v1(
    project, bid, reveled_muts_info, refactoredAts, mutatedAt
):
    parsedRevealedInfoAbug = {}
    for revealedAt, revealedInfo in reveled_muts_info:
        workdir_path = prepare_toBlameDir(
            D4J_HOME, 
            project,
            bid, 
            revealedAt, 
            "temp"
        )
        all_commits = git_utils.getAllCommits(workdir_path) # from HEAD to the end
        idx_mutatedAt = all_commits.index(mutatedAt)
        idx_revealedAt = all_commits.index(revealedAt)
        # get file-renamed
        _, _, renamedFiles = git_utils.getDiffFiles(repo_path, revealedAt, all_commits[idx_revealedAt + 1])

        for fpath, info_pmut in revealedInfo.items():
            #mutLR_info = mutLRs[fpath]['pairs']
            matchedFPath = getMatchedFile(workdir_path, revealedAt, fpath, renamedFiles = renamedFiles) # temp. file at the commit 
            for mutNo, info in info_pmut.items():
                mutNo = int(mutNo)
                mutStartPos, mutEndPos = info['mutLoc']
                # get the line number of the line where the muant was injected when revaling
                mutatedLnos = getLineNos(
                    mutStartPos, mutEndPos, 
                    workdir_path, matchedFPath)
                mutatedLnos = list(set(mutatedLnos)) # just in case
                modifiedAts = []
                for mutatedLno in mutatedLnos:
                    all_modifiedAts = git_utils.getModifiedAts(
                        workdir_path, 
                        matchedFPath,
                        mutatedLno, # start 
                        mutatedLno, # end
                        deeperCheck=True 
                    ) # all commits that modified this line 
                    for modifiedAt in all_modifiedAts:
                        idx = all_commits.index(modifiedAt)
                        modifiedAts.append([modifiedAt, idx, mutatedLno])
                #
                parsedRevealedInfoAbug[mutNo] = [(fpath, matchedFPath), []]
                #has_interesting = False
                for modifiedAt, idx_modifiedAt, mutatedLno in modifiedAts:
                    diff = idx_mutatedAt - idx_modifiedAt
                    if diff > 0: 
                        isRefactored = checkRefactored(
                            refactoredAts, modifiedAt, 
                            matchedFPath, mutatedLno, deeperCheck
                        )
                        parsedRevealedInfoAbug[mutNo][-1].append(
                            [   
                                #modifiedAt in refactoredAts,
                                isRefactored,
                                (idx_mutatedAt, idx_revealedAt, idx_modifiedAt), 
                                (mutatedAt, revealedAt, modifiedAt), 
                                mutatedLno
                            ]
                        )
    return parsedRevealedInfoAbug

def checkCommitWithSemanticChages(
    repo_path:str, rev:str, prev_rev:str, 
    inputfile:str, 
) -> bool:
    from utils.semantic_checker import compareFile
    kA = f"{TEMP_K}_{rev}-{os.path.basename(inputfile)[:-5]}"
    fileA = getAndSaveTempFile(repo_path, inputfile, rev, kA)
    #
    kB = f"{TEMP_K}_{prev_rev}-{os.path.basename(inputfile)[:-5]}"
    prev_inputfile = getMatchedFile(repo_path, prev_rev, inputfile, )
    fileB = getAndSaveTempFile(repo_path, prev_inputfile, prev_rev, kB)
    #print (fileA)
    #print (fileB)
    is_the_same_file = compareFile(fileA, fileB)
    # clean-up
    #os.remove(fileA)
    #os.remove(fileB)
    #print (is_the_same_file)
    #print (kB)
    #print ("HERERE")
    #sys.exit()
    return is_the_same_file 

def genRevealedInfo_detailed(
    project,
    outputdir, 
    mutLRs, mapped_to_uniq_mutK, 
    reveled_muts_info, refactoredAts, mutatedAt, repo_path
):
    parsedRevealedInfoAbug = {}
    all_parsedRevealedInfoAbug = {} # record all
    all_commits = git_utils.getAllCommits(
        repo_path, 
        branch = 'trunk' if any([v in repo_path for v in ['lang', 'math']]) else 'master'
    ) # from HEAD to the end
    idx_mutatedAt = git_utils.getCommitIdx(all_commits, mutatedAt)
    aft_mutatedAt = all_commits[idx_mutatedAt-1]
    for i, (revealedAt, revealedInfo) in enumerate(reveled_muts_info):
        #print (i, revealedAt)
        #if revealedAt != 'ed6faace':
        #    continue 
        appliedAtInfos = getAppliedAt(outputdir, mutatedAt, revealedAt) #key: fpath, values: Dict (key: new_fapth, value:str(mutNo))
        temp_store, processed = {}, set()
        #print ("tageat", revealedInfo.keys())
        for fpath, info_pmut in revealedInfo.items(): # fpath = original path at mutatedAt
            if fpath not in mutLRs.keys(): 
                continue
            _info_pmut = {}
            for mutNo, revealInfos in info_pmut.items():
                if int(mutNo) not in mutLRs[fpath].keys(): continue
                _info_pmut[mutNo] = revealInfos
            info_pmut = _info_pmut
            k = f"{TEMP_K}_{mutatedAt}-{os.path.basename(fpath)[:-5]}"
            temp_orgFpath = os.path.join(TEMP_WORKDIR, f"{k}.java")
            if not os.path.exists(temp_orgFpath):
                _fpath = getAndSaveTempFile(repo_path, fpath, mutatedAt, k)
                assert temp_orgFpath == _fpath, f"{temp_orgFpath} vs {_fpath}"

            try:
                mutLR_info = mutLRs[fpath]['pairs']
                which_mut = 'major'
            except KeyError:
                mutLR_info = mutLRs[fpath]
                which_mut = 'pit'
            #matchedFPath = getMatchedFile_v2(repo_path, revealedAt, fpath)
            matchedFPath_pmut = get_mut_files_at_reveal_from_appliedAts(
                fpath, info_pmut, appliedAtInfos[fpath], mapped_to_uniq_mutK)
            start_end_pos_pline_pmut = {}
            for mutNo, _matchedFile in matchedFPath_pmut.items():
                #print (mutNo, _matchedFile)
                if _matchedFile not in processed:
                    start_end_pos_pline_pmut[mutNo]  = getStartEndParseInfo(revealedAt, repo_path, _matchedFile)
                    temp_store[_matchedFile] = start_end_pos_pline_pmut[mutNo] 
                    processed.add(_matchedFile)
                    #print ('here', revealedAt,  _matchedFile)
                else:
                    start_end_pos_pline_pmut[mutNo] = temp_store[_matchedFile]
            #continue
            #start_end_pos_pline = getStartEndParseInfo(revealedAt, repo_path, matchedFPath)
            mutated_at_matchedFPath = fpath 
            #mutated_at_matchedFPath = getMatchedFile_v2(repo_path, mutatedAt, fpath) # temp. file at the commit 
            file_content = git_utils.show_file(mutatedAt, mutated_at_matchedFPath, repo_path)
            cnt, mutated_at_start_end_chars_cnt_dict = 0, {}
            for idx, lno_content in enumerate(file_content.split("\n")):
                mutated_at_start_end_chars_cnt_dict[idx + 1] = (cnt, cnt + len(lno_content) + 1)
                cnt += len(lno_content) + 1
            ####
            for mutNo, info in info_pmut.items():
                mutNo = int(mutNo)
                matchedFPath = matchedFPath_pmut[mutNo]
                start_end_pos_pline = start_end_pos_pline_pmut[mutNo]
                if which_mut == 'major':
                    org_mutatedLno = mutLR_info[mutNo]['pos']['end_lno']
                else:
                    _, _, end_chars_cnt = mutLR_info[mutNo]['pos']
                    org_mutatedLno = isInLno(mutated_at_start_end_chars_cnt_dict, end_chars_cnt) # get the line number at mutatedAt
                ##
                mutStartPos, mutEndPos = info['mutLoc']
                # get the line number of the line where the muant was injected when revaling
                muatedAtRevealLnos = getLineNos_byRepo(
                    start_end_pos_pline, 
                    mutStartPos, mutEndPos, 
                )
                assert len(muatedAtRevealLnos) >=1 , f"{fpath}, {mutNo}"
                parsedRevealedInfoAbug[(fpath, mutNo)] = []
                all_parsedRevealedInfoAbug[(fpath, mutNo)] = [] 
                ## new 
                isMutatedLnoChgd = checkWhetherLnoChgdAft(
                    CHGDIR, project, 
                    mutatedAt, revealedAt, 
                    org_mutatedLno, mutated_at_matchedFPath
                )
                if not isMutatedLnoChgd:
                    for muatedAtRevealLno in muatedAtRevealLnos:
                        all_parsedRevealedInfoAbug[(fpath, mutNo)].append(
                            [
                                (False, True), 
                                (mutatedAt, revealedAt, None), # None for modifiedAt  
                                (fpath, matchedFPath, None), # None for newFile as it was never changed
                                (org_mutatedLno, muatedAtRevealLno, None) # None for newLno as it was never changed
                            ]
                        )
                    continue 
                ## 
                for muatedAtRevealLno in muatedAtRevealLnos: ## here, what we need to do is to check whether this line was changed between mutatedAt and revealedAt (revealedAt -> should be included)
                    all_modifiedAts = git_utils.getModifiedAts_v2(
                        repo_path, revealedAt, 
                        matchedFPath, muatedAtRevealLno, muatedAtRevealLno, 
                        end_rev = aft_mutatedAt # a commit right after mutatedAt
                    ) # rev and end_rev are included. All modifications before the revelaedAt
                    ### below code will only target the cases where the mutant is revelaed B/C at the revealed comit
                    ### it was changed
                    # for this, if the line has not been modified at all, then ... yes, it will be emtpy
                    for modifiedAt, modifiedAtInfo in all_modifiedAts.items(): # check the commits (i.e., all_modfiedAtS) that modified the line muatedAtRevealLno
                        modifiedAt = modifiedAt[:8]
                        prevToModifiedAt = all_commits[all_commits.index(modifiedAt) + 1]
                        newChgs, _ = modifiedAtInfo
                        for newFile, newLnos in newChgs: # for line changes in each file
                            # check whether this commit is a valid one, i.e., not only changing comment -> whether the files are the same 
                            no_semantic_change = checkCommitWithSemanticChages(
                                repo_path, modifiedAt, prevToModifiedAt, 
                                newFile
                            )
                            ### new
                            if no_semantic_change: # meaning the same file (may with some whitespace or comment chanegs)#
                                # if no no_semantic_change, also means no refactoring
                                for newLno in newLnos:
                                    isRefactored = checkRefactored_v2(
                                        refactoredAts, 
                                        modifiedAt, 
                                        newFile,
                                        newLno, 
                                        deeperCheck
                                    )                          
                                    #all_info_to_save_pline.append(
                                    all_parsedRevealedInfoAbug[(fpath, mutNo)].append(
                                        [
                                            (isRefactored, True), 
                                            (mutatedAt, revealedAt, modifiedAt), 
                                            (fpath, matchedFPath, newFile),
                                            (org_mutatedLno, muatedAtRevealLno, newLno)
                                        ]
                                    )
                                continue # skip this one -> !!!!! 2dxxx shoudl be filetered!!!
                            ##
                            # check for semtantic
                            #fileAtModified = getMatchedFile(repo_path, modifiedAt, fpath)
                            k = f"{TEMP_K}_{modifiedAt}-{os.path.basename(newFile)[:-5]}"
                            temp_modFpath = os.path.join(TEMP_WORKDIR, f"{k}.java")
                            if not os.path.exists(temp_modFpath):
                                temp_modFpath = getAndSaveTempFile(repo_path, newFile, modifiedAt, k)
                            paired_line_infos = []
                            for newLno in newLnos: # for each line change
                                isRefactored = checkRefactored_v2(
                                    refactoredAts, 
                                    modifiedAt, 
                                    newFile,
                                    newLno, 
                                    deeperCheck
                                )
                                # check for semantic
                                ## simScore -> Leven
                                isTheSameLine, simScore = semantic_checker.compareLine(
                                    temp_orgFpath, org_mutatedLno, 
                                    temp_modFpath, newLno, computeSimScore = True
                                ) # check whether the are on the same line and the similarity value between them
                                paired_line_infos.append(
                                    [newLno, isRefactored, isTheSameLine, simScore]
                                )
                            ## keep the one with the highest sim score -> meaning, if there is a same line -> then isTheSameLine = True
                            simScores = [v[-1] for v in paired_line_infos]
                            idx_to_maxsim = simScores.index(max(simScores))
                            newLno, isRefactored, isTheSameLine, _ = paired_line_infos[idx_to_maxsim]
                            parsedRevealedInfoAbug[(fpath, mutNo)].append(
                                [   
                                    (isRefactored, isTheSameLine), 
                                    (mutatedAt, revealedAt, modifiedAt), 
                                    (fpath, matchedFPath, newFile),
                                    (org_mutatedLno, muatedAtRevealLno, newLno)
                                ]
                            ) 
                            ####
                            all_parsedRevealedInfoAbug[(fpath, mutNo)].append(
                                [   
                                    (isRefactored, isTheSameLine), 
                                    (mutatedAt, revealedAt, modifiedAt), 
                                    (fpath, matchedFPath, newFile),
                                    (org_mutatedLno, muatedAtRevealLno, newLno)
                                ]
                            )

                if len(all_parsedRevealedInfoAbug[(fpath, mutNo)]) == 0: # some case missed though no change
                    all_parsedRevealedInfoAbug[(fpath, mutNo)].append(
                        [   
                            (False, True), 
                            (mutatedAt, revealedAt, None), 
                            (fpath, matchedFPath, None),
                            (org_mutatedLno, muatedAtRevealLno, None)
                        ]
                    )  
    return parsedRevealedInfoAbug, all_parsedRevealedInfoAbug

def genRevealedInfo_weak(
    chgDir, project, bid, 
    reveled_muts_info, mutatedAt, repo_path
):
    parsedRevealedInfo = {}
    chgsFile = os.path.join(chgDir, f"{project}.{bid}.{mutatedAt}.chgdInfo.pkl")
    with open(chgsFile, 'rb') as f:
        chgInfo = pickle.load(f)
    #
    for revealedAt, revealedInfo in reveled_muts_info:
        for fpath, info_pmut in revealedInfo.items():
            k = f"{TEMP_K}_{mutatedAt}-{os.path.basename(fpath)[:-5]}"
            temp_orgFpath = os.path.join(TEMP_WORKDIR, f"{k}.java")
            if not os.path.exists(temp_orgFpath):
                _fpath = getAndSaveTempFile(repo_path, fpath, mutatedAt, k)
                assert temp_orgFpath == _fpath, f"{temp_orgFpath} vs {_fpath}"

            mutLR_info = mutLRs[fpath]['pairs']
            matchedFPath = getMatchedFile(repo_path, revealedAt, fpath) # temp. file at the commit 
            for mutNo, info in info_pmut.items():
                mutNo = int(mutNo)
                org_mutatedLno = mutLR_info[mutNo]['pos']['end_lno']
                modifiedAts = chgInfo[fpath][org_mutatedLno][0] # a list of commits and line numbers
                mutStartPos, mutEndPos = info['mutLoc']
                # get the line number of the line where the muant was injected when revaling
                muatedAtRevealLno = getLineNos_byRepo(
                    revealedAt, 
                    mutStartPos, mutEndPos, 
                    repo_path, 
                    matchedFPath
                )[0]
                parsedRevealedInfo[mutNo] = []
                for modifiedLno, modifiedAt in modifiedAts:
                    isRefactored = checkRefactored_v2(
                        refactoredAts, 
                        modifiedAt, 
                        matchedFPath, # b/c we didn't record the file-renamed ...
                        modifiedLno, 
                        deeperCheck
                    )
                    ##
                    # check for semtantic
                    fileAtModified = getMatchedFile(repo_path, modifiedAt, fpath)
                    temp_modFpath = getAndSaveTempFile(
                        repo_path, fileAtModified, modifiedAt, 
                        f"{TEMP_K}_{modifiedAt}-{os.path.basename(fileAtModified)[:-5]}"
                    )
                    isTheSameLine = semantic_checker.compareLine(
                        temp_orgFpath, org_mutatedLno, 
                        temp_modFpath, modifiedLno
                    )
                    ##
                    parsedRevealedInfo[mutNo].append(
                        [   
                            (isRefactored, isTheSameLine), 
                            (mutatedAt, revealedAt, modifiedAt), 
                            (fpath, matchedFPath, fileAtModified),
                            (org_mutatedLno, muatedAtRevealLno, modifiedLno)
                        ]
                    )
    return parsedRevealedInfo

def cleanUpTempFiles(repo_path):
    tempfiles = glob.glob(os.path.join(repo_path, f"{TEMP_K}.*.java"))
    for tempfile in tempfiles:
        if os.path.exists(tempfile):
            os.remove(tempfile)

if __name__ == "__main__":
    project = sys.argv[1]
    dest = sys.argv[2]
    outputdir = sys.argv[3]
    repo_path = sys.argv[4]
    remain = int(sys.argv[5])
    if bool(remain): 
        NEED_TO_FILTER_MUTOPS = []

    print (NEED_TO_FILTER_MUTOPS)
    os.makedirs(TEMP_WORKDIR, exist_ok=True)
    rev_bids, bid_revs = get_activebugs(project)
    deeperCheck = True 
    refactoring_dir = "output/refactor"
    #refactoring_dir = outputdir # actually ... now we don't need it as we have all ... # nope, not yet can be used
    #outputdir = f"output/new/final/{project}"
    root_outputdir = outputdir
    outputdir = f"{outputdir}/{project}"
    inter_outputdir = os.path.join(outputdir, "inter")
    revealeds_files_pbug = {} 
    for rev in os.listdir(inter_outputdir):
        if rev not in rev_bids.keys(): continue 
        # to check whether this is processed
        #_to_check_file = os.path.join(outputdir, f"{project}.{rev}.revealed.pkl") 
        #if os.path.exists(_to_check_file):
        ## -> because, in some cases, failed to run to the end due to some compilation errors, 
        ## nonetheless, the information of mutant propagation have been saved up to that point
        bid = rev_bids[rev] 
        revealeds = getRevealeds(inter_outputdir, rev)
        if len(revealeds) > 0:
            revealeds_files_pbug[bid] = revealeds
        else:
            print (f"nothing revealed: {rev}") # here, if nothing is revealed, then all_cand_XX will not contain  it 
    # 
    chgDir = f"output/temp/new/{project}"
    parsedRevealedInfo = {}
    all_parsedRevealedInfo = {}
    cnt = 0
    for i, (bid, reveled_muts_info) in enumerate(tqdm(list(revealeds_files_pbug.items()))):
        #if bid != 20: continue
        mutatedAt = bid_revs[bid]
        mutLR_file = os.path.join(inter_outputdir, f"{mutatedAt}/mutLRPair_pfile.pkl")
        mutLRs = pd.read_pickle(mutLR_file)
        # filter 
        _mutLRs = {}
        for fpath, muts in mutLRs.items():
            _mutLRs[fpath] = {}
            for mutNo, mutInfo in muts.items():
                if mutInfo['mutOp'][0] in NEED_TO_FILTER_MUTOPS: 
                    continue
                _mutLRs[fpath][mutNo] = mutInfo 
            if len(_mutLRs[fpath]) == 0:
                del _mutLRs[fpath]
        mutLRs = _mutLRs
        if deeperCheck:
            refactoredAts = getRefactedAtDeep(inter_outputdir, mutatedAt)
            combinedRefactoringfile = os.path.join(
                refactoring_dir, f"{project}/{project}.{mutatedAt}.refactorings.pkl"
            )
            refactoredAts = getRefactedAtDeep_v2(combinedRefactoringfile)
        else:
            refactoredAts = set(getRefactedAt(inter_outputdir, mutatedAt))
        # currently renaming has not been handled here
        ## get appliedAt info 
        ##
        _,_,mapped_to_uniq_mutK = analysis_utils.group_same_mutants(root_outputdir, project, mutatedAt)
        ##
        parsedRevealedInfo[bid], all_parsedRevealedInfo[bid] = genRevealedInfo_detailed(
            project, 
            inter_outputdir, # new 
            mutLRs, 
            mapped_to_uniq_mutK, 
            reveled_muts_info, refactoredAts, mutatedAt, repo_path)

    #sys.exit()
    # clean-up
    #print (parsedRevealedInfo)
    #cleanUpTempFiles(repo_path)
    #sys.exit()
    os.makedirs(os.path.join(dest, project), exist_ok=True)
    with open(os.path.join(dest, f'cands_{project}_{deeperCheck}.pkl'), 'wb') as f:
        pickle.dump(parsedRevealedInfo, f)

    with open(os.path.join(dest, f'all_{project}_{deeperCheck}.pkl'), 'wb') as f:
        pickle.dump(all_parsedRevealedInfo, f)
