"""
"""
import pandas as pd 
import os, sys 
from typing import List, Dict, Tuple, Set 
from .git_utils import get_repo, getCommitedDateTime

#MUT_TYPES = {
    #"sc_c":0, 
    #"sc_nc":1, 
    #"rc_c":2, 
    #"rc_nc":3, 
    #"nc_nc":4 
#}

def getBidRevDict(d4j_home:str, project:str, get_buggy:bool = False) -> Tuple[Dict[int,str], Dict[str,int]]:
    active_bugfile = os.path.join(d4j_home, f"framework/projects/{project}/active-bugs.csv")
    df = pd.read_csv(active_bugfile)
    if get_buggy:
        bid_fixedRev_pairs = df[['bug.id', 'revision.id.buggy']].values 
    else:
        bid_fixedRev_pairs = df[['bug.id', 'revision.id.fixed']].values 
    bidToRev, revToBid = {}, {}
    for bid, rev in bid_fixedRev_pairs:
        bidToRev[bid] = rev[:8] 
        revToBid[rev[:8]] = bid 
    return bidToRev, revToBid

def getToFoucsBugs(targetdir:str, project:str, onlySemChgd:bool = False, fulldir:str =  None) -> List[int]:
    # toFoucsBug:
    #   a fixed version with at least one "survived" mutant injected at the place that will go through a semantic change 
    #   => this is to focus on cases where we can find interesting and also compare wit those less interesing 
    if fulldir is None:
        toFoucsFile = os.path.join(targetdir, f"toFocus/{project}.csv" if not onlySemChgd else f"toFocus/wSemChgd/{project}.csv")
    else:
        toFoucsFile = os.path.join(fulldir, f"{project}.csv" if not onlySemChgd else f"wSemChgd/{project}.csv")
    with open(toFoucsFile) as f:
        bids = [int(l.strip()) for l in f.readlines() if bool(l.strip())]
    return bids 

def getCompletedMutInfos(outputdir:str, project:str, fixedVer:str, uniq_muts:bool = False) -> Dict[str, Dict]:
    if not uniq_muts:
        mutLRFile = os.path.join(outputdir, f"{project}/inter/{fixedVer}/mutLRPair_pfile.pkl")
    else:
        mutLRFile = os.path.join(outputdir, f"{project}/inter/{fixedVer}/uniq_mutLRPair_pfile.pkl")
    with open(mutLRFile, 'rb') as f:
        import pickle 
        mutCompInfos = pickle.load(f)
    rets = {}
    for fpath, mutInfos in mutCompInfos.items():
        for mutNo, mutInfo in mutInfos.items():
            rets[f"{fpath}-{mutNo}"] = mutInfo
    return rets

def getCoveredMutants(root_covgdir:str, project:str, bid:int) -> List[str]:
    def load(fpath:str):
        import gzip, pickle 
        with gzip.open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data 
    
    covg_dir = os.path.join(root_covgdir, f"{project}_{bid}")
    import glob 
    revealedAts = []
    for covg_file in glob.glob(os.path.join(covg_dir, "coverageAt.*.pkl")):
        covg_file_basename = os.path.basename(covg_file)
        revealedAt = covg_file_basename.split(".")[1]
        if "_" in revealedAt:
            revealedAt = revealedAt.split("_")[0]
        revealedAts.append(revealedAt)
    revealedAts = list(set(revealedAts))
    all_mutKs = []
    for revealedAt in revealedAts:
        covg_file = os.path.join(covg_dir, f"coverageAt.{revealedAt}.pkl") 
        if os.path.exists(covg_file):
            covg_data = load(covg_file)
        else:
            covg_data = {}

        mutKs = []
        for raw_mutK in covg_data.keys():
            mutated_fpath, mutNo = raw_mutK.split("-")
            mutated_fpath_K = mutated_fpath[:-5].replace("/", ".")
            mutKs.append(f"{mutated_fpath_K}-{mutNo}")
            
        for _covg_file in glob.glob(os.path.join(covg_dir, f"coverageAt.{revealedAt}_*.pkl")):
            _covg_file_basename = os.path.basename(_covg_file)
            mutK = _covg_file_basename.split("_")[1][:-4]   
            mutKs.append(mutK)
        mutKs = list(set(mutKs))
        all_mutKs.extend(mutKs)
    all_mutKs = list(set(all_mutKs))
    return all_mutKs

# deprecated -> drop this 
def getRevealStatus(root_outputdir:str, project:str, mutatedAt:str) -> int:
    """
    0: revealed mutants exist
    1: no mutants revealed
    2: failed to start refactoring, e.g., due to compilation error (related to project versoin issue)
    3: refactoring never tried 
    """
    revealedAtFile_pbid = os.path.join(root_outputdir, project, 
        f"{project}.{mutatedAt}.revealed.pkl")
    
    if not os.path.exists(revealedAtFile_pbid):
        return 3 
    else:
        with open(revealedAtFile_pbid, 'rb') as f:
            import pickle 
            data = pickle.load(f)
        if len(data) == 0: # refactoring failed from the start: e.g., compilation error 
            return 2 
        else:
            cnt_revealed = 0
            for fpath in data.keys():
                cnt_revealed += len(data[fpath])
            if cnt_revealed == 0:
                return 1 
            else:
                return 0 

def getMutantsRevealInfoFile(revealedInfodir:str, project:str) -> str:
    return os.path.join(revealedInfodir, f"all_{project}_True.pkl")

def getMutantsRevealInfos(revealedInfodir:str, project:str) -> Dict[int, Dict[int, List]]:
    # return (Dict): key = bug id, Dict: mutants info
    import pickle 
    mutantRevealInfoFile = getMutantsRevealInfoFile(revealedInfodir, project)
    with open(mutantRevealInfoFile, 'rb') as f:
        revealedInfo = pickle.load(f)
    return revealedInfo   

def computeMutLevel(isNotRefactoring:bool, isSemanticChange:bool, isChanged:bool) -> int:
    if isNotRefactoring and isSemanticChange: 
        assert isChanged, f"{isNotRefactoring} vs {isSemanticChange} vs {isChanged}"
        return 0 
    elif isChanged:
        if not isNotRefactoring and isSemanticChange:
            return 1
        elif isNotRefactoring and not isSemanticChange: # ... 
            return 2 
        else: # isNotRefactoring = False, isSemanticChange = False 
            return 3 
    else: # not changed at all
        return 4

def computeRevealedMutType_v1(isNotRefactoring:bool, isSemanticChange:bool, revealedAt:str, modifiedAts:List[str]) -> int:
    """
    -> these are on the "first" revealing points 
    Here, semantic change actually refers to semnatic change & not refactoriung 
    Five level: 
    - sc_c, sc_nc
    - rc_c, rc_nc
    - nc_nc
    """
    if isSemanticChange:
        if isNotRefactoring:
            #if revealedAt == modifiedAt:
            if revealedAt in modifiedAts:
                return "sc_c"
            else:
                return "sc_nc"
        else:
            #if revealedAt == modifiedAt:
            if revealedAt in modifiedAts:
                return "rc_c"
            else:
                return "rc_nc"
    else:
        #if revealedAt == modifiedAt:
        return "nc_nc" 

def computeRevealedMutType(isNotRefactorings:List[bool], isSemanticChanges:List[str], revealedAt:str, modifiedAts:List[str]) -> List:
    """
    -> these are on the "first" revealing points 
    Here, semantic change actually refers to semnatic change & not refactoriung 
    Five level: 
    - sc_c, sc_nc
    - rc_c, rc_nc
    - nc_nc
    """
    #
    rv_flag = 'c' if revealedAt in modifiedAts else 'nc'
    chg_status = []
    #print ("--", isNotRefactorings, isSemanticChanges)
    for isNotRefactoring, isSemanticChange in zip(isNotRefactorings, isSemanticChanges):
        #print ("======", isSemanticChange, isNotRefactoring)
        if isSemanticChange:
            if isNotRefactoring: # if we encounter any of this case, then 
                return f"sc_{rv_flag}", True, True, True, True 
            else:
                chg_status.append('rc')
        else:
            chg_status.append('nc')
    if 'rc' in  chg_status:
        return f'rc_{rv_flag}', False, False, True, True 
    else:
        #assert rv_flag == 'nc', f"{rv_flag}, {chg_status}"
        if rv_flag != 'nc': rv_flag = 'nc' # for non-semantic change 
        return f"nc_{rv_flag}", False, False, False, False 

    
def processInfos(info:List, only_chg:bool = False) -> List:
    if only_chg:
        (isRefactored, isTheSameLine), \
        (mutatedAt, modifiedAt), \
        (fpath, fileAtModified), \
        (org_mutatedLno, modifiedLno) = info
    else:
        (isRefactored, isTheSameLine), \
        (mutatedAt, revealedAt, modifiedAt), \
        (fpath, matchedFPath, fileAtModified), \
        (org_mutatedLno, muatedAtRevealLno, modifiedLno) = info
    isNotRefactoring = not isRefactored
    isSemanticChange = not isTheSameLine 
    isChanged = modifiedAt is not None 
    others = info[1:]
    return (isNotRefactoring, isSemanticChange, isChanged), others 


def categoriseMutants(revealedInfoPbug:Dict[int, List]) -> Dict[int, Dict[int, List]]:
    """
    Five levels: 
        is_not_refactoring
        is_semantic_change 
        is_changed 
        All true, => interesting (LV.0)
        is semantic change, but refacoring (LV.1)
        is not semanitc chanege but refactoring (LV.2)
        is not semantic change and not refactoring (LV.3)
        is not changed (LV.4)

        # 5 = not revealed
    """
    categorised = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
    for (fpath, mutNo), mutInfoLst in revealedInfoPbug.items():
        mutLevels = []
        for mutInfo in mutInfoLst: # include per-change info
            (isNotRefactoring, isSemanticChange, isChanged), _ = processInfos(mutInfo)
            mutLevel = computeMutLevel(isNotRefactoring, isSemanticChange, isChanged)
            mutLevels.append(mutLevel)
        #if len(mutLevels) == 0:
        #    print (fpath, mutNo, mutInfoLst)
        final_mutLevel = min(mutLevels)
        categorised[final_mutLevel][(fpath, mutNo)] = mutInfoLst 
    return categorised

def getMutLevelCategoInfo(categoriseds:Dict) -> pd.DataFrame:
    # isRefactored, isTheSameLine, isChanged, 
    retInfos = {
        'bid':[], 'level':[], 'mutK':[],
        'isNotRefAndSemChg':[], 'isNotRefactoring':[], 'isSemanticChange':[], 'isChanged':[],
        'nDiverged':[], 
        'details':[]
    }
    for bid, infoPBug in categoriseds.items():
        for level in [0, 1, 2, 3, 4]:
            for (fpath, mutNo), infos in infoPBug[level].items():
                retInfos['bid'].append(bid)
                retInfos['level'].append(level)
                retInfos['mutK'].append(f"{fpath}-{mutNo}")
                retInfos['nDiverged'].append(len(infos))
                cntNotRefAndSemantic, cntNotRefactored, cntSemanticChange, cntChanged = 0, 0, 0, 0 
                details = []
                for info in infos:
                    (isNotRefactoring, isSemanticChange, isChanged), others = processInfos(info)
                    cntNotRefAndSemantic += int(isNotRefactoring & isSemanticChange)
                    cntNotRefactored += int(isNotRefactoring)
                    cntSemanticChange += int(isSemanticChange)
                    cntChanged += int(isChanged)
                    details.append(others)
                    #
                retInfos['isNotRefAndSemChg'].append(cntNotRefAndSemantic)
                retInfos['isNotRefactoring'].append(cntNotRefactored)
                retInfos['isSemanticChange'].append(cntSemanticChange)
                retInfos['isChanged'].append(cntChanged)
                retInfos['details'].append(details)
    return pd.DataFrame(retInfos)

## will be computed per level ()
def getCommitIndex(allCommits:List[str], target_c:str):
    try:
        return allCommits.index(target_c)
    except ValueError:
        for i,c in enumerate(allCommits):
            if c.startswith(target_c) or target_c.startswith(c):
                return i 
        return None
        
def computeTechnicalDebt(
    #revealInfoOfMuts:pd.DataFrame,
    revealInfoOfMuts:Dict,
    allCommits:List[str], 
    root_repoPath:str, 
) -> pd.DataFrame:
    """
    For each mutant, will compute
        1) the number of commits between mutatedAt and revealedAt  
        2) the time between mutatedAt and revealedAt 
        3) the number of changed commits between mutatedAt and revealedAt 
            3-1) the number of total changes
            3-2) the number of interesting changes # i.e., semantic changes that are not refactoring
            3-4) the number of semantic changes # can include refactorings (acutally... syntax changes)
    """
            
    #assert revealInfoOfMuts.level.unique().shape[0] == 1, revealInfoOfMuts.level.unique()
    repo = get_repo(root_repoPath)
    rets = {'mutK':[], 
            'nbtcs':[], 'dbtcs':[], 
            'nDiverged':[],
            'isNotRefAndSemChg':[], 
            'isSemanticChange':[], 
            'isChanged':[]}
    #for _, row in revealInfoOfMuts.iterrows(): # per mut (mutKey: fpath-mutNo)
    for mutKey, mutInfoLst in revealInfoOfMuts.items():
        n_btwn_commits_lst = []
        days_btwn_commits_lst = []
        chg_cnts, mut_dvgd_ks = {}, []#'isNotRefAndSemChg':0, 'isSemanticChange':0, 'isChanged':0}
        for info in mutInfoLst: # for each diverged mutant case
            (isNotRefactoring, isSemanticChange, isChanged), others = processInfos(info)
            mutatedAt, revealedAt, modifiedAt = others[0]
            #idxToMutatedAt = allCommits.index(mutatedAt)
            idxToMutatedAt = getCommitIndex(allCommits, mutatedAt)
            #idxToRevealedAt = allCommits.index(revealedAt)
            idxToRevealedAt = getCommitIndex(allCommits, revealedAt)
            #
            n_btwn_commits = idxToMutatedAt - idxToRevealedAt
            #assert n_btwn_commits >= 0, f"{row.bid} {row.mut}: {mutatedAt} vs {revealedAt}"
            assert n_btwn_commits >= 0, f"{mutKey}: {mutatedAt} vs {revealedAt} ({modifiedAt})"
            n_btwn_commits_lst.append(n_btwn_commits)
            #
            datetime_mutatedAt = getCommitedDateTime(repo, mutatedAt)
            datetime_revealedAt = getCommitedDateTime(repo, revealedAt)
            sec_btwn_commits = (datetime_revealedAt - datetime_mutatedAt).total_seconds()/(60*60*24)
            assert sec_btwn_commits >= 0, f"{mutKey}: {mutatedAt} vs {revealedAt} ({modifiedAt})"
            days_btwn_commits_lst.append(sec_btwn_commits)
            #
            _, matchedFPath, _ =  others[1]
            _, muatedAtRevealLno, _ = others[2]
            mut_dvgd_k = (matchedFPath, muatedAtRevealLno)
            if mut_dvgd_k not in chg_cnts.keys():
                chg_cnts[mut_dvgd_k] = {'isNotRefAndSemChg':0, 'isSemanticChange':0, 'isChanged':0}
            chg_cnts[mut_dvgd_k]['isNotRefAndSemChg'] += int(isNotRefactoring & isSemanticChange)
            chg_cnts[mut_dvgd_k]['isSemanticChange'] += int(isSemanticChange)
            chg_cnts[mut_dvgd_k]['isChanged'] += int(isChanged)
            mut_dvgd_ks.append(mut_dvgd_k)
            #
        max_nbtcs = max(n_btwn_commits_lst)
        idx_to_max = n_btwn_commits_lst.index(max_nbtcs)
        #sum_nbtcs = sum(n_btwn_commits_lst) -> for this, need to group mutants per diverged case (matchedFpath, mutatedAtRevealLno)
        max_dbtcs = days_btwn_commits_lst[idx_to_max]
        #sum_dbtcs = sum(days_btwn_commits_lst)
        # between mutatedAt and revelatedAt, ...
        cntNotRefAndSemantic_at_max = chg_cnts[mut_dvgd_ks[idx_to_max]]['isNotRefAndSemChg']
        cntIsSemanticChange_at_max = chg_cnts[mut_dvgd_ks[idx_to_max]]['isSemanticChange']
        cntIsChanged_at_max = chg_cnts[mut_dvgd_ks[idx_to_max]]['isChanged']
        
        # update
        rets['mutK'].append(mutKey)
        rets['nbtcs'].append(max_nbtcs)
        rets['dbtcs'].append(max_dbtcs)
        rets['nDiverged'].append(len(set(mut_dvgd_ks)))
        rets['isNotRefAndSemChg'].append(cntNotRefAndSemantic_at_max)
        rets['isSemanticChange'].append(cntIsSemanticChange_at_max)
        rets['isChanged'].append(cntIsChanged_at_max)
    return pd.DataFrame(rets)

def get_init_indv_mut_status(
    root_outputdir:str, 
    project:str, 
    bid_rev_pairs:List[Tuple[int, str]] = None, 
    MUTOPS:List[str] = None,
) -> pd.DataFrame:
    """
    return the initial number of mutants 
        - injected, covered, killed, time-out (killed), and survived 
    """
    import xml.etree.ElementTree as ET 
    import glob 
    from tqdm import tqdm 

    if bid_rev_pairs is not None:
        bid_rev_pairs = {rev:bid for bid, rev in bid_rev_pairs}
    root_outputdir = os.path.join(root_outputdir, project + "/inter")
    mutfiles = glob.glob(os.path.join(root_outputdir, "**/mutations.xml"))
    initStatusPBug = []
    for mutfile in tqdm(mutfiles):
        rev = os.path.dirname(mutfile).split("/")[-1]
        if bid_rev_pairs is not None: 
            try:
                bid = bid_rev_pairs[rev]
            except Exception:
                assert False
            k = f"{bid}-{rev}"
        else:
            k = rev 
        rev = os.path.dirname(mutfile).split("/")[-1]
        tree = ET.parse(mutfile)
        all_mutants = tree.findall("mutation")
        #
        if MUTOPS is not None:
            _all_mutants = []
            for m in all_mutants:
                mutOp = m.find('mutator').text
                if mutOp in MUTOPS:
                    _all_mutants.append(m)
            all_mutants = _all_mutants

        for mut in all_mutants:
            properties = get_checking_properties(mut)
            mutOp = mut.find('mutator').text.split(".")[-1]
            properties = [bid, k, mutOp] + properties
            initStatusPBug.append(properties)
    return pd.DataFrame(initStatusPBug, 
        columns = ['bid', 'k', 'mutOp', 'status', 'sourceFile', 'mutatedClass', 'lineNumber', 'index', 'block', 'description'])

def getInitMutStatus(
    root_outputdir:str, 
    project:str, 
    bid_rev_pairs:List[Tuple[int, str]] = None, 
    logging:bool = True, 
    MUTOPS:List[str] = None,
) -> pd.DataFrame:
    """
    return the initial number of mutants 
        - injected, covered, killed, time-out (killed), and survived 
    """
    import xml.etree.ElementTree as ET 
    import glob 

    if bid_rev_pairs is not None:
        bid_rev_pairs = {rev:bid for bid, rev in bid_rev_pairs}

    root_outputdir = os.path.join(root_outputdir, project + "/inter")
    mutfiles = glob.glob(os.path.join(root_outputdir, "**/mutations.xml"))
    initStatusPBug = {'bid':[], 'k':[], 'all':[], 'killed':[], 'timed_out':[], 'no_covg':[], 
        'non_viable':[], 'memory_error':[], 'run_error':[], 
        'survived':[]}
    
    def get_muts(muts, status):
        rets = []
        for m in muts:
            if m.attrib['status'] == status:
                rets.append(m)
        return rets 
    
    for mutfile in mutfiles:
        rev = os.path.dirname(mutfile).split("/")[-1]
        tree = ET.parse(mutfile)
        all_mutants = tree.findall("mutation")
        #
        if MUTOPS is not None:
            _all_mutants = []
            for m in all_mutants:
                mutOp = m.find('mutator').text
                if mutOp in MUTOPS:
                    _all_mutants.append(m)
            all_mutants = _all_mutants
        #            
        n_all = len(all_mutants)
        #killed_muts = tree.findall("mutation/[@status = 'KILLED']")
        killed_muts = get_muts(all_mutants, 'KILLED')
        n_killed = len(killed_muts)
        #no_covg_muts = tree.findall("mutation/[@status = 'NO_COVERAGE']")
        no_covg_muts = get_muts(all_mutants, 'NO_COVERAGE')
        n_no_covg = len(no_covg_muts)
        #survived_muts = tree.findall("mutation/[@status = 'SURVIVED']")
        survived_muts = get_muts(all_mutants, 'SURVIVED')
        n_survived = len(survived_muts)
        #timed_out_muts = tree.findall("mutation/[@status = 'TIMED_OUT']")
        timed_out_muts = get_muts(all_mutants, 'TIMED_OUT')
        n_timed_out = len(timed_out_muts)
        #
        #non_viable_muts = tree.findall("mutation/[@status = 'NON_VIABLE']")
        non_viable_muts = get_muts(all_mutants, 'NON_VIABLE')
        n_non_viable = len(non_viable_muts)
        #memory_error_muts = tree.findall("mutation/[@status = 'MEMORY_ERROR']")
        memory_error_muts = get_muts(all_mutants, 'MEMORY_ERROR')
        n_memory_error = len(memory_error_muts)
        #run_error_muts = tree.findall("mutation/[@status = 'RUN_ERROR']")
        run_error_muts = get_muts(all_mutants, 'RUN_ERROR')
        n_run_error = len(run_error_muts)

        if bid_rev_pairs is not None: 
            try:
                bid = bid_rev_pairs[rev]
            except Exception:
                print (mutfile)
                assert False
            k = f"{bid}-{rev}"
        else:
            k = rev 
        
        initStatusPBug['bid'].append(bid)
        initStatusPBug['k'].append(k)
        initStatusPBug['all'].append(n_all)
        initStatusPBug['killed'].append(n_killed)
        initStatusPBug['no_covg'].append(n_no_covg)
        initStatusPBug['survived'].append(n_survived)
        initStatusPBug['timed_out'].append(n_timed_out)
        initStatusPBug['non_viable'].append(n_non_viable)
        initStatusPBug['memory_error'].append(n_memory_error)
        initStatusPBug['run_error'].append(n_run_error)
        if logging:
            print (f"For {k} (with {n_all} mutants),\n\tKilled: {n_killed}\n\tNo Coverage: {n_no_covg}\n\tSurvived: {n_survived}\n\tTimed out: {n_timed_out}")
    return pd.DataFrame(initStatusPBug)

def get_checking_properties(e) -> List[str]:
    properties = []
    needToCheck_lst = ['status', 'sourceFile', 'mutatedClass', 'lineNumber', 'index', 'block', 'description']
    for needToCheck in needToCheck_lst:
        if needToCheck == 'status':
            v = e.attrib[needToCheck]
        else:
            v = e.find(needToCheck).text
        properties.append(v)
    return properties
#def check_the_same_muts(e1, e2) -> bool:
    #"""
    #compare sourceFile, line
    #"""
    #needToCheck_lst = ['status', 'sourceFile', 'lineNumber', 'index', 'block', 'description']
    #for needToCheck in needToCheck_lst:
        #if needToCheck == 'status':
            #v1 = e1.attrib[needToCheck]
            #v2 = e2.attrib[needToCheck]
        #else:
            #v1 = e1.find(needToCheck).text
            #v2 = e2.find(needToCheck).text
        #if v1 != v2:
            #return False
    #return True 

def getInitMutStatus_of_unique(
    root_outputdir:str, 
    project:str, 
    bid_rev_pairs:List[Tuple[int, str]] = None, 
    logging:bool = True,
    MUTOPS:List[str] = None,
) -> pd.DataFrame:
    import xml.etree.ElementTree as ET 
    import glob 
    from tqdm import tqdm 
    
    if bid_rev_pairs is not None:
        bid_rev_pairs = {rev:bid for bid, rev in bid_rev_pairs}
    root_outputdir = os.path.join(root_outputdir, project + "/inter")
    mutfiles = glob.glob(os.path.join(root_outputdir, "**/mutations.xml"))
    initStatusPBug = {
        'bid':[], 'k':[], 
        'u_all':[], 'u_killed':[], 'u_timed_out':[], 'u_no_covg':[], 
        'u_non_viable':[], 'u_memory_error':[], 'u_run_error':[], 
        'u_survived':[]}
    #
    for mutfile in tqdm(mutfiles):
        rev = os.path.dirname(mutfile).split("/")[-1]
        tree = ET.parse(mutfile)
        muts = tree.findall('mutation')
        #
        if MUTOPS is not None:
            _muts = []
            for m in muts:
                mutOp = m.find('mutator').text
                if mutOp in MUTOPS:
                    _muts.append(m)
            muts = _muts        
        #
        inspected = []
        inspected_properties = set()
        for mut in muts:
            props_of_mut = "-".join(get_checking_properties(mut))
            if len(inspected) == 0: 
                inspected.append(mut)
                inspected_properties.add(props_of_mut)
            else:
                if props_of_mut in inspected_properties: continue
                inspected.append(mut)
                inspected_properties.add(props_of_mut)
        #
        n_all = len(inspected)
        n_killed, n_no_covg, n_survived, n_timed_out, n_non_viable, n_memory_error, n_run_error = 0, 0, 0, 0, 0, 0, 0 
        for mut in inspected:
            status = mut.attrib['status']
            if status == 'KILLED':
                n_killed += 1
            elif status == 'NO_COVERAGE':
                n_no_covg += 1
            elif status == 'SURVIVED':
                n_survived += 1
            elif status == 'TIMED_OUT':
                n_timed_out += 1
            elif status == 'NON_VIABLE':
                n_non_viable += 1 
            elif status == "MEMORY_ERROR":
                n_memory_error += 1
            else:
                assert status == 'RUN_ERROR', f"{status} in {mutfile}"
                n_run_error += 1 
        #
        if bid_rev_pairs is not None: 
            try:
                bid = bid_rev_pairs[rev]
            except Exception:
                print (mutfile)
                assert False
            k = f"{bid}-{rev}"
        else:
            k = rev 
        # save 
        initStatusPBug['bid'].append(bid)
        initStatusPBug['k'].append(k)
        initStatusPBug['u_all'].append(n_all)
        initStatusPBug['u_killed'].append(n_killed)
        initStatusPBug['u_no_covg'].append(n_no_covg)
        initStatusPBug['u_survived'].append(n_survived)
        initStatusPBug['u_timed_out'].append(n_timed_out)
        initStatusPBug['u_non_viable'].append(n_non_viable)
        initStatusPBug['u_memory_error'].append(n_memory_error)
        initStatusPBug['u_run_error'].append(n_run_error)

        if logging:
            print (f"For {k} (with {n_all} mutants),\n\tKilled: {n_killed}\n\tNo Coverage: {n_no_covg}\n\tSurvived: {n_survived}\n\tTimed out: {n_timed_out}")
    return pd.DataFrame(initStatusPBug)

def gen_init_mutator_freq(only_unique:bool = False) -> Dict[str, List]:
    init_mutator_freq = {'k':[], 'bid':[]}
    mutOps = [
        'MathMutator',
        'ConditionalsBoundaryMutator',
        'IncrementsMutator', 
        'InvertNegsMutator', 
        'NegateConditionalsMutator',
        'VoidMethodCallMutator', 
        'PrimitiveReturnsMutator', 
        'EmptyObjectReturnValsMutator', 
        'BooleanFalseReturnValsMutator', 
        'BooleanTrueReturnValsMutator', 
        'NullReturnValsMutator', 
        'InlineConstantMutator',
        'ConstructorCallMutator', 
        'NonVoidMethodCallMutator', 
        'RemoveIncrementsMutator', 
        'AOD1Mutator', 'AOD2Mutator', 
        'OBBN1Mutator', 'OBBN2Mutator', 'OBBN3Mutator',
        'CRCR1Mutator', 'CRCR2Mutator', 'CRCR3Mutator', 'CRCR4Mutator', 'CRCR5Mutator', 'CRCR6Mutator',
        'ROR1Mutator', 'ROR2Mutator', 'ROR3Mutator', 'ROR4Mutator', 'ROR5Mutator',
        'AOR1Mutator', 'AOR2Mutator', 'AOR3Mutator', 'AOR4Mutator'
    ]
    if only_unique:
        mutOps = [f'u_{mutOp}' for mutOp in mutOps]
    for mutOp in mutOps:
        init_mutator_freq[mutOp] = []
    return init_mutator_freq

def get_mutator_freq(
    root_outputdir:str, 
    project:str, 
    bid_rev_pairs:List[Tuple[int, str]] = None, 
    only_unique:bool = False, 
) -> pd.DataFrame:
    import xml.etree.ElementTree as ET 
    import glob 
    from tqdm import tqdm 
    
    if bid_rev_pairs is not None:
        bid_rev_pairs = {rev:bid for bid, rev in bid_rev_pairs}
    root_outputdir = os.path.join(root_outputdir, project + "/inter")
    mutfiles = glob.glob(os.path.join(root_outputdir, "**/mutations.xml"))
    init_mutator_freq = gen_init_mutator_freq(only_unique)
    #print (only_unique, init_mutator_freq.keys())
    #
    for mutfile in tqdm(mutfiles):
        rev = os.path.dirname(mutfile).split("/")[-1]
        tree = ET.parse(mutfile)
        muts = tree.findall('mutation')
        if only_unique:
            inspected = []
            inspected_properties = set()
            for mut in muts:
                props_of_mut = "-".join(get_checking_properties(mut))
                if len(inspected) == 0: 
                    inspected.append(mut)
                    inspected_properties.add(props_of_mut)
                else:
                    if props_of_mut in inspected_properties: continue
                    inspected.append(mut)
                    inspected_properties.add(props_of_mut)
        else:
            inspected = muts 
        
        # init: add conunt 0 for each 
        for data_k in init_mutator_freq.keys():
            if data_k in ['bid', 'k']: continue
            init_mutator_freq[data_k].append(0)
        # save
        for mut in inspected:
            mutOp = mut.find('mutator').text
            mutOp = mutOp.split(".")[-1]
            if only_unique:
                mutOp = f"u_{mutOp}"
            init_mutator_freq[mutOp][-1] += 1 

        if bid_rev_pairs is not None: 
            try:
                bid = bid_rev_pairs[rev]
            except Exception:
                print (mutfile)
                assert False
            k = f"{bid}-{rev}"
        else:
            k = rev 
        init_mutator_freq['bid'].append(bid)
        init_mutator_freq['k'].append(k)
    return pd.DataFrame(init_mutator_freq)


def getToSaveFiles(dest:str, project:str, commit_hash:str) -> Tuple[str]: 
    final_output_key = f"{project}.{commit_hash}"
    revealed_mut_file = os.path.join(dest, f"{final_output_key}.revealed.pkl")
    surv_mut_file = os.path.join(dest, f"{final_output_key}.survived.pkl")
    mut_deadat_file = os.path.join(dest, f"{final_output_key}.mutDeadAt.pkl")
    #mut_deadat_file = os.path.join(dest, f"deadmuts/{final_output_key}.mutDeadAt.pkl")
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


def get_mut_refactor_status(outputdir:str, project:str, 
    fixedRev:str, 
    needToFilterOps:List[str] = None, 
    needToFocusOps:List[str] = None, 
    formatted_invalid_mutKs:Set[str] = None, 
    regen_uniq:bool = False) -> Dict:
    # check wh
    (revealed_mut_file, surv_mut_file, 
        mut_deadat_file, refactoring_file) = getToSaveFiles(
            os.path.join(outputdir, "processed", project), project, fixedRev)
    all_processed = os.path.exists(revealed_mut_file) and (
        os.path.exists(surv_mut_file)) and (
        os.path.exists(mut_deadat_file)) and (
        os.path.exists(refactoring_file))
    
    def mapToUniqueMuts(targets, mapped_to_uniq_mutK, filtered):
        return list(set([mapped_to_uniq_mutK[target] for target in targets if target not in filtered]))

    if all_processed: # meaning reach to the end
        #import pickle 
        #mutLR_file = os.path.join(outputdir, project, "inter", fixedRev, 'mutLRPair_pfile.pkl')
        # get the list of targeted mutants 
        #with open(mutLR_file, 'rb') as f:
        #    muts = pickle.load(f) 
        # this will automatically filter out those 
        filtered, grouped, mapped_to_uniq_mutK = group_same_mutants(outputdir, project, fixedRev, 
            needToFilterOps = needToFilterOps, 
            needToFocusOps = needToFocusOps, 
            regen_uniq=regen_uniq) 

        # get dead mutants (currently, this one also include the revealed)
        deadMuts = list(map(format_mutK, getDeadMutKs(mut_deadat_file)))
        deadMuts = mapToUniqueMuts(deadMuts, mapped_to_uniq_mutK, filtered)
        deadMuts = list(set([mutK for mutK in deadMuts if mutK not in formatted_invalid_mutKs]))
        # get surived mutants 
        survivedMuts = list(map(format_mutK, getSurvMuts(surv_mut_file)))
        survivedMuts = mapToUniqueMuts(survivedMuts, mapped_to_uniq_mutK, filtered)
        survivedMuts = list(set([mutK for mutK in survivedMuts if mutK not in formatted_invalid_mutKs]))
        # get revealed mutants 
        #revealdMuts = list(map(format_mutK, getRevealedMutKs(revealed_mut_file)))
        ## need to filter out those that are not in the latest list ->  temporary 
        import glob 
        revealed_files = glob.glob(os.path.join(outputdir, project, "inter", fixedRev, "revealedAt.*.json"))
        revealdMuts = []
        #print ('revealed files', revealed_files)
        for revealed_file in revealed_files:
            import json 
            with open(revealed_file) as f:
                data = json.load(f)
            revealedAt = os.path.basename(revealed_file).split(".")[1]
            #
            appliedAt_file = os.path.join(outputdir, project, "inter", fixedRev, f"appliedAt.{revealedAt}.json")
            with open(appliedAt_file) as f:
                appliedAt_data = json.load(f)
            #
            for fpath in data:
                try:
                    appliedAt_data_pfile = appliedAt_data[fpath]
                except KeyError: 
                    continue 
                #revealdMuts.extend([f"{fpath}-{mutNo}" for mutNo in data[fpath].keys()])
                for mutNo in data[fpath].keys():
                    try:
                        _ = appliedAt_data_pfile[mutNo]
                    except KeyError:
                        continue 
                    revealdMuts.append(f"{fpath}-{mutNo}")
                    #print('revealed', f"{fpath}-{mutNo}")
        revealdMuts = list(set(revealdMuts))             
        revealdMuts = list(map(format_mutK, revealdMuts))
        #for rvm in revealdMuts:
        #    if rvm not in filtered:
        #        try:
        #            _ = mapped_to_uniq_mutK[rvm]
        #        except KeyError:
        #            print (rvm)
        #print (mapped_to_uniq_mutK)
        #print(revealdMuts)
        #print ([v.split('-')[1] for v in filtered])
        revealdMuts = mapToUniqueMuts(revealdMuts, mapped_to_uniq_mutK, filtered)
        revealdMuts = list(set([mutK for mutK in revealdMuts if mutK not in formatted_invalid_mutKs]))
        # 
        pure_deadMuts = list(set(deadMuts) - set(revealdMuts))
        ### filter those remained due to using redundant mutants for the first trial 
        survivedMuts = list(set(survivedMuts) - set(revealdMuts))
        # 
        survivedMuts = list(set(survivedMuts) - set(deadMuts))
        ###
        # take union for checking 
        processed_muts = set(pure_deadMuts + survivedMuts + revealdMuts)
        muts_belong_nowhere = list(set(grouped.keys()) - processed_muts)
        muts_belong_nowhere = [mutK for mutK in muts_belong_nowhere if mutK not in formatted_invalid_mutKs]
        assert len(muts_belong_nowhere) == 0, f"{project}, {fixedRev}, {str(muts_belong_nowhere)}"
        #return {'all':n_pure_dead + n_revealed + n_survived, 'surv':n_survived, 'reveal':n_revealed, 'dead':n_pure_dead}
        return {'surv':survivedMuts, 'reveal':revealdMuts, 'dead':pure_deadMuts, 'nowhere':muts_belong_nowhere}
    else: # this m
        return None


def addBaseName(infoDf:pd.DataFrame):#, byIndex:bool = False):
    if len(infoDf) == 0:
        infoDf['fbasename'] = None 
    else:
        if isinstance(infoDf.mutK.values[0], Tuple):
            infoDf['fbasename'] = infoDf.mutK.apply(lambda v:os.path.basename(v[0])).values 
        elif isinstance(infoDf.mutK.values[0], str):
            infoDf['fbasename'] = infoDf.mutK.apply(lambda v:os.path.basename(v.split("-")[0])).values 
        else:
            print ("Something is wrong...", infoDf.mutK.values[0])
            assert False  
###
#def getMutsDead(root_outputdir:str, project:str):
    #pass 
#def getMutsSurvived():
    #pass 
def getAllTargetedMutants(
    root_outputdir:str, project:str, revToBid:Dict[str, int]
) -> Dict[int, pd.DataFrame]:
    import glob 
    import pickle 
    inter_outputdir = os.path.join(root_outputdir, project + "/inter")
    files = glob.glob(os.path.join(inter_outputdir, "*/mutLRPair_pfile.pkl"))
    #ret_mutants = {'bid':[], 'mutK':[], 'mutatedAt':[], 'mutOp':[]} 
    ret_mutants = {}
    for file in files:
        mutatedAt = os.path.dirname(file).split("/")[-1] # mutated at 
        bid = revToBid[mutatedAt]
        ret_mutants[bid] = {'mutK':[], 'mutOp':[]}
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for fpath, mutInfos in data.items():
            for mutNo, mutInfo in mutInfos.items():
                mutK = f"{fpath}-{mutNo}" 
                mutOp = mutInfo['mutOp'][0]
                ret_mutants[bid]['mutK'].append(mutK)
                ret_mutants[bid]['mutOp'].append(mutOp)
        ret_mutants[bid] = pd.DataFrame(ret_mutants[bid])
        ret_mutants[bid] = ret_mutants[bid].set_index('mutK')
    return ret_mutants

def getDeadMutKs(mutFile:str) -> List[Tuple]:
    import pickle 
    with open(mutFile, 'rb') as f:
        muts = pickle.load(f)
    deadMuts = []
    for (fpath,mutNo), deadAt in muts.items():
        if deadAt is not None:
            deadMuts.append(f"{fpath}-{mutNo}")
    #deatMuts = [f"{fpath}-{mutNo}" for fpath,mutNo in muts.keys()]
    #for (fpath, mutNo), deadAt in muts.items():
    #    pass 
    return deadMuts

def getSurvMuts(mutFile:str) -> List[Tuple]:
    import pickle 
    with open(mutFile, 'rb') as f:
        muts = pickle.load(f)
    survMuts = []
    for fpath, mutNos in muts.items():
        for mutNo in mutNos:
            survMuts.append(f"{fpath}-{mutNo}") 
    return survMuts

def getRevealedMutKs(mutFile:str) -> List[Tuple]:
    import pickle 
    with open(mutFile, 'rb') as f:
        muts = pickle.load(f)
    revealdMuts = []
    for fpath, mutInfos in muts.items(): 
        for mutNo in mutInfos.keys():
            revealdMuts.append(f"{fpath}-{mutNo}")
    return revealdMuts

def addMutantFinalStatusInfo(
    root_outputdir:str, 
    project:str, 
    ret_mutants:Dict[int, pd.DataFrame], 
    bidToRev:Dict[int, str]
) -> Dict[int, pd.DataFrame]:
    """
    1 = revealed
    0 = survived
    -1 = dead
    """
    outputdir = os.path.join(root_outputdir, project)
    for bid in ret_mutants.keys():
        mutFinalStatus = [0] * len(ret_mutants[bid])
        ret_mutants[bid]['finalStatus'] = mutFinalStatus
        mutatedAt = bidToRev[bid]

        deadMutsFile = os.path.join(outputdir, f"{project}.{mutatedAt}.mutDeadAt.pkl")
        if not os.path.exists(deadMutsFile): # from somre reason, failed... 
            ret_mutants[bid]['finalStatus'] = None
            continue 
        deadMutKs = getDeadMutKs(deadMutsFile) # a list of mutKeys 
        ret_mutants[bid].loc[deadMutKs, 'finalStatus'] = -1 

        # among the dead, revealed also included, so here, rewrite
        revealedMutsFile = os.path.join(outputdir, f"{project}.{mutatedAt}.revealed.pkl")
        revealedMutKs = getRevealedMutKs(revealedMutsFile)
        ret_mutants[bid].loc[revealedMutKs, 'finalStatus'] = 1
    return ret_mutants

def format_mutK(mutK:str) -> str:
    fpath_K, mutNo = mutK.split("-")
    fpath_K = fpath_K.replace("/", ".")
    if fpath_K.endswith(".java"): 
        fpath_K = ".".join(fpath_K.split(".")[:-1])
    return f"{fpath_K}-{mutNo}" 

def reverse_format_mutK(mutK:str) -> str:
    fpath_K, mutNo = mutK.split("-")
    if fpath_K.endswith(".java"):
        front_fpath_K = fpath_K[:-5].replace(".", "/")
        fpath_K = front_fpath_K + ".java"
    else:
        fpath_K = fpath_K.replace(".", "/") + ".java"
    return f"{fpath_K}-{mutNo}" 

def getUniqueMutants(mutLRPair_pmut_pfile:Dict[str, Dict], 
    needToFocusOps:List[str], needToFilterOps:List[str], seed:int = 0) -> Dict[str, Dict]:
    import numpy as np
    np.random.seed(seed)
    processed = dict() # key = (fpath-is_neg-pos[0]-pos[1]-right), value =
    if needToFilterOps is not None and needToFocusOps is not None:
        assert len(set(needToFilterOps).intersection(set(needToFocusOps))) == 0
    for targetFile, mutInfos in mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            mutK = (targetFile, int(mutNo))
            mutOp = mutInfo['mutOp'][0]
            # check whether to further process
            if needToFilterOps is not None and mutOp in needToFilterOps: continue 
            if needToFocusOps is not None and mutOp not in needToFocusOps: continue 
            #
            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            try:
                processed[k].append(mutK)
            except KeyError:
                processed[k] = [mutK]

    # drop redundant
    uniq_mutants = dict()
    for k, the_same_muts in processed.items():
        indices = np.arange(len(the_same_muts))
        selected = the_same_muts[np.random.choice(indices, 1)[0]]
        fpath, mutNo = selected
        if fpath not in uniq_mutants.keys():
            uniq_mutants[fpath] = dict()
        uniq_mutants[fpath][mutNo] = mutLRPair_pmut_pfile[fpath][mutNo]
    return uniq_mutants


def group_same_mutants_v1(outputdir:str, project:str, fixedRev:str, 
    needToFilterOps:List[str] = None, needToFocusOps:List[str] = None, mutLRFile:str = None, 
    regen_uniq:bool = False, 
) -> Tuple[Dict[str,List[str]], Dict[str, str]]:
    """
    mutLRFile: should be the fpath to the all mutants file: mutLRPair_pfile.pkl
    used for those revealed mutants: 
    """
    import pickle 
    if mutLRFile is None:
        mutLRFile = os.path.join(outputdir, project, "inter", fixedRev, "mutLRPair_pfile.pkl")
    with open(mutLRFile, 'rb') as f:
        mutLRPair_pmut_pfile = pickle.load(f)

    filtered = []
    processed = {} # key = (fpath-is_neg-pos[0]-pos[1]-right), value = 
    for targetFile, mutInfos in mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            mutK = format_mutK(f"{targetFile}-{mutNo}")
            mutOp = mutInfo['mutOp'][0]
            # check whether to further process
            if needToFocusOps is None:
                if needToFilterOps is not None:
                    if mutOp in needToFilterOps: 
                        filtered.append(mutK); continue 
            else:
                if mutOp not in needToFocusOps: 
                    filtered.append(mutK); continue 

            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            try:
                processed[k].add(mutK)
            except KeyError:
                processed[k] = set([mutK])

    if not regen_uniq:
        #uniq_mutLRFile = os.path.join(outputdir, project, "inter", fixedRev, "uniq_mutLRPair_pfile.pkl")
        uniq_mutLRFile = os.path.join(os.path.dirname(mutLRFile), 
            "uniq_" + os.path.basename(mutLRFile))
        with open(uniq_mutLRFile, 'rb') as f:
            uniq_mutLRPair_pmut_pfile = pickle.load(f)
    else:
        uniq_mutLRPair_pmut_pfile = getUniqueMutants(mutLRPair_pmut_pfile, needToFocusOps, needToFilterOps)
    #
    grouped = {}
    mapped_to_uniq_mutK = {}
    ks = []
    for targetFile, mutInfos in uniq_mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            mutK = format_mutK(f"{targetFile}-{mutNo}")
            if mutK in filtered: continue
            mutOp = mutInfo['mutOp'][0]
            if needToFilterOps:
                if mutOp in needToFilterOps: continue 
            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            ks.append(k)
            the_same_mutants = processed[k]
            grouped[mutK] = the_same_mutants
            for mk in the_same_mutants:
                mapped_to_uniq_mutK[mk] = mutK
    return filtered, grouped, mapped_to_uniq_mutK

def group_same_mutants(outputdir:str, project:str, fixedRev:str, 
    needToFilterOps:List[str] = None, needToFocusOps:List[str] = None, mutLRFile:str = None, 
    regen_uniq:bool = False, 
) -> Tuple[Dict[str,List[str]], Dict[str, str]]:
    """
    mutLRFile: should be the fpath to the all mutants file: mutLRPair_pfile.pkl
    used for those revealed mutants: 
    """
    import pickle 
    if mutLRFile is None:
        mutLRFile = os.path.join(outputdir, project, "inter", fixedRev, "mutLRPair_pfile.pkl")
    with open(mutLRFile, 'rb') as f:
        mutLRPair_pmut_pfile = pickle.load(f)
    all_mutKs = []
    processed = {} # key = (fpath-is_neg-pos[0]-pos[1]-right), value = 
    for targetFile, mutInfos in mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            mutK = format_mutK(f"{targetFile}-{mutNo}")
            mutOp = mutInfo['mutOp'][0]
            # check whether to further process
            #if needToFocusOps is None:
                #if needToFilterOps is not None:
                    #if mutOp in needToFilterOps: 
                        #filtered.append(mutK); continue 
            #else:
                #if mutOp not in needToFocusOps: 
                    #filtered.append(mutK); continue 
            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            try:
                processed[k].add(mutK)
            except KeyError:
                processed[k] = set([mutK])
            all_mutKs.append(mutK)

    if not regen_uniq:
        #uniq_mutLRFile = os.path.join(outputdir, project, "inter", fixedRev, "uniq_mutLRPair_pfile.pkl")
        uniq_mutLRFile = os.path.join(os.path.dirname(mutLRFile), 
            "uniq_" + os.path.basename(mutLRFile))
        with open(uniq_mutLRFile, 'rb') as f:
            uniq_mutLRPair_pmut_pfile = pickle.load(f)
    else:
        uniq_mutLRPair_pmut_pfile = getUniqueMutants(mutLRPair_pmut_pfile, needToFocusOps, needToFilterOps)
    #
    covered = []
    grouped = {}
    mapped_to_uniq_mutK = {}
    #ks = []
    for targetFile, mutInfos in uniq_mutLRPair_pmut_pfile.items():
        for mutNo, mutInfo in mutInfos.items():
            mutK = format_mutK(f"{targetFile}-{mutNo}")
            #if mutK in filtered: continue
            mutOp = mutInfo['mutOp'][0]
            # recheck: check mutOp as this info will be used as the uniq mutK
            if needToFilterOps is not None and mutOp in needToFilterOps: continue 
            if needToFocusOps is not None and mutOp not in needToFocusOps: continue 
            lno, start_p, end_p = mutInfo['pos']
            new_content = mutInfo['right']
            is_neg = isinstance(mutInfo['text'], tuple)
            k = (targetFile, lno, start_p, end_p, is_neg, new_content)
            #ks.append(k)
            the_same_mutants = processed[k]
            grouped[mutK] = the_same_mutants
            for mk in the_same_mutants:
                mapped_to_uniq_mutK[mk] = mutK
                covered.append(mk)
    filtered = list(set(all_mutKs) - set(covered))
    return filtered, grouped, mapped_to_uniq_mutK

def filter_invalid(record_file:str, to_look_mutKs:List[str]) -> List[str]:
    # record_file: e.g., output/evaluation/mut_val_check/Lang/Lang_8_mut_valid_invalid.json
    with open(record_file) as f:
        import json 
        data = json.load(f)
    #valid_mutKs = data['valid']
    invalid_mutKs = list(data['invalid'].keys())
    formatted_to_look_mutKs = list(map(format_mutK, to_look_mutKs))
    formatted_invalid_mutKs  = set(list(map(format_mutK, invalid_mutKs)))
    filtered = []
    for i,org_mutK in enumerate(to_look_mutKs):
        k = formatted_to_look_mutKs[i]
        if k not in formatted_invalid_mutKs:
            filtered.append(org_mutK) 
    return filtered 

def get_formatted_muKs_to_invalid(record_file:str) -> Set[str]:
    # record_file: e.g., output/evaluation/mut_val_check/Lang/Lang_8_mut_valid_invalid.json
    with open(record_file) as f:
        import json 
        data = json.load(f)
    invalid_mutKs = list(data['invalid'].keys())
    formatted_invalid_mutKs  = set(list(map(format_mutK, invalid_mutKs)))
    return formatted_invalid_mutKs 

def get_only_target_uniq_muts(to_look_mutKs, mapped_to_uniq_mutK:Dict[str,str]) -> List[str]:
    rets = []
    for mutK in to_look_mutKs:
        mutK = format_mutK(mutK)
        try:
            new_mutK = mapped_to_uniq_mutK[mutK]
        except KeyError:
            continue
        rets.append(new_mutK)
    return list(set(rets)) 

#def match_processed_mut_w_raw_mut(mutated_fpath:str, mutInfo:Dict, element) -> bool:
    ##import xml.etree.ElementTree as ET
    #mutOp, description = mutInfo['mutOp']
    #_mutOp = element.find("mutator").text.split(".")[-1]
    #if mutOp != _mutOp: return False
    #_mutatedClass = element.find("mutatedClass")
    #if mutated_fpath.endswith(_sourceFile):
        #_lno = element.find("lineNumber")
        #if _lno != lno: return False 
        #mutOp, description = mutInfo['mutOp']
        ##
        #_lno = element.find("lineNumber")
        #_description = element.find("description").text
        #_mutOp = element.find("mutator").text.split(".")[-1]
        #_mutatedClass = element.find("mutatedClass")
    #else:
        #return False 

def analyse_mutOp_and_mut_status(MUTOPS:List[str], df:pd.DataFrame, project:str, target_status:str, decimal_points:int, cnt_top_n_freqs, n_top:int):
    import numpy as np
    percs = []
    line = ""
    for mutOp in MUTOPS:
        adf = df.loc[df.mutOp == mutOp]
        n = len(adf)
        if n == 0: 
            percs.append(-1); continue
        n_target_muts = adf.loc[adf.status == target_status].shape[0]
        perc = np.round(100 * n_target_muts/n, decimals=decimal_points)
        percs.append(perc)
    #
    _percs = [v for v in percs]
    sorted_percs = np.sort(_percs)[::-1] 
    max_perc = sorted_percs[0]
    idx_to_max = np.where(percs == max_perc)[0]
    indices_to_remain_top = []
    if len(idx_to_max) < n_top:
        for v in sorted_percs[len(idx_to_max):n_top]:
            indices_to_remain_top.extend(np.where(percs == v)[0].tolist())
    #
    line = ""
    for i, perc in enumerate(percs):
        if perc == -1: # non applied
            line += " & -"
        else:
            if i in idx_to_max:
                #line += " & \\textbf{" + str(perc) + "}"
                line += " & \\cellcolor{blue!25}" + "\\textbf{" + str(perc) + "}"
                cnt_top_n_freqs[i] += 1
            elif i in indices_to_remain_top:
                #line += " & \\underline{" + str(perc) + "}"
                line += " & \\cellcolor{green!25}" + str(perc)
                cnt_top_n_freqs[i] += 1
            else:
                line += f" & {perc}"

    line = project + line + "\\\\"
    print (line)
    return cnt_top_n_freqs

