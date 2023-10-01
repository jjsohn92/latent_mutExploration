import os, sys  
import glob 
from tqdm import tqdm 
from typing import List, Dict, Tuple
import gzip, pickle 
sys.path.append("../")
import utils.analysis_utils as analysis_utils
import utils.git_utils as git_utils
import utils.file_utils as file_utils
import utils.parser_utils as parser_utils

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
    'JacksonXml':os.path.join(PROJECT_REPO_DIR, 'jackson-dataformat-xml.git'),
    'Jsoup':os.path.join(PROJECT_REPO_DIR, 'jsoup.git'),
    'JxPath':os.path.join(PROJECT_REPO_DIR, 'commons-jxpath.git')
}

def load_data(fpath):
    with gzip.open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data 

def getLnoAtCommit(repo_path:str, commit_hash:str, 
    fpath:str, start_p:int, end_p:int):
    #
    file_content = git_utils.show_file(commit_hash, fpath, repo_path)
    start_end_pos_pline = file_utils.compute_start_end_pos_pline(file_content)
    modified_lnos = []
    for lno, (_start_p, _end_p) in start_end_pos_pline.items():
        if (_start_p <= start_p) and (end_p <= _end_p):
            modified_lnos.append(lno)
        if (_start_p <= end_p) and (end_p <= _end_p):
            modified_lnos.append(lno)
    modified_lnos = list(set(modified_lnos))
    return modified_lnos

def getAppliedAtInfo(output_dir:str, project:str, 
    fixedRev:str, revealedAt:str) -> Dict:
    appliedAts_file = os.path.join(output_dir, 
        f"{project}/inter/{fixedRev}/appliedAt.{revealedAt}.json")
    print (appliedAts_file)
    if os.path.exists(appliedAts_file):
        with open(appliedAts_file) as f:
            import json 
            data = json.load(f)
        return data 
    else:
        return None 

def getMutLoc_atRevealed(lno_revealedAt:int, build_pos_dict, lno_to_node_dict):
    # might-be
    all_chgd_lnos = parser_utils.get_lno_positions_of_chgd_nodes(
        lno_revealedAt, build_pos_dict, lno_to_node_dict)
    return all_chgd_lnos 

def getRevealedAtFile(mut_appliedAts_info:Dict, mutLoc:Tuple[int,int]) -> str:
    revealedAt_fpath = None
    for fpath, appliedAts in mut_appliedAts_info.items():
        for _, appliedAt in appliedAts:
            if (appliedAt[0] == mutLoc[-2]) and (appliedAt[1] == mutLoc[-1]):
                revealedAt_fpath = fpath 
                break 
        if revealedAt_fpath is not None: break 
    assert revealedAt_fpath is not None, mutLoc
    return revealedAt_fpath

def process_and_add_lno_info_pmut(
    datafile, 
    project, 
    revealedAt, 
    appliedAts_info:Dict,
    mapped_to_uniq_mutK:Dict, 
    covg_of_muts:Dict = None, 
    ##
    d_file_contents = None, 
    parsed = None, 
    parsed_pos_node_dict = None, 
    **kwargs
):
    # e.g., coverageAt.57afb23d_src.main.java.org.apache.commons.math.geometry.euclidean.threed.Rotation-117.pkl
    if datafile is None:
        assert covg_of_muts is not None
        mutated_fpath = kwargs['mutated_fpath']
        mutNo = kwargs['mutNo']
    else:
        covg_of_muts = load_data(datafile)
        mutK = os.path.basename(datafile).split("_")[1][:-4]
        mutated_fpath, mutNo = mutK.split("-")
        mutated_fpath = mutated_fpath.replace(".", "/") + ".java"
        mutNo = int(mutNo)
        #print(datafile)
    
    try:
        mut_appliedAts_info = appliedAts_info[mutated_fpath][str(mutNo)]
    except KeyError:
        _mutfpath_K = mutated_fpath[:-5].replace("/", ".")
        k_to_uniq_mut = mapped_to_uniq_mutK[f"{_mutfpath_K}-{mutNo}"]
        mutNo = k_to_uniq_mut.split("-")[1]
        mut_appliedAts_info = appliedAts_info[mutated_fpath][str(mutNo)]

    processed_covg_of_muts = {}    
    if d_file_contents is None:
        d_file_contents, parsed, parsed_pos_node_dict = {}, {}, {}
    #print ("starting", len(d_file_contents), len(parsed), len(parsed_pos_node_dict))
    #print ("\t", d_file_contents.keys())
    for mutLoc, covg_of_a_mut in covg_of_muts.items():
        revealedAt_fpath = getRevealedAtFile(mut_appliedAts_info, mutLoc)
        lno_revealedAt = min(getLnoAtCommit(PROJECT_REPOS[project], revealedAt, 
            revealedAt_fpath, mutLoc[-2], mutLoc[-1]))
        #print ("Revealed at", lno_revealedAt)
        try:
            build_pos_dict, lno_to_node_dict = parsed_pos_node_dict[revealedAt_fpath]
        except KeyError:
            try:
                tree = parsed[revealedAt_fpath]
            except KeyError:
                try:
                    file_content = d_file_contents[revealedAt_fpath]
                except KeyError:
                    file_content = git_utils.show_file(
                        revealedAt, revealedAt_fpath, PROJECT_REPOS[project])
                    d_file_contents[revealedAt_fpath] = file_content
                tree = parser_utils.parse(file_content)
                parsed[revealedAt_fpath] = tree 
            build_pos_dict, lno_to_node_dict = parser_utils.build_positon_dict(tree)
            #print ([(type(v), v.position) for v in lno_to_node_dict[455]])
            #print ([(type(v), v.position) for v in lno_to_node_dict[456]])
            #print (lno_to_node_dict[455][0] == lno_to_node_dict[456][0])
            #sys.exit()
            #print ('At here', revealedAt_fpath, revealedAt)
            parsed_pos_node_dict[revealedAt_fpath] = (build_pos_dict, lno_to_node_dict)
            #print ('here!')
        new_all_chgd_lnos = getMutLoc_atRevealed(
            lno_revealedAt, build_pos_dict, lno_to_node_dict)
        min_new_lno, max_new_lno = min(new_all_chgd_lnos), max(new_all_chgd_lnos)
        #print ("final", min_new_lno, max_new_lno)
        #sys.exit()
        #new_mutLoc = (lno_at_revealed, mutLoc[0], mutLoc[1])
        new_mutLoc = (f"{min_new_lno}:{max_new_lno}", mutLoc[-2], mutLoc[-1])
        #print ("New mut key", new_mutLoc)
        processed_covg_of_muts[new_mutLoc] = covg_of_a_mut
    return processed_covg_of_muts


def process_and_add_lno_info(
    datafile, 
    project, 
    revealedAt, 
    appliedAts_info:Dict,
    mapped_to_uniq_mutK:Dict, 
):
    print (datafile)
    covg_data = load_data(datafile)
    processed_covg = {}    
    d_file_contents, parsed, parsed_pos_node_dict = {}, {}, {}
    for mutK, covg_of_muts in covg_data.items():
        mutated_fpath, mutNo = mutK.split("-")
        mutNo = int(mutNo)
        print ('Processing', mutK, revealedAt)
        processed_covg[mutK] = process_and_add_lno_info_pmut(
            None, 
            project, 
            revealedAt, 
            appliedAts_info, # geting per-mut info will be handled inside
            mapped_to_uniq_mutK, 
            covg_of_muts, 
            #
            d_file_contents = d_file_contents, 
            parsed = parsed, 
            parsed_pos_node_dict = parsed_pos_node_dict, 
            #
            mutated_fpath = mutated_fpath, 
            mutNo = mutNo, 
        )
    return processed_covg

def combine_appliedAtInfos(appliedAts_info_a:Dict, appliedAts_info_b:Dict) -> Dict:
    if appliedAts_info_a is not None:
        a_fks = list(appliedAts_info_a.keys())
    else:
        a_fks = []
    if appliedAts_info_b is not None:
        b_fks = list(appliedAts_info_b.keys())
    else:
        b_fks = []
    ab_fks = list(set(a_fks + b_fks))
    combined = {}
    for fpath in ab_fks:
        combined[fpath] = {}
        try:
            row_a = appliedAts_info_a[fpath] # pmut
            combined[fpath].update(row_a)
        except Exception:
            pass 
        try:
            row_b = appliedAts_info_b[fpath] # pmut 
            combined[fpath].update(row_b)
        except Exception:
            pass
    return combined

if __name__ == "__main__":
    import argparse 
    root_outputdir = sys.argv[1]
    # additional
    if root_outputdir.endswith("/"): root_outputdir = root_outputdir[:-1]
    root_outputdir_for_remain = root_outputdir + "_remain"
    root_covg_dir = sys.argv[2]
    projects = ['Lang', 'Math', 'Time', 'Closure', 
                'Cli', 'Compress', 'Codec', 'Collections', 'Csv', 
                'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'JxPath', 'Jsoup']
    #projects = ['Compress', 'Codec', 'Collections', 'Csv', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'JxPath', 'Jsoup']
    #projects = ['Compress', 'Codec', 'Collections', 'Csv',  'JacksonCore', 
    # 'JacksonDatabind', 'JacksonXml', 'JxPath', 'Jsoup']
    d4j_home = os.getenv("D4J_HOME")
    for project in projects:
    #for project in ['Lang']:
        print (project)
        bidToRev, revToBid = analysis_utils.getBidRevDict(d4j_home, project)
        files = glob.glob(os.path.join(root_covg_dir, f"{project}_*/coverageAt.*.pkl")) # here, currently, two types of files may exist: per bug, or per mut 
        if len(files) == 0:
            print (f"None in {project}")
            continue
        for file in tqdm(files):
            dirpath = os.path.dirname(file)
            bid = int(dirpath.split("/")[-1].split("_")[1])
            #if bid != 67: continue
            fixedRev = bidToRev[bid]
            revealedAt = os.path.basename(file).split(".")[1] 
            ## here, we don't have to think about needToFilter as they are alredy filterd out durign the collecitn if the filter is on
            _, _, mapped_to_uniq_mutK = analysis_utils.group_same_mutants(root_outputdir, project, fixedRev, needToFilterOps=NEED_TO_FILTER_OPS) # key: e.g., src.main.java.org.apache.commons.lang3.math.NumberUtils-38
            ##
            #if revealedAt != '80b1e90b':continue
            if "_" in revealedAt: # the coverage file colleced per mut 
                revealedAt = revealedAt.split("_")[0] # 
                is_per_mut = True 
            else:
                is_per_mut = False
            appliedAts_info = getAppliedAtInfo(root_outputdir, project, fixedRev, revealedAt)
            #if appliedAts_info is None and NEED_TO_FILTER_OPS is not None:
            appliedAts_info_remain = getAppliedAtInfo(root_outputdir_for_remain, project, fixedRev, revealedAt)
            # combine 
            appliedAts_info = combine_appliedAtInfos(appliedAts_info, appliedAts_info_remain)
            if not is_per_mut:
                covg_data = process_and_add_lno_info(
                    file, project, revealedAt, appliedAts_info, mapped_to_uniq_mutK)
            else:
                ## here error!
                covg_data = process_and_add_lno_info_pmut(
                    file, project, revealedAt, appliedAts_info, mapped_to_uniq_mutK)
            ## overwrite
            with gzip.open(file, 'wb') as f:
                pickle.dump(covg_data, f)
            print (f"Save to {file}")
