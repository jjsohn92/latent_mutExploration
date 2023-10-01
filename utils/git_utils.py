"""
utilities related to git
"""
import git
import pandas as pd
from typing import List, Dict, Tuple
git_show_args = ["--ignore-all-space", "--ignore-blank-lines"]
def get_repo(repo_path:str) -> git.Repo:
    return git.Repo(repo_path)

def get_tags(repo_path:str) -> Dict[str, Tuple]: 
    repo = get_repo(repo_path)
    tags = list(git.TagReference.list_items(repo))
    tag_points_at = {}
    for tag in tags:
        tag_name = tag.name 
        tag_points_at[tag_name] = (tag.commit, tag.commit.hexsha)
    return tag_points_at

def get_current_commit(repo_path:str) -> str:
    repo = get_repo(repo_path)
    return repo.head.object.hexsha[:8]

def checkout(repo_path:str, commit_hash):
    repo = get_repo(repo_path)
    repo.git.checkout("-f", commit_hash)


def check_file_exist(commit_hash:str, file_path:str, repo_path:str) -> bool:
    import subprocess 
    cmd = f"git ls-tree -r {commit_hash} -- {file_path}" \
        if file_path is not None else f"git ls-tree -r {commit_hash}"
    output = subprocess.check_output(
        cmd, shell = True, cwd = repo_path
        ).decode('utf-8', 'backslashreplace')
    return bool(output) 

def get_full_fpath(commit_hash:str, partial_file_path:str, repo_path:str, class_id:str = None) -> str:
    import subprocess 
    cmd = f"git ls-tree -r {commit_hash} | grep '{partial_file_path}'" 
    output = subprocess.check_output(
        cmd, shell = True, cwd = repo_path
        ).decode('utf-8', 'backslashreplace')
    cands =[]
    for line in output.split("\n"):
        if not bool(line): continue
        cands.append(line.split("\t")[1])
    if len(cands) == 1:
        return cands[0]
    elif len(cands) == 0:
        return None 
    else:
        if class_id is None: return None 
        last_class = partial_file_path[:-5].split("/")[-1]
        cls_tks = class_id.split(".")
        idx_to_last_class = cls_tks.index(last_class)
        class_id = "/".join(cls_tks[:idx_to_last_class + 1]) + ".java"
        for cand in cands:
            if cand.endswith(class_id):
                return cand 
        return None 
    return bool(output) 

def show_file(commit_hash:str, file_path:str, repo_path:str) -> str:
    """
    here .. need to think about subproject case (...)
    """
    import subprocess 
    cmd = f"git show {commit_hash}:{file_path}" \
        if file_path is not None else f"git show {commit_hash}"
    output = subprocess.check_output(
        cmd, shell = True, cwd = repo_path
        ).decode('utf-8', 'backslashreplace')
    return output 

def get_commits_upto_recent(
    repo_path:str, startCommit:str, branch_name:str) -> List[str]:
    import subprocess
    from subprocess import CalledProcessError
    repo = get_repo(repo_path)
    if branch_name is None:
        cmd = f"git log -n 1 --oneline"
    else:
        cmd = f"git log -n 1 {branch_name} --oneline"  
    try:
        output = subprocess.check_output(
            cmd, shell = True, cwd = repo_path).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e:
        print (e)
        assert False 
    recent_commit_hexsha = output.split(" ")[0]
    ### new (due to missing commits)
    cs = [c.hexsha[:8] for c in repo.iter_commits(branch_name)]
    idxToCut = None
    for i, c in enumerate(cs):
        if c.startswith(startCommit) or startCommit.startswith(c):
            idxToCut = i 
            break 
    assert idxToCut is not None, f"{startCommit}..{recent_commit_hexsha}"
    btwn_commits = cs[:idxToCut + 1]
    btwn_commits.reverse() # oldest to the recent
    return btwn_commits

def getDiffFiles(
    repo_path:str, curr_cid:str, prev_cid:str, 
) -> Tuple[List[str], List[str], Dict[str,List[str]]]:
    """
    return a list of files, which can be from curr_cid or prev_cid 
    -> need to handle renaming 
    """
    import subprocess
    #cmd = f"git diff -M --name-status {curr_cid} {prev_cid}"
    # -M (= --find-renames) -> to also handle moved files, as RMiner also detects moved files
    cmd = f"git diff -M -C --name-status {prev_cid} {curr_cid}" # -> the name of the
    # -> for the cost reason, we use -C option, not --find-copies-harder that takes the files unchanged
    # also as the candidates for the file copying 
    output = subprocess.check_output(
        cmd, shell = True, cwd = repo_path).decode('utf-8', 'backslashreplace')
    diffFiles = []
    deletedFiles = []
    renamedOrCopiedFiles = {}
    for line in output.split("\n"):
        line = line.strip()
        if not bool(line): continue
        ts = line.replace("\t", " ").split(" ")
        name_status, fpath = ts[0], ts[1]
        #if name_status == 'A': # not our target as it doesn't exist anymore in curr_cid
        if name_status == 'D': # not our target as it doesn't exist anymore in curr_cid 
            deletedFiles.append(fpath)
            #print (line)
            continue 
        elif name_status.startswith("R"): # rename
            new_fpath = ts[-1]
            try:
                renamedOrCopiedFiles[fpath].append(new_fpath)
            except KeyError:
                renamedOrCopiedFiles[fpath] = [new_fpath]
        elif name_status.startswith("C"): # copied file 
            new_fpath = ts[-1]
            try:
                renamedOrCopiedFiles[fpath].append(new_fpath)
            except KeyError:
                renamedOrCopiedFiles[fpath] = [new_fpath]
        diffFiles.append(fpath) # ./....SHOULD I CONSIDER THE RENAME
    return diffFiles, deletedFiles, renamedOrCopiedFiles

# slow
def getDiffFilesV2(repo_path:str, curr_cid:str, prev_cid:str) -> List[str]:
    repo = git.Repo(repo_path)
    diffs = repo.commit(curr_cid).diff(repo.commit(prev_cid))
    diffFiles = [diff.b_path for diff in diffs] # look for those in prev_cid
    return diffFiles

def getLastModifiedAt(
    repo_path:str, rev:str, fpath:str, mutStartLno:int, mutEndLno:int
) -> List[str]:
    """
    blamed output
    """
    repo = git.Repo(repo_path)
    rev_opts = ['-C', '-M', '-L'] + [f"{mutStartLno},{mutEndLno}"] 
    blamed_output = repo.blame(rev, fpath, rev_opts = rev_opts)
    commits_last_modified = [out[0].hexsha[:8] for out in blamed_output]
    return commits_last_modified 

def showMidfiedAt(
    rev:str, repo_path:str, inputfile:str, mutStartLno:int, mutEndLno:int
):
    g = git.Git(repo_path)
    logout = g.log("-r", rev, "-L", f"{mutStartLno},{mutEndLno}:{inputfile}", "-n", 1)
    now_print = False
    for logline in logout.split("\n"):
        if logline.startswith("@@"):
            now_print = True 
        else:
            if now_print and logline.startswith("+"):
                print (logline[1:])

def getModifiedAts(
    repo_path:str, fpath:str, mutStartLno:int, mutEndLno:int,
    deeperCheck:bool = False,
) -> List[str]:
    """
    -> need to think about renaming
    get a list of commits that modified a give range of lines
    """
    g = git.Git(repo_path)
    if mutStartLno is None and mutEndLno is None:
        logout = g.log("--follow", "--", fpath)
    else:
        logout = g.log("-L", f"{mutStartLno},{mutEndLno}:{fpath}")
    all_commits = getAllCommits(repo_path, None)
    commits_last_modified = []
    prev_to_commits_last_modified = []
    diff_files = []
    prev_diff_files = []
    for logline in logout.split("\n"):
        if logline.startswith("commit "):
            commit = logline.strip().split("commit ")[-1]
            commits_last_modified.append(commit[:8])
        elif logline.startswith("+++ b"):
            filename = logline.split("+++ b/")[1]
            diff_files.append(filename)
            idx_c = all_commits.index(commits_last_modified[-1])
            idx_prev_c = idx_c + 1
            prev_to_commits_last_modified.append(all_commits[idx_prev_c])
        elif logline.startswith("--- "):
            filename = logline.split("--- ")[1]
            if filename.startswith("a/"):
                filename = filename[2:]
            else:
                filename = filename[1:]
            if filename != 'dev/null':
                prev_diff_files.append(filename)
            else:
                prev_diff_files.append(None)

    if deeperCheck: # currenlty support only at file-level => needed to changed at line level
        _commits_last_modified = []
        import utils.semantic_checker as semantic_checker
        import os
        for c, prev_c, file, prevfile in zip(
            commits_last_modified, prev_to_commits_last_modified, diff_files, prev_diff_files
            ):
            if prevfile is None: # meaning added file -> not our target
                continue
            fileA = os.path.join(repo_path, f"tempA_{c}.java")
            fileB = os.path.join(repo_path, f"tempB_{prev_c}.java")
            contentA = show_file(c, file, repo_path)
            with open(fileA, 'w') as f:
                f.write(contentA)
            contentB = show_file(prev_c, prevfile, repo_path)
            with open(fileB, 'w') as f:
                f.write(contentB)
            is_the_same = semantic_checker.compareFiles(
                os.path.basename(fileA),
                os.path.basename(fileB),
                repo_path)
            if is_the_same: # the same semantic -> not our target
                continue
            else:
                _commits_last_modified.append(c)
        commits_last_modified = _commits_last_modified
    return commits_last_modified

def getModifiedAts_v2(
    repo_path:str, rev:str, fpath:str, mutStartLno:int, mutEndLno:int, 
    end_rev:str = None
) -> List[str]:
    """
    -> need to think about renaming ->looks like automatically handled
    get a list of commits that modified a give range of lines
    """
    import numpy as np
    g = git.Git(repo_path)
    if mutStartLno is None and mutEndLno is None:
        logout = g.log("-r", rev, '-C', '-M', "--follow", "--", fpath)
    else:
        logout = g.log("-r", rev, '-C', '-M', "-L", f"{mutStartLno},{mutEndLno}:{fpath}")
    #
    logsGrouped = []
    for logline in logout.split("\n"): # recent to old
        if logline.startswith("commit "):
            commit = logline.strip().split("commit ")[-1]
            logsGrouped.append([commit, []])
        else:
            logsGrouped[-1][1].append(logline)
    
    rets = {}
    all_commits = getAllCommits(repo_path, None) # fron the latest to the oldest
    if end_rev is None:
        idx_to_end = len(all_commits) - 1
    else:
        idx_to_end = all_commits.index(end_rev[:8])
    for commit, logsInCommit in logsGrouped: 
        idx_to_c = all_commits.index(commit[:8])
        if idx_to_end < idx_to_c: # if the commit is older than end_rev, break 
            break # out of our concerns (our concerns: idx_to_end >=...)
        newChgs, oldChgs = [], []
        for logline in logsInCommit:
            if logline.startswith("+++ b"):
                filename = logline.split("+++ b/")[1]
                newChgs.append([filename])
            elif logline.startswith("--- "): 
                filename = logline.split("--- ")[1]
                if filename.startswith("a/"): 
                    filename = filename[2:]
                else:
                    filename = filename[1:]
                if filename != 'dev/null':
                    oldChgs.append([filename])
                else: 
                    oldChgs.append([None])
            elif logline.startswith("@@"):
                import re 
                logline = logline.strip()
                pat = "@@.*-([0-9]+),([0-9]+)\s+\+([0-9]+),([0-9]+)\s+@@"
                matched = re.search(pat, logline)
                prev_lno, num_del, lno, num_add = map(int, matched.groups())
                prev_lnos = np.arange(prev_lno, prev_lno + num_del).tolist()
                lnos = np.arange(lno, lno + num_add).tolist()
                newChgs[-1].append(lnos)
                oldChgs[-1].append(prev_lnos)
        rets[commit] = (newChgs, oldChgs)
    return rets

def list_files_in_commit(commit):
    """
    Lists all the files in a repo at a given commit
    :param commit: A gitpython Commit object
    """
    file_list = []
    stack = [commit.tree]
    while len(stack) > 0:
        tree = stack.pop()
        # enumerate blobs (files) at this level
        for b in tree.blobs:
            file_list.append(b.path)
        for subtree in tree.trees:
            stack.append(subtree)
    # you can return dir_list if you want directories too
    return file_list

def getAllCommits(repo_path:str, branch:str = 'trunk'):
    commits = get_repo(repo_path).iter_commits(branch) if branch is not None else get_repo(repo_path).iter_commits()
    cids = [c.hexsha[:8] for c in commits]
    return cids

def setDiffMergeRenameToMax(workdir:str):
    import subprocess 
    from subprocess import CalledProcessError
    cmd = "git config merge.renameLimit 999999"
    try:
        out = subprocess.run(cmd, shell = True, cwd = workdir)
    except CalledProcessError as e:
        print (e)
        print (out.stderr)
        print (out.stdout)
    print ("Set merge renamed to the max")

    cmd = "git config diff.renameLimit 999999"
    try:
        out = subprocess.run(cmd, shell = True, cwd = workdir)
    except CalledProcessError as e:
        print (e)
        print (out.stderr)
        print (out.stdout)
        assert False
    print ("Set diff renamed to the max")


def getCommitIdx(all_commits:List[str], target:str):
    if len(target) >= 8:
        return all_commits.index(target[:8])
    else:
        for idx,c in enumerate(all_commits):
            if c.startswith(target):
                return idx  
    return None

def getCommitedDateTime(repo:git.Repo, commits_hash:str):
    return repo.commit(commits_hash).committed_datetime

def getAuthor(repo:git.Repo, commits_hash:str):
    return repo.commit(commits_hash).author.name 