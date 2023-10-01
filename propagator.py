"""
"""
import os
from typing import List, Dict, Tuple, Set, Union
import pandas as pd 
from utils import git_utils, file_utils, mvn_utils, java_utils, ant_d4jbased_utils
import time 
from mutants.pit_mutant import PitMutantCont

def run_cmd(cmd:str, workdir:str):
    import subprocess 
    from subprocess import CalledProcessError
    try:
        output = subprocess.check_output(
            cmd, shell = True, cwd = workdir).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e: # ... this actually will also catch a simple test failure
        print (cmd)
        print (e)
        return None, e
    return output, None

MATCHING_TYPES = {
    'name':['SimpleName'], 
    'operator':['.*OPERATOR'], 
    'call':['MethodInvocation'], 
    'literal':['.*Literal', 'METHOD_INVOCATION_ARGUMENTS'], 
    'argument_list':None 
}

def checkIn(target_type:str, to_compare_type:str) -> bool: 
    return target_type == to_compare_type

class MutantPropagator():
    RMINER_HOME = os.path.join(os.getenv("RMINER_HOME"), "bin")
    MAX_CONT_COMPILATION_FAIL_PASS = 10 # the maximum number of compilation failure
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_DefaultDiffMapAndRefactorFiles(dest:str, fileKey:str) -> Tuple[str,str]:
        dstDiffMapJSONFile = os.path.join(dest, f"diffMap_{fileKey}.json")
        dstRefactorJSONFile = os.path.join(dest, f"refactorings_{fileKey}.json")
        return dstDiffMapJSONFile, dstRefactorJSONFile

    @staticmethod
    def readDiffFile(
        dstDiffMapJSONFile:str, 
        workdir:str, 
        currCommit:str, prevCommit:str
    ) -> Dict[Tuple, pd.DataFrame]:
        # will check gumtree 
        import json 
        with open(dstDiffMapJSONFile) as f:
            diffData = json.load(f)
        formatted = {} # Key: srcdest -> (srcpath=prev, destpath=curr)
        for srcdest, vs in diffData.items():
            # curr_fpath -> from the commit 
            prev_fpath, curr_fpath = srcdest.split("-")
            data = {
                'prev_type':[], 'curr_type':[],
                'prev_pos':[], 'curr_pos':[], 
                'prev_content':[], 'curr_content': []    
            }
            for v in vs:
                prev_v, curr_v = v 
                data['prev_type'].append(prev_v[0])
                if len(prev_v) == 3:
                    data['prev_content'].append(prev_v[1])
                    data['prev_pos'].append(tuple(prev_v[2]))
                else:
                    data['prev_content'].append(None)
                    data['prev_pos'].append(tuple(prev_v[1]))
                data['curr_type'].append(curr_v[0])
                if len(curr_v) == 3:
                    data['curr_content'].append(curr_v[1])
                    data['curr_pos'].append(tuple(curr_v[2]))
                else:
                    data['curr_content'].append(None)
                    data['curr_pos'].append(tuple(curr_v[1]))
            data = pd.DataFrame(data)
            ###  new 
            ### here, only the cases of infix ops are considered
            prevFileContent = git_utils.show_file(prevCommit, prev_fpath, workdir)
            currFileContent = git_utils.show_file(currCommit, curr_fpath, workdir)
            import utils.gumtree as gumtree 
            _, added_missed_infixOps = gumtree.processABFile_gumtree(
                prevFileContent, currFileContent, ret_added=True)
            mappedPairs = gumtree.mapPositions_fromList(added_missed_infixOps)
            mappedPairs_data = pd.DataFrame(mappedPairs)
            data = pd.concat([data, mappedPairs_data])
            formatted[(prev_fpath, curr_fpath)] = data
        return formatted 

    @staticmethod
    def readRefactoringFile(dstRefactorJSONFile:str) -> pd.DataFrame:
        # this one maynot be needed
        import json 
        with open(dstRefactorJSONFile) as f:
            refactorings = json.load(f)
        refactorings = refactorings['commits'][0] # since we work per commit, this is always a fixed length list
        # 'repository', 'sha1', 'url', 'refactorings' -> for each item in refactorings
        refactoring_data = {'idx':[], 'filePath':[], 'isLeft':[], 'start':[], 'end':[], 'codeElementType':[]}
        for idx, refactoring in enumerate(refactorings['refactorings']): # 
            # 'type': the type of refactoring
            # 'description': a detailed description of the refactoring
            # leftSideLocations': from the previous : List of CodeRange (Dict)
            # 'rightSideLocations: the current : List of CodeRange (Dict)
            for ls in refactoring['leftSideLocations']:
                refactoring_data['idx'].append(idx)
                refactoring_data['isLeft'].append(1)
                refactoring_data['filePath'].append(ls['filePath'])
                refactoring_data['start'].append((ls['startLine'], ls['startColumn']))
                refactoring_data['end'].append((ls['endLine'], ls['endColumn']))
                refactoring_data['codeElementType'].append(ls['codeElementType'])
            for rs in refactoring['rightSideLocations']:
                refactoring_data['idx'].append(idx)
                refactoring_data['isLeft'].append(0)
                refactoring_data['filePath'].append(rs['filePath'])
                refactoring_data['start'].append((rs['startLine'], rs['startColumn']))
                refactoring_data['end'].append((rs['endLine'], rs['endColumn']))
                refactoring_data['codeElementType'].append(rs['codeElementType'])
        return pd.DataFrame(refactoring_data)

    @staticmethod 
    def run_gumtree_v2(
        workdir:str, currCommit:str, prevCommit:str, files:Dict[str,str],  
        renamedFiles:Dict[str,List[str]]
    ): 
        # workdir -> already checkout to currCommit (since this is called after single_run)
        import utils.gumtree as gumtree 
        from subprocess import CalledProcessError
        _currfiles = None
        ret_positions = {}
        for prev_file, orgFpath in files.items():
            # get previous file content 
            prevContent = git_utils.show_file(prevCommit, prev_file, workdir)
            # get current file content
            new_files = [prev_file] if prev_file not in renamedFiles.keys() else renamedFiles[prev_file]
            #print ("\t", new_files)
            ret_positions[(orgFpath, prev_file)] = []
            for new_file in new_files:
                try:
                    currContent = git_utils.show_file(currCommit, new_file, workdir)
                except CalledProcessError:
                    basename = os.path.basename(new_file)
                    if _currfiles is None:
                        repo = git_utils.get_repo(workdir)
                        _currfiles = git_utils.list_files_in_commit(repo.commit(currCommit))
                    for _currfile in _currfiles:
                        if not _currfile.endswith('.java'):
                            continue
                        _basename = os.path.basename(_currfile)
                        if basename == _basename: ## temporary
                            new_file = _currfile # found

                            break 
                    assert new_file is not None, basename 
                    currContent = git_utils.show_file(currCommit, new_file, workdir)
                # run gumtree to get the parsing results
                tree = gumtree.processABFile_gumtree(prevContent, currContent)
                mappedPositiosn = gumtree.mapPositions(tree) # prevContent -> will be the key 
                ret_positions[(orgFpath, prev_file)].append((new_file, mappedPositiosn))
        #
        return ret_positions

    @staticmethod
    def updateMutPosByGumtree(
        ret_positions:Dict, 
        mutConts_pfile:Dict[str, Dict], 
        survivedMuts:Dict[str,List[int]]
    ):
        to_removed = []
        for orgFpath, mutConts in mutConts_pfile.items():
            for mutNo in mutConts.keys():
                _new_appliedAts = {}
                for prevFpath, appliedAtPFile in mutConts[mutNo].appliedAts.items():
                    k = (orgFpath, prevFpath)
                    if k in ret_positions.keys():
                        for newFPath, mappedPositiosn in ret_positions[(orgFpath, prevFpath)]:
                            for toplt_txt, toplt_loc  in appliedAtPFile:
                                try:
                                    new_loc = mappedPositiosn[toplt_loc] # but, this is just one 
                                except KeyError as e:
                                    #print ("key error", e)
                                    #print ("\t", mutNo, mutConts[mutNo].appliedAts)
                                    continue # b/c failed to handle 
                                try:
                                    _new_appliedAts[newFPath].append([toplt_txt, new_loc])
                                except KeyError:
                                    _new_appliedAts[newFPath] = [[toplt_txt, new_loc]]
                    else: # not our target 
                        _new_appliedAts[prevFpath] = appliedAtPFile 
                mutConts_pfile[orgFpath][mutNo].appliedAts = _new_appliedAts 
        # clean-up
        for orgFpath, mutNo in to_removed:
            del mutConts_pfile[orgFpath][mutNo]
            survivedMuts[orgFpath].remove(mutNo)

    @staticmethod
    def single_run(
        targetFiles:str, workdir:str, cid:str, 
        dstDiffMapJSONFile:str, dstRefactorJSONFile:str, 
    ) -> bool:
        """
        here, targetFiles contains a list of files that exist in the previous commit of cid
            -> for diffAtCommit: between prev_commit and cid (so to cid)
        """
        import subprocess 
        from subprocess import CalledProcessError
        rminer_binary = os.path.join(MutantPropagator.RMINER_HOME, "RefactoringMiner")
        targetFiles_str = ",".join(targetFiles)
        cmd = f"{rminer_binary} -cd {workdir} {cid} {targetFiles_str} {dstDiffMapJSONFile} -json {dstRefactorJSONFile}"
        try:
            output = subprocess.check_output(
                cmd, shell=True, cwd = workdir).decode('utf-8', 'backslashreplace')
        except CalledProcessError as e:
            print (cmd)
            print ("output:\n", output)
            print (e)
            return False  
        return True

    @staticmethod
    def formulate_mutLRPairs(mutLRPairInfo_pfile:Dict[str, Dict]) -> Dict[str, Dict[int, PitMutantCont]]:
        mutConts_pfile = {}
        for mutFpath, mutLRPairInfos in mutLRPairInfo_pfile.items():
            if len(mutLRPairInfos) == 0: # nothing:
                continue 
            mutConts_pfile[mutFpath] = {}
            for mutNo, mutLRPairInfo in mutLRPairInfos.items():
                mutCont_inst = PitMutantCont(
                    mutNo, 
                    mutLRPairInfo['left'], 
                    mutLRPairInfo['right'], 
                    mutLRPairInfo['mutOp'], 
                    mutLRPairInfo['targeted'], #  
                    mutFpath,
                    mutLRPairInfo['pos'],
                    mutLRPairInfo['text'], 
                ) 
                mutConts_pfile[mutFpath][mutNo] = mutCont_inst
        return mutConts_pfile

    @staticmethod 
    def match(diffMaps:pd.DataFrame, toplt_loc:Tuple, toplt_type:str) -> Tuple[int, pd.Series]:
        """
         # -> should be about the tag, instead of content, which may not have
        only look at the poistion (can be None, when it is deleted)
        return (an index to mapping, new location)
        """
        foundMap = diffMaps.loc[diffMaps.prev_pos == toplt_loc]
        if len(foundMap) == 1: # 
            ret = foundMap.iloc[0]
            return (foundMap.index.values[0], ret)
        elif len(foundMap) > 0: # more than one mapping, then we further check 
            _foundMap = foundMap.loc[
                foundMap.prev_type.apply(lambda v:checkIn(v, toplt_type))
            ]
            if len(_foundMap) == 1:
                ret = _foundMap.iloc[0]
                return (_foundMap.index.values[0], ret) 
            elif len(_foundMap) > 1: # still more than one -> something wrong 
                print (f"Shouldn't be happened: remain: {len(_foundMap)}")
                return (None, None) 
            else: 
                return (None, None) # For the deleted element .. but in fact, sometimes for method invoc, prev_content can be indeed 
        else: # len(foundMap) == 0 
            print ('no fond map...', toplt_loc)
            return (None, None)  # e.g., deleted 
    
    @staticmethod
    def checkMatching(
        foundMapping:pd.Series, 
        new_toplt_txt:str, 
        contentToCompare:str, 
        typeToCompare:str, 
        mutOp:Union[str, Tuple[str,str]]
    ) -> bool:
        """
        check the found matching
        typeToCompare -> prev type (since we do not allow type change (at leasr for pit), it will be the original type)
        """
        mappedEType = foundMapping.prev_type # from diffMaps, which is based on Gumtree
        mappedEContent = new_toplt_txt #foundMapping.prev_content # -> for some cases, this may not work 
        is_targeted_mut_type = checkIn(mappedEType, typeToCompare)
        if is_targeted_mut_type:
            #print ("\t", contentToCompare == mappedEContent, contentToCompare, mappedEContent, typeToCompare, mutOp[0])
            if mutOp[0] in ['MATH', 'CONDITIONALS_BOUNDARY', 'INCREMENTS', 'INVERT_NEGS', 'NEGATE_CONDITIONALS']:
                return contentToCompare == mappedEContent  
            elif mutOp[0] in ['EMPTY_RETURNS', 'FALSE_RETURNS', 'TRUE_RETURNS', 'NULL_RETURNS', 'PRIMITIVE_RETURNS', 'VOID_METHOD_CALLS']: 
                if typeToCompare.endswith("Name"): # variable name change, skip it 
                    return True 
                else:
                    return contentToCompare == mappedEContent 
            else:
                print (f"Unsupported mutation operator: {mutOp[0]}, {mutOp[1]}")
                assert False
        else:
            return False 
            
    @staticmethod
    def updateMutPos(
        targetFileContent:str,
        diffMaps:pd.DataFrame, 
        newPath, prevPath, 
        mutCont:PitMutantCont, 
    ) -> None:
        """
        Per-mutant
        diffMaps -> for a single file
        """
        for i, (toplt_txt, toplt_loc) in enumerate(mutCont.appliedAts[prevPath]): 
            # toplt_type -> the type of mutated code element: type changed, no-match
            toplt_type = mutCont.target_node.attrib['type'] # from Gumtree 
            # match
            matchIdx, matchedMap = MutantPropagator.match(diffMaps, toplt_loc, toplt_type) 
            if matchIdx is None: # target was deleted
                continue
            else: # valid then further check by comparing the text of mutated element: toplt_txt
                new_toplt_txt = targetFileContent[matchedMap['curr_pos'][0]:matchedMap['curr_pos'][1]]
                if not MutantPropagator.checkMatching(
                    matchedMap, new_toplt_txt, toplt_txt, toplt_type, mutCont.mutOp, 
                ): # failed to map: mismatch in parsing or changed line changed & mutant no longer exists
                    continue # skip this one 
                # update the position information: add to tempApplyAt for the later injection
                mutCont.addNewApplyAt(prevPath, newPath, new_toplt_txt, matchedMap['curr_pos']) 

    ### update core
    @staticmethod
    def applyMutations(
        workdir:str, 
        commit_hash:str, 
        mutConts_pfile:Dict[str, Dict], 
        diffMaps:Dict[Tuple, pd.DataFrame], 
        targetFilesTracking:Dict[str, List[str]], 
        testClassPatterns:Union[Dict[str,str], str],  
        woMutFailed_or_Error_Tests:Tuple[Set[str], Set[str]], 
        timeout:int, 
        dest:str,
        skipTesting:bool = False, 
        ant_or_mvn:str = 'mvn', 
        **kwargs
    ) -> Tuple[Dict[str,Dict], Dict[str,List[int]]]:
        """
        workdir = repo
        commit_hash -> for saving mutation information -> e.g., where to apply (loc at commit_hash)
        mutLRPair_pmut[str, Dict]:
            key: targetFile -> match with the original file path, 
            value: 
                key = mutant number (int)
                value = a dictionary of mutant pair info (org & mut)
                    left, right, pos, mutOp, 
                    targeted: {node, start, end, start_chars_cnt, end_chars_cnt}
        diffMaps [Tuple, pd.DataFrame] -> will contain only the filesToInspect 
            key: prevfpath, currfpath 
            value: a dataframeo of diff 
                columns: prev_type, curr_type, prev_pos, curr_pos, prev_content, curr_content
        targetFilesTracking [str, List[str]]:
            key = an original file path of the mutated file (the starting point)
            value = a list of files diverged from the key (may (likely) include the key itself)
       
        # doing: process per commit 
        1) mapping between the elements of previous and current commits
        2) injection of mutation based on this mapping 

        woMutFailedTests -> on dvgdFpath(?)
        """
        def getKeysToDiffMapsToLook(
            prev_fpath:str, prevAndCurrFpathPairs:List[Tuple[str,str]]
        ) -> List[Tuple[str,str]]:
            target_keys = []
            for k in prevAndCurrFpathPairs:
                _prev_fpath = k[0] # k = (prev_fpath, curr_fpath)
                if prev_fpath == _prev_fpath:
                    target_keys.append(k)
            return target_keys
        noMutRemains = {}
        revealedMuts = {} # contain a least of revealed mutants
        for orgFpath, dvgdFpaths in targetFilesTracking.items(): # orgFpath -> dvgdFpaths 
            ## .. -> if orgFpath and dvgdFpath are the same ... 
            mutConts = mutConts_pfile[orgFpath] # (expected to cluster by orgFpath)
            revealedMuts[orgFpath] = {}
            for dvgdFpath in dvgdFpaths: # apply to every variation 
                # get files to apply mutations
                targetKeys = getKeysToDiffMapsToLook(dvgdFpath, list(diffMaps.keys())) # this can be [] when the file is not changed
                # check for each mutant 
                # even though the mutant itself may not be modified, it can be broken, and thereby we check all in the same file
                # by the changes in others, plus, the location likely changes
                for targetKey in targetKeys: 
                    toMutatedFPath = targetKey[1] 
                    targetFileContent = file_utils.readFile(os.path.join(workdir, toMutatedFPath)) 
                    for mutNo, mutCont in mutConts.items(): 
                        # check whether mutNo is already covered
                        if mutNo in revealedMuts[orgFpath].keys():continue
                        # new mapping info included in mutCont.tempApplyAts and no-valid removed
                        # tempApplyAts udpated in mutCont -> for the following injectAndTest (preparation)
                        MutantPropagator.updateMutPos(
                            targetFileContent, 
                            diffMaps[targetKey], 
                            toMutatedFPath, dvgdFpath, 
                            mutCont
                        ) 
                        # per mutant, but here, mutlipe variations of mutants (by propagation) can be tested (self.appliedAts[orgFpath] ...)
                        if not skipTesting:
                            t1 = time.time()
                            revealedMutInfo = mutCont.injectAndTest(
                                workdir, # checkout dir = repo  
                                targetFileContent, # checkout & get
                                dvgdFpath, 
                                toMutatedFPath,
                                testClassPatterns[toMutatedFPath] if isinstance(testClassPatterns, Dict) else testClassPatterns, 
                                woMutFailed_or_Error_Tests,
                                timeout = timeout,
                                with_test_compile = False, 
                                ant_or_mvn = ant_or_mvn, 
                                **kwargs
                            ) #-> here, in tempApplyAts will contain 
                            if revealedMutInfo is None: # unexpected error occurs & stop
                                return (None, None)
                            # mutation application -> the target doesn't chagned
                            elif len(revealedMutInfo) > 0: # this mutant has been revealed
                                revealedMuts[orgFpath][mutNo] = revealedMutInfo # excldue alredy broken mutant
                        else:
                            continue # do nothing 
                # update dvgdFpath e of appliedAts (invalid were already handled by deletedNoLogerValids in injectAndTest)
                if len(targetKeys) > 0: # targetKeys: new fpaths for dvgdFpath
                    for mutNo in mutConts: # mutConts -> for
                        mutConts[mutNo].updateAppliedAts(dvgdFpath) 
            if len(revealedMuts[orgFpath]) == 0: 
                del revealedMuts[orgFpath] # b/c nothing to return 
            
            # due to this, unless the case of unexpected return (None, None), noMutRemains will always have orgFpath
            noMutRemains[orgFpath] = []
            for mutNo in mutConts: # check for mutants in orgFpath
                if mutConts[mutNo].isEmpty(): # no appliedAt remains 
                    noMutRemains[orgFpath].append(mutNo)
        _ = MutantPropagator.saveAppliedAtInfoOfMuts(mutConts_pfile, commit_hash, dest)
        return revealedMuts, noMutRemains

    @staticmethod
    def updateSurvivedMutants(
        survivedMuts:Dict[str,List[int]], 
        revealedMuts:Dict[str,Dict[int,Dict]], 
        noMutRemains:Dict[str,List[int]]
    ):
        # both the first key is the file path 
        for fpath in list(survivedMuts.keys()):
            mutNos = survivedMuts[fpath]
            mutNosOfNoRemain = noMutRemains[fpath]
            try:
                mutNosOfRevealed = list(revealedMuts[fpath].keys())
            except KeyError: # b/c can nothing revealed from the mutants in fpath (deleted)
                mutNosOfRevealed = []
            mutNosToExclude = mutNosOfNoRemain + mutNosOfRevealed
            mutNosToExclude = set(mutNosToExclude)
            remained = [mutNo for mutNo in mutNos if mutNo not in mutNosToExclude]
            if len(remained) > 0:
                survivedMuts[fpath] = remained
            else: # nothing remained 
                del survivedMuts[fpath]

    @staticmethod
    def getMutContOfSurvived(
        mutConts_pfile, 
        survivedMuts:Dict[str,List[int]]
    ):
        ret = {}
        for fpath in survivedMuts:
            ret[fpath] = {}
            for mutNo in survivedMuts[fpath]:
                ret[fpath][mutNo] = mutConts_pfile[fpath][mutNo]
        return ret  

    @staticmethod
    def getTargetTests():
        pass 

    @staticmethod 
    def saveAppliedAtInfoOfMuts(
        mutConts_pfile:Dict[str, Dict[int,PitMutantCont]], 
        commit_hash:str, 
        dest:str, 
    ) -> str:
        combinedToSave = {}
        for orgFpath, mutConts in mutConts_pfile.items():
            combinedToSave[orgFpath] = {}
            for mutNo, mutCont in mutConts.items():
                combinedToSave[orgFpath][int(mutNo)] = mutCont.appliedAts 
        mutinfo_file = os.path.join(
            dest, 
            f"appliedAt.{commit_hash[:8]}.json"
        ) 
        with open(mutinfo_file, 'w') as f:
            import json  
            f.write(json.dumps(combinedToSave))
        return mutinfo_file
    
    @staticmethod
    def run(
        workdir:str, 
        mutLRPairInfo_pfile:Dict[str, Dict], 
        commonGID:str, 
        targetCommits:List[str], 
        dest:str, 
        _testClassPatterns:str = None, 
        ant_or_mvn:str = 'mvn',
        **kwargs
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        workdir -> repodir
        targetFiles -> a list of files to track. Will be dynami
        targetCommits -> should include the commit where the mutation was applied (since we need it for ..diff)

        Return: 
        - revealedMuts  
            key: path to the target files (targetFiles)
            value: Dict
                key:int (mutNo) 
                value: a dictionary about mutation revealing (tests, location at the time, mutant info)
        - survivedMuts:
            key: path to the target files 
            value (List[int]): a list of mutNos survived (none of its variatnts killed) 
        - refactoringOccurred: 
            key: commit 
            value: a list of refacroings collected => later for analysis ()e.g., comparison to SZZ)
        """
        from tqdm import tqdm 
        from mutants import pitMutationTool

        # inspect commits in targetCommits 
        def getFilesToInspect(
            trackDict:Dict, diffFiles:List[str], 
        ) -> Dict[str,str]:
            filesToInspect = {} # key: tracking file , value: org file
            for orgFpath, tracked in trackDict.items():
                _ks = {k:True for k in tracked} 
                for diffFile in diffFiles: # will be faster
                    try:
                        _ = _ks[diffFile] # diff file in one of the tracked file (tracked -> contain the files from the last commit)
                        filesToInspect[diffFile] = orgFpath # = one of tracked
                    except KeyError: 
                        continue 
            return filesToInspect

        def updateFileTracking(
            trackDict:Dict[str,List[str]],
            updated:Dict[str,List[str]],
            survivedMuts:Dict[str,List[int]]
        ):
            # general file update
            for (prev_fpath, org_fpath), new_fpaths in updated.items():
                try:
                    _ = survivedMuts[org_fpath] 
                    if prev_fpath not in new_fpaths: 
                        trackDict[org_fpath].remove(prev_fpath) 
                    trackDict[org_fpath].extend(new_fpaths)
                    trackDict[org_fpath] = list(set(trackDict[org_fpath]))
                except KeyError: # all revealed (= no mutants in org_fpath survived to this point)
                    del trackDict[org_fpath] # no need to track further
        
        def updateFileTrackingByDelete(
            trackDict:Dict[str,List[str]],
            deletedFiles:List[str]
        ):
            # by deletion
            deletedFiles = set(deletedFiles)
            for orgFpath, tracked in trackDict.items():
                tracked = [afile for afile in tracked if afile not in deletedFiles] # excluded deleted files from tracking
                trackDict[orgFpath] = tracked  # here, if nothing left to be tracked, then 

        def updateFileTrackingByRename(
            trackDict:Dict[str,List[str]],
            renamedFiles:Dict[str,List[str]]  
        ):
            for orgFpath, tracked in trackDict.items():
                _tracked = []
                for afile in tracked:
                    try:
                        new_afiles = renamedFiles[afile]
                    except KeyError:
                        new_afiles = [afile] 
                    _tracked.extend(new_afiles)
                trackDict[orgFpath] = _tracked 

        def updateMutContByDelete(
            mutConts_pfile:Dict[str, Dict],
            deletedFiles:List[str] 
        ):
            for mutConts in mutConts_pfile.values():
                for mutCont in mutConts.values(): # here, mutCont not yet-updated for currCommit 
                    mutCont.deleteNoLongerExistByDelFiles(deletedFiles) # delete from self.appliedAts 
        
        def updateSurvMutByDelete(
            survivedMuts:Dict[str,List[int]], 
            deletedFiles:List[str]
        ):
            for deletedFile in deletedFiles:
                try: 
                    del survivedMuts[deletedFile]
                except KeyError: 
                    continue
        
        def updateMutContNoRemain(
            mutConts_pfile:Dict[str,Dict], 
            noMutRemains:Dict[str,List[int]]
        ):
            #if noMutRemains is not None: # if None, nothing to do
            for fpath, mutNos in noMutRemains.items(): # fpath = org_fapth 
                for mutNo in mutNos:
                    del mutConts_pfile[fpath][mutNo] # stop tracking 

        def updateMutDeatAts(
            commit_hash:str, 
            mutDeadAts:Dict[Tuple[str,int],str],
            survivedMuts:Dict[str,List[int]], 
        ):
            survived = {}
            for fpath in list(survivedMuts.keys()):
                mutNos = survivedMuts[fpath]
                survived.update({(fpath, mutNo):True for mutNo in mutNos})
            for k,flag in mutDeadAts.items():
                if flag is None: # not set yet, 
                    try:
                        _ = survived[k] # survived, do nothing
                    except KeyError: # not set, but also not among the survived
                        mutDeadAts[k] = commit_hash

        # set rename to max 
        git_utils.setDiffMergeRenameToMax(workdir)

        # start  
        targetFilesTracking = {targetFile:[targetFile] for targetFile in mutLRPairInfo_pfile.keys()}
        ## formulate mutLRPairInfos per file
        mutConts_pfile = MutantPropagator.formulate_mutLRPairs(mutLRPairInfo_pfile)
        survivedMuts = {fpath:list(mutConts_pfile[fpath].keys()) for fpath in mutConts_pfile}
        revealedMuts = {targetFile:{} for targetFile in targetFilesTracking.keys()}
        mutDeadAts = {(fpath,_no):None \
                      for fpath in mutConts_pfile for _no in mutConts_pfile[fpath]}

        refactoringOccurred = {}
        cnt_nothing_to_inspect = 0
        cnt_CONT_COMPILATION_FAIL_PASS = 0 
        print ("Start...")
        #print (survivedMuts)
        for prevCommit, currCommit in tqdm(
            list(zip(targetCommits[:-1], targetCommits[1:]))
        ):
            ## check for early-end
            if sum([len(vs) for vs in survivedMuts.values()]) == 0: # all processsed
                print (f"All processed at {currCommit}")
                break 
            ## check whether this commit is our target: compared with the prev commit of targetCommit
            #print (f"at {prevCommit}, {currCommit}")
            diffFiles, deletedFiles, renamedFiles = git_utils.getDiffFiles(
                workdir, 
                currCommit, 
                prevCommit
            ) # pathes are from prevCommit as they will be compared with the files from the last commi
            #print ('Deleted',deletedFiles)
            filesToInspect = getFilesToInspect(targetFilesTracking, diffFiles) # k=prev_path, v=original path
            updateFileTrackingByDelete(targetFilesTracking, deletedFiles) 
            # mutants
            updateMutContByDelete(mutConts_pfile, deletedFiles)
            updateSurvMutByDelete(survivedMuts, deletedFiles)
            if len(filesToInspect) == 0: # nothing to look
                #print ("Nothing to inspect")
                #print ("\t", sum([len(vs) for vs in survivedMuts.values()]))
                cnt_nothing_to_inspect += 1
                # still need to check renaming 
                updateFileTrackingByRename(targetFilesTracking, renamedFiles)
                continue # -> original code
            
            print ("yet-to-process: currently survived", sum([len(vs) for vs in survivedMuts.values()]))
            (
                dstDiffMapJSONFile, 
                dstRefactorJSONFile
            ) = MutantPropagator.get_DefaultDiffMapAndRefactorFiles(dest, currCommit[:8])
            # now start
            java_utils.changeJavaVer(17) # for RMINER 
            run_status = MutantPropagator.single_run(
                list(filesToInspect.keys()), # prev path, meaning if this is renamed, then it will be 
                workdir, currCommit, 
                dstDiffMapJSONFile, dstRefactorJSONFile
            )
            if not run_status: # stop
                print (f"Failed at {currCommit} and therby stop")
                break 
            #print (f"At commit {currCommit}", workdir)
            diffMaps = MutantPropagator.readDiffFile(
                dstDiffMapJSONFile, workdir, currCommit, prevCommit)
            if len(diffMaps) == 0: # for case where no meaningful changes (e.g., only comments chaged) & skipped, but will affect the position information 
                ret_positions = MutantPropagator.run_gumtree_v2(
                    workdir, currCommit, prevCommit, filesToInspect, renamedFiles
                ) 
                MutantPropagator.updateMutPosByGumtree(ret_positions, mutConts_pfile, survivedMuts)
                updateFileTrackingByRename(targetFilesTracking, renamedFiles)
                continue # here, no need to process further
            refactorings = MutantPropagator.readRefactoringFile(dstRefactorJSONFile) # for later analysis ..?
            if refactorings.shape[0] > 0:
                refactoringOccurred[currCommit] = refactorings # for saving
            
            ## prepare the repository 
            git_utils.checkout(workdir, currCommit) 
            java_utils.changeJavaVer(8) # for testing
            if 'lang' in workdir.lower(): mvn_utils.preprocess_lang(workdir) 
            # test without any mutation 
            filesToTest = list(set([curr_fpath for _, curr_fpath in diffMaps.keys()])) 
            if _testClassPatterns is None: # -
                testClassPatterns = java_utils.getTestClassPats(filesToTest, commonGID) 
                test_class_pat = ",".join(list(testClassPatterns.values()))
            else:
                test_class_pat = _testClassPatterns
            
            skipCompile = False
            try:
                if ant_or_mvn in ['ant_d4j', 'ant']:
                    # for ant, we need full path
                    test_classes_dir = os.path.abspath(os.path.join(workdir, 
                        ant_d4jbased_utils.export_TestClassesDir_d4jbased(
                            kwargs['d4j_home'], kwargs['project'], workdir)))
                    _, _ = ant_d4jbased_utils.test_compile(kwargs['d4j_home'], kwargs['project'], workdir)
                    test_class_pat = java_utils.getFullTestClasses(test_class_pat, test_classes_dir)
            except Exception as e:
                print (e)
                cnt_CONT_COMPILATION_FAIL_PASS += 1
                skipTesting = True
                skipCompile = True
                timeout = None
                testClassPatterns = None
                failed_testcases = None
                error_testcases = None
            #
            if not skipCompile: # in some cases, failed to compile due to the failure to find the src diretory, etc 
                t1 = time.time()
                (src_compiled, test_compiled, 
                failed_testcases, error_testcases) = pitMutationTool.compile_and_run_test(
                    workdir, 
                    test_class_pat, 
                    with_test_compile = False, 
                    ant_or_mvn = ant_or_mvn,
                    **kwargs # d4j_home, project
                )
                t2 = time.time()
                timeout = 2 * (t2 - t1) # if takes more than twice the exeuction time of the original, timeout
                #
                exec_status = pitMutationTool.checkWhetherPass(src_compiled, test_compiled) 
                if exec_status == 0: # ok, nothing-to-do
                    skipTesting = False 
                    cnt_CONT_COMPILATION_FAIL_PASS = 0 # 
                elif exec_status == 1: # for this case, we will not reflect the test results
                    # e.g., the case where, nevethelss, 
                    skipTesting = True
                    cnt_CONT_COMPILATION_FAIL_PASS = 0 
                else: # exec_status = 2
                    cnt_CONT_COMPILATION_FAIL_PASS += 1
                    if cnt_CONT_COMPILATION_FAIL_PASS >= MutantPropagator.MAX_CONT_COMPILATION_FAIL_PASS:
                        break # Stop goinng further
                    else:
                        skipTesting = True 

            _revealedMuts, noMutRemains = MutantPropagator.applyMutations(
                workdir, 
                currCommit, 
                mutConts_pfile,
                diffMaps,
                targetFilesTracking, 
                test_class_pat, 
                (failed_testcases, error_testcases), # -> on the current commit 
                timeout, 
                dest,
                skipTesting = skipTesting,
                ant_or_mvn = ant_or_mvn, 
                **kwargs 
            ) # currently not updated for diffMaps 
            if _revealedMuts is None: # in this case, noMutRemains will also None 
                print (f"Failed at {currCommit} due to test compilation error and therby stop")
                break 
            elif len(_revealedMuts) > 0:
                # update to the returned variable & save
                for orgFpath, pmutInfo in _revealedMuts.items(): 
                    revealedMuts[orgFpath].update(pmutInfo) # update this one
                revealInfoFile = os.path.join(dest, f"revealedAt.{currCommit[:8]}.json")
                print (f"At commit {currCommit[:8]}, {len(_revealedMuts)} revealed")
                with open(revealInfoFile, 'w') as f:
                    import json 
                    f.write(json.dumps(_revealedMuts)) 

            updateMutContNoRemain(mutConts_pfile, noMutRemains)
            ## get the surviving mutants
            MutantPropagator.updateSurvivedMutants(survivedMuts, _revealedMuts, noMutRemains)
            mutConts_pfile = MutantPropagator.getMutContOfSurvived(mutConts_pfile, survivedMuts)
            # update mutDeadAts 
            updateMutDeatAts(currCommit, mutDeadAts, survivedMuts)
            # update the files to track 
            fileUpdates = {} # local file update
            for prev_fpath, curr_fpath in diffMaps.keys(): # prev_fpath -> one d, the filesToInspect
                try: # k = (prev_fpath, org_fpath)
                    fileUpdates[(prev_fpath, filesToInspect[prev_fpath])].append(curr_fpath)
                except KeyError:
                    fileUpdates[(prev_fpath, filesToInspect[prev_fpath])] = [curr_fpath]
            # extension & the files where all mutants were revealed will also be removed
            updateFileTracking(targetFilesTracking, fileUpdates, survivedMuts) 
            for orgFpath in mutConts_pfile:
                for mutNo in mutConts_pfile[orgFpath]:
                    mutConts_pfile[orgFpath][mutNo].initTempAppliedAts() 
            print ("Currently survived", sum([len(vs) for vs in survivedMuts.values()]))
        print (
            f"Out of {len(targetCommits)}, nothings to inspect in {cnt_nothing_to_inspect} commits"
        )
        return revealedMuts, survivedMuts, mutDeadAts, refactoringOccurred