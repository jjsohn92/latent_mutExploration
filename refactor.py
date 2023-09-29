"""
"""
import os, sys
from typing import List, Dict, Tuple, Set, Union
import pandas as pd 
from utils import git_utils, file_utils, mvn_utils, java_utils, ant_d4jbased_utils, ant_mvn_utils
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

def checkIn(target_type:str, to_compare_type:str, mutation_type:str) -> bool: #, toCheckList:List[str]):
    # process both case of srcML vs GUMTREE and GUMTREE vs GUMTREE
    if mutation_type == 'pit':
        #print ('heerere!', target_type, to_compare_type, target_type == to_compare_type)
        return target_type == to_compare_type
    elif mutation_type == 'major':
        import re 
        toCheckList = MATCHING_TYPES[to_compare_type]
        if toCheckList is None: return False
        for toCheck in toCheckList:
            if bool(re.search(toCheck, target_type)):
                return True 
        return False
    else:
        print (f"Invalid mutation type: {mutation_type}")
        assert False 

class RMinerProcesser():
    RMINER_HOME = os.path.join(
        os.getenv("RMINER_HOME"), 
        "build/distributions/RefactoringMiner-2.4.0/bin"
    )
    MAX_CONT_COMPILATION_FAIL_PASS = 10 # the maximum number of compilation failure

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_DefaultDiffMapAndRefactorFiles(dest:str, fileKey:str) -> Tuple[str,str]:
        dstDiffMapJSONFile = os.path.join(dest, f"diffMap_{fileKey}.json")
        dstRefactorJSONFile = os.path.join(dest, f"refactorings_{fileKey}.json")
        return dstDiffMapJSONFile, dstRefactorJSONFile

    @staticmethod
    def readDiffFile(dstDiffMapJSONFile:str) -> Dict[Tuple, pd.DataFrame]:
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
            # drop those where 
            #data = data.loc[~data.prev_content.isna()] 
            formatted[(prev_fpath, curr_fpath)] = data
        return formatted 

    @staticmethod
    def readDiffFile_new(
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
            ### new end 
            # drop those where 
            #data = data.loc[~data.prev_content.isna()] 
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
        #commit = refactorings['sha1']
        refactoring_data = {'idx':[], 'filePath':[], 'isLeft':[], 'start':[], 'end':[], 'codeElementType':[]}
        for idx, refactoring in enumerate(refactorings['refactorings']): # 
            # 'type': the type of refactoring
            # 'description': a detailed description of the refactoring
            # leftSideLocations': from the previous : List of CodeRange (Dict)
            # 'rightSideLocations: the current : List of CodeRange (Dict)
            for ls in refactoring['leftSideLocations']:
                #'filePath', 'startLine', 'endLine', 'startColumn', 'endColumn', 'codeElementType', 'description', 'codeElement'
                refactoring_data['idx'].append(idx)
                refactoring_data['isLeft'].append(1)
                refactoring_data['filePath'].append(ls['filePath'])
                refactoring_data['start'].append((ls['startLine'], ls['startColumn']))
                refactoring_data['end'].append((ls['endLine'], ls['endColumn']))
                refactoring_data['codeElementType'].append(ls['codeElementType'])
            for rs in refactoring['rightSideLocations']:
                #'filePath', 'startLine', 'endLine', 'startColumn', 'endColumn', 'codeElementType', 'description', 'codeElement'
                refactoring_data['idx'].append(idx)
                refactoring_data['isLeft'].append(0)
                refactoring_data['filePath'].append(rs['filePath'])
                refactoring_data['start'].append((rs['startLine'], rs['startColumn']))
                refactoring_data['end'].append((rs['endLine'], rs['endColumn']))
                refactoring_data['codeElementType'].append(rs['codeElementType'])
        return pd.DataFrame(refactoring_data)

    @staticmethod 
    def run_gumtree(
        workdir:str, currCommit:str, prevCommit:str, files:Dict[str,str],  #List[str]
        renamedFiles:Dict[str,str]
    ): #-> Dict[Tuple[int,int],Tuple[int,int]]: # ...
        # workdir -> already checkout to currCommit (since this is called after single_run)
        # files -> from previous commit
        import utils.gumtree as gumtree 
        from subprocess import CalledProcessError
        _currfiles = None
        ret_positions = {}
        for prev_file, orgFpath in files.items():
            # get previous file content 
            prevContent = git_utils.show_file(prevCommit, prev_file, workdir)
            # get current file content
            new_file = prev_file if prev_file not in renamedFiles.keys() else renamedFiles[prev_file]
            try:
                currContent = git_utils.show_file(currCommit, new_file, workdir)
                #print('here!', prev_file, new_file)
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
                        new_file = _currfile 
                        break 
                assert new_file is not None, basename 
                #full_fpath = os.path.join(workdir, new_file)
                #currContent = file_utils.readFile(full_fpath) 
                currContent = git_utils.show_file(currCommit, new_file, workdir)
            # run gumtree to get the parsing results
            tree = gumtree.processABFile_gumtree(prevContent, currContent)
            mappedPositiosn = gumtree.mapPositions(tree) # prevContent -> will be the key 
            ret_positions[(orgFpath, prev_file)] = (new_file, mappedPositiosn)
        return ret_positions


    @staticmethod 
    def run_gumtree_v2(
        workdir:str, currCommit:str, prevCommit:str, files:Dict[str,str],  
        renamedFiles:Dict[str,List[str]]
    ): #-> Dict[Tuple[int,int],Tuple[int,int]]: # ...
        # workdir -> already checkout to currCommit (since this is called after single_run)
        # files -> from previous commit
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
                            #print ("\t found by matching basename", new_file, basename, _basename)
                            break 
                    assert new_file is not None, basename 
                    #full_fpath = os.path.join(workdir, new_file)
                    #currContent = file_utils.readFile(full_fpath) 
                    currContent = git_utils.show_file(currCommit, new_file, workdir)
                # run gumtree to get the parsing results
                tree = gumtree.processABFile_gumtree(prevContent, currContent)
                mappedPositiosn = gumtree.mapPositions(tree) # prevContent -> will be the key 
                ret_positions[(orgFpath, prev_file)].append((new_file, mappedPositiosn))
        #
        return ret_positions
    
    @staticmethod
    def updateMutPosByGumtree(
        ret_positions:Dict[str,Dict],
        mutConts_pfile:Dict[str, Dict], 
        survivedMuts:Dict[str,List[int]]
    ):
        to_removed = []
        for orgFpath, mutConts in mutConts_pfile.items():
            for mutNo in mutConts.keys():
                _new_appliedAts = {}
                for prevFpath, appliedAtPFile in mutConts[mutNo].appliedAts.items():
                    newFPath, mappedPositiosn = ret_positions[(orgFpath, prevFpath)]
                    for toplt_txt, toplt_loc  in appliedAtPFile:
                        try:
                            new_loc = mappedPositiosn[toplt_loc] # but, this is just one 
                        except KeyError as e:
                            print ("key error", e)
                            print ("\t", mutNo, mutConts[mutNo].appliedAts)
                            #del mutConts[mutNo] # no-longer valid, so drop from mutConts_pfile
                            continue # b/c failed to handle 
                        try:
                            _new_appliedAts[newFPath].append([toplt_txt, new_loc])
                        except KeyError:
                            _new_appliedAts[newFPath] = [[toplt_txt, new_loc]]
                if len(_new_appliedAts) > 0:
                    mutConts_pfile[orgFpath][mutNo].appliedAts = _new_appliedAts # update -> if all failed to match (e.g.,due to the handling of gumtree, then this can be empty)
                else:
                    to_removed.append([orgFpath, mutNo])
        # clean-up
        for orgFpath, mutNo in to_removed:
            del mutConts_pfile[orgFpath][mutNo]
            survivedMuts[orgFpath].remove(mutNo)

    @staticmethod
    def updateMutPosByGumtree_v2(
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
                                    print ("key error", e)
                                    print ("\t", mutNo, mutConts[mutNo].appliedAts)
                                    #del mutConts[mutNo] # no-longer valid, so drop from mutConts_pfile # can be parsing diff btwn srcml and gutree, and also due to the element no longer exist (other_pos -> don't have)
                                    continue # b/c failed to handle 
                                try:
                                    _new_appliedAts[newFPath].append([toplt_txt, new_loc])
                                except KeyError:
                                    _new_appliedAts[newFPath] = [[toplt_txt, new_loc]]
                    else: # not our target 
                        _new_appliedAts[prevFpath] = appliedAtPFile 
                #if len(_new_appliedAts) > 0:
                mutConts_pfile[orgFpath][mutNo].appliedAts = _new_appliedAts # update -> if all failed to match (e.g.,due to the handling of gumtree, then this can be empty)
                #else:
                #    to_removed.append([orgFpath, mutNo])
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
        rminer_binary = os.path.join(RMinerProcesser.RMINER_HOME, "RefactoringMiner")
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
    def run_simple(
        targetFiles:List[str], workdir:str, 
        targetCommits:List[str], dest:str
    ):
        """
        actually we need to track down to dynamically update a list of files, 
        if we want to save the space (otherwise, stored a lot)
        """
        from tqdm import tqdm 
        # inspect commits in targetCommits 
        failed = []
        for targetCommit in tqdm(targetCommits):
            dstDiffMapJSONFile, dstRefactorJSONFile = RMinerProcesser.get_DefaultDiffMapAndRefactorFiles(dest, targetCommit[:8])
            run_status = RMinerProcesser.single_run(
                targetFiles, workdir, targetCommit, 
                dstDiffMapJSONFile, dstRefactorJSONFile
            )
            if not run_status: 
                print (f"\tFailed to process {targetCommit}")
                failed.append(targetCommit[:8])
                continue # b/c here, each run is indepedent
        print (f"Finished: out of {len(targetCommits)}, failed to process {len(failed)} of them")
        print ("\t:" + ", ".join(failed))

    @staticmethod
    def formulate_mutLRPairs(
        mutLRPairInfo_pfile:Dict[str, Dict], use_sdk:bool = False, 
        which_mutant:str = 'majoir'
    ) -> Dict[str, Dict[int, Union[MutantCont, PitMutantCont]]]:
        mutConts_pfile = {}
        for mutFpath, mutLRPairInfos in mutLRPairInfo_pfile.items():
            if len(mutLRPairInfos) == 0: # nothing:
                continue 
            mutConts_pfile[mutFpath] = {}
            for mutNo, mutLRPairInfo in mutLRPairInfos.items():
                if which_mutant == 'major':
                    mutCont_inst = MutantCont(
                        mutNo, 
                        mutLRPairInfo['left'], 
                        mutLRPairInfo['right'], 
                        mutLRPairInfo['mutOp'], 
                        mutLRPairInfo['targeted'], #  
                        mutFpath,
                        use_sdk = use_sdk
                    )
                else:
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

    # we can either go strictly or flexibly (e.g., by allowing xx% to mismatch)
    ### -> here, for PIT, ... tag (toplt_+tu)
    # diffMaps -> from the gumtree output by RMiner
    @staticmethod 
    def match(diffMaps:pd.DataFrame, toplt_loc:Tuple, toplt_type:str, mutation_type:str) -> Tuple[int, pd.Series]:
        """
         # -> should be about the tag, instead of content, which may not have
        only look at the poistion (can be None, when it is deleted)
        return (an index to mapping, new location)
        """
        foundMap = diffMaps.loc[diffMaps.prev_pos == toplt_loc]
        #with open("temp.pkl", 'wb') as f:
        #    import pickle
        #    pickle.dump(foundMap, f)
        #print ("comparing", toplt_loc)
        #print ("foundMap", foundMap, len(foundMap))
        if len(foundMap) == 1: # 
            ret = foundMap.iloc[0]
            return (foundMap.index.values[0], ret)
        elif len(foundMap) > 0: # more than one mapping, then we further check 
            #_foundMap = foundMap.loc[foundMap.prev_content == toplt_txt]
            _foundMap = foundMap.loc[
                #foundMap.prev_type.apply(lambda v:checkIn(v, MATCHING_TYPES[toplt_type]))
                foundMap.prev_type.apply(lambda v:checkIn(v, toplt_type, mutation_type))
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
            # -> can be either due to the location info mismatch btwn srcml and javaparser
            # or .. really the issue with the range or parsing algorithm insides 
            # or the issue with mutation part -> argument_list 
            # OR deleted element -> no mapping exist in this case
            #print (f"Shouldn't be happened: remain: {len(_foundMap)}")
            print ('no fond map...', toplt_loc)
            return (None, None)  # e.g., deleted 
    
    #@staticmethod
    #### PIT -> THE SAME
    #def checkMatching(
        #foundMapping:pd.Series, 
        #new_toplt_txt:str, 
        #contentToCompare:str, # from mutant
        #tagToCompare:str
    #) -> bool:
        #"""
        #check the found matching
        #"""
        #mappedEType = foundMapping.prev_type # from diffMaps, which is based on Gumtree
        #mappedEContent = new_toplt_txt #foundMapping.prev_content # -> for some cases, this may not work 
        #if tagToCompare == 'name': # in matcher, only allowed when mutOp = COR & IDENTIFIER...
            ## the case of COR (<IDENTIFIER(boolean)>)
            ## for variable renaming, this won't work, but it has to work 
            #return True
        #else:
            #if checkIn(mappedEType, MATCHING_TYPES[tagToCompare]):
                #if tagToCompare == 'call': 
                    #return True 
                #else: # operator, literal, (argument_list -> always None) -> assume the mutation context changed -> so no-longer valid
                    #return contentToCompare == mappedEContent 
            #else: # nothing, wrong type -> likely some-mismatch or the changes in the context => false
                #return False
    @staticmethod
    def checkMatching(
        foundMapping:pd.Series, 
        new_toplt_txt:str, 
        contentToCompare:str, # from mutant
        typeToCompare:str, 
        mutation_type:str, 
        #mutCont:Union[PitMutantCont, MutantCont]
        mutOp:Union[str, Tuple[str,str]]
    ) -> bool:
        """
        check the found matching
        typeToCompare -> prev type (since we do not allow type change (at leasr for pit), it will be the original type)
        """
        mappedEType = foundMapping.prev_type # from diffMaps, which is based on Gumtree
        mappedEContent = new_toplt_txt #foundMapping.prev_content # -> for some cases, this may not work 
        is_targeted_mut_type = checkIn(mappedEType, typeToCompare, mutation_type)
        #mutOp = mutCont.mutOp
        if mutation_type == 'major':
            if typeToCompare == 'name': # in matcher, only allowed when mutOp = COR & IDENTIFIER...
                # the case of COR (<IDENTIFIER(boolean)>)
                # for variable renaming, this won't work, but it has to work 
                return True
            else:
                if is_targeted_mut_type:
                    if typeToCompare == 'call': 
                        return True 
                    else: # operator, literal, (argument_list -> always None) -> assume the mutation context changed -> so no-longer valid
                        return contentToCompare == mappedEContent 
                else: # nothing, wrong type -> likely some-mismatch or the changes in the context => false
                    return False
        else:
            #if is_targeted_mut_type:
            #    if typeToCompare not in ['MethodInvocation', '']:
            #        return contentToCompare == mappedEContent 
            #    else:
            #        return True # e.g., MethodInvocation for voidMethodCall
            #else:
            #    return False  
            if is_targeted_mut_type:
                #print ("\t", contentToCompare == mappedEContent, contentToCompare, mappedEContent, typeToCompare, mutOp[0])
                if mutOp[0] in ['MATH', 'CONDITIONALS_BOUNDARY', 'INCREMENTS', 'INVERT_NEGS', 'NEGATE_CONDITIONALS', 
                                'OBBN1', 
                                'INLINE_CONSTS', 
                                'CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6', 
                                'ROR1', 'ROR2', 'ROR3', 'ROR4', 'ROR5', 
                                'AOR1', 'AOR2', 'AOR3', 'AOR4']:
                    #if mutOp[0] in ['INLINE_CONSTS', 'CRCR1', 'CRCR2', 'CRCR3', 'CRCR4', 'CRCR5', 'CRCR6']:
                    #    ig
                    #else: 
                    #if mutCont.neg_prefix is None:
                    return contentToCompare == mappedEContent  
                    #else:
                    #    return contentToCompare == mappedEContent  
                elif mutOp[0] in ['EMPTY_RETURNS', 'FALSE_RETURNS', 'TRUE_RETURNS', 'NULL_RETURNS', 'PRIMITIVE_RETURNS', 
                                  'VOID_METHOD_CALLS', 'CONSTRUCTOR_CALLS', 'REMOVE_INCREMENTS', 
                                  'AOD_1', 'AOD_2', 'OBBN2', 'OBBN3']: 
                    #if typeToCompare.endswith("Literal"): 
                    #    return contentToCompare == mappedEContent 
                    #else: # the case of variable (e.g., simpleName)
                    #   return True 
                    if typeToCompare.endswith("Name"): # if it is just variable name change, skip it 
                        return True 
                    else: # otherwise, check it (actually in most cases, ) => this is to prevent the effect of overwritting by later-applied mutants
                        return contentToCompare == mappedEContent 
                else:
                    print (f"Unsupported mutation operator: {mutOp[0]}, {mutOp[1]}")
                    assert False
            else:
                return False # slightly treaky ... e.g., for 
            
    @staticmethod
    def updateMutPos(
        targetFileContent:str,
        diffMaps:pd.DataFrame, 
        newPath, prevPath, 
        mutCont:Union[MutantCont, PitMutantCont], 
        which_mutant:str
    ) -> None:
        """
        Per-mutant
        diffMaps -> for a single file
        """
        #no_longer_valid = [] # e.g., deleted in the newPath
        # toplt_txt = mutant text, toplt_loc = mutant injection location info (start_loc, end_loc)
        for i, (toplt_txt, toplt_loc) in enumerate(mutCont.appliedAts[prevPath]): # here... all mutants will be considered regardless of whether they were changed or not, as diffMap itself cover the entire file and appliedAt 
            # toplt_type -> the type of mutated code element  => if the type changed, then there will be no-match
            if isinstance(mutCont, MutantCont):
                toplt_type = mutCont.target_node.tag # from srcML
            elif isinstance(mutCont, PitMutantCont):
                toplt_type = mutCont.target_node.attrib['type'] # from Gumtree 
            else:
                print (f"Invalid mutant containter type: {type(mutCont)}")
                assert False
            # matching ....
            matchIdx, matchedMap = RMinerProcesser.match(diffMaps, toplt_loc, toplt_type, which_mutant) 
            if matchIdx is None: # the case, when the target was deleted (?) => CURRENTLY, then all the method invocation will be in trouble ... 
                # -> no mapping (to the current version): even if it is mapped in diffMaps, 
                # can't find the match due to the change of .. content 
                #return False # this var of the mut is no longer valid -> remove it from appliedAts[prevPath]
                #no_longer_valid.append(i)
                #print ("match index none", toplt_loc, toplt_txt)
                continue
            else: # valid then further check by comparing the text of mutated element: toplt_txt
                new_toplt_txt = targetFileContent[matchedMap['curr_pos'][0]:matchedMap['curr_pos'][1]]
                if not RMinerProcesser.checkMatching(
                    matchedMap, new_toplt_txt, toplt_txt, toplt_type, which_mutant, mutCont.mutOp, 
                ): # failed to map
                    # can be either some mismatch in parsing or matching OR the line changed, thereby the mutant no longer exist 
                    # -> for the latter case, 
                    #no_longer_valid.append(i) # e.g., content changed, no longer the same one
                    continue # skip this one 
                # update the position information: add to tempApplyAt for the later injection
                mutCont.addNewApplyAt(
                    prevPath, newPath, new_toplt_txt, matchedMap['curr_pos']
                ) # -> will be added to temp_diff
        # remove no-longer valid from mutCont.appliedAts 
        #mutCont.deleteNoLongerValids(prevPath, no_longer_valid) # delete no-longer-valid # -> tempApplyAt & update will handle this b/c

    ### this is actually the core of update
    # before this, checkout & is_junit should be done
    @staticmethod
    def applyMutations(
        workdir:str, 
        commit_hash:str, 
        mutConts_pfile:Dict[str, Dict], 
        which_mutant:str, 
        diffMaps:Dict[Tuple, pd.DataFrame], 
        targetFilesTracking:Dict[str, List[str]], 
        testClassPatterns:Union[Dict[str,str], str], 
        use_junit4:bool, 
        woMutFailed_or_Error_Tests:Tuple[Set[str], Set[str]], 
        timeout:int, 
        dest:str,
        #compile_test:bool = False,
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
        ## compile tests  -> Not needed as the test com
        #if (compile_test) and (not skipTesting): # if test-execution is skipped, no need to compile tests (compile-fase) -> for this case, shoudl be turn ton
            #test_output, test_error = mvn_utils.compile_tests(workdir)  
            #if test_error is not None:  
                #print (test_output); print ("test compilation failed")   
                #return (None, None) ##  
        noMutRemains = {}
        revealedMuts = {} # contain a least of revealed mutants
        ## the line below will iterate over the entire files that we keep track of (despite of a certain file not changed)
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
                for targetKey in targetKeys: # prevFpath(dvdgth)-newMath(toMutated) -> meaning, yes, it is included
                    toMutatedFPath = targetKey[1] # targetkey[0] == dvgdFpath, targetKey[1] = current path (at workdir) of dvgdFpath
                    targetFileContent = file_utils.readFile(os.path.join(workdir, toMutatedFPath)) #  everything here (as later we have to test)
                    for mutNo, mutCont in mutConts.items(): # -> SHOULD BE MODIFIED -> mutLRPair as an instance of a mutant class
                        #if mutNo != 1291: continue
                        #print ("==", mutCont.appliedAts.keys(), commit_hash)
                        # check whether mutNo is already covered
                        if mutNo in revealedMuts[orgFpath].keys(): # since mutConts (= the mutants at orgFPath) can be applied to multiple files due to refactoring between files or withine the file 
                            continue # sicne already revealed by other dvgdFpath and toMutatedFPath pair
                        print ("IN MUT NO: ", mutNo) 
                        #print (mutCont.appliedAts)
                        # new mapping info included in mutCont.tempApplyAts and no-valid removed
                        # tempApplyAts udpated in mutCont -> for the following injectAndTest (preparation)
                        RMinerProcesser.updateMutPos(
                            targetFileContent, 
                            diffMaps[targetKey], 
                            toMutatedFPath, dvgdFpath, 
                            mutCont, 
                            which_mutant
                        ) # -> CURRENTLY, nothing is added to mutCont.tempApplyAts for ArrayUtils.java  -> Checkwhehther this is normal
                        #print (mutCont.appliedAts)
                        #print (mutCont.tempApplyAts)
                        #continue ##        
                        # per mutant, but here, mutlipe variations of mutants (by propagation) can be tested (self.appliedAts[orgFpath] ...)
                        if not skipTesting:
                            t1 = time.time()
                            revealedMutInfo = mutCont.injectAndTest(
                                workdir, # checkout dir = repo  
                                targetFileContent, # checkout & get
                                dvgdFpath, 
                                toMutatedFPath,
                                use_junit4, 
                                testClassPatterns[toMutatedFPath] if isinstance(testClassPatterns, Dict) else testClassPatterns, # toMutatedFPath in both tracking and diffmap
                                woMutFailed_or_Error_Tests,
                                timeout = timeout,
                                with_test_compile = False, 
                                ## new
                                ant_or_mvn = ant_or_mvn, 
                                which_mutant = which_mutant,
                                **kwargs
                            ) #-> here, in tempApplyAts will contain 
                            #t2 = time.time()
                            #print ('Time for testing mutants', t2 - t1)
                            if revealedMutInfo is None: # meaning, unexpected error occurs, and thereby should stop
                                # -> but, here, this is just for one muntant, i.e., mutCont 
                                # so, for the other mutants, it is fine to go with (just record this as invalid and delete)
                                # currently, here, only the case is when test compilation failed suddenely eventhough nothing changed, for that case, we stop running
                                return (None, None)
                            # mutation application -> the target doesn't chagned
                            elif len(revealedMutInfo) > 0: # this mutant has been revealed
                                revealedMuts[orgFpath][mutNo] = revealedMutInfo # paired with the prev if mutNo in .. -> ewill excldue alredy broken mutants
                                #### if this mutant (mutNo) is revealed, no need to further check for its
                            # BASED ON THE results (here, one of the variation of mutCont -> by deleting from appliedAts)
                        else:
                            continue # do nothing 
                        # .. acutally if killed here, then this mutant will be completely out of concern AS IT WAS REVELASDD 
                ## upto here, tempApplyAts contains all the variations of valid mut per mut inject ed in orgFpat
                # Now update appliedAts and save it 
                if len(targetKeys) > 0: # targetKeys: new fpaths for dvgdFpath
                    for mutNo in mutConts: # mutConts -> for
                        # dvgdFpath -> may not be modified here
                        ## -> here, if dvgdFpath was not changed at all, meaning no 
                        mutConts[mutNo].updateAppliedAts(dvgdFpath) # update dvgdFpath e of appliedAts (invalid were already handled by deletedNoLogerValids in injectAndTest)
            if len(revealedMuts[orgFpath]) == 0: 
                del revealedMuts[orgFpath] # b/c nothing to return 
            
            # due to this, unless the case of unexpected return (None, None), noMutRemains will always have orgFpath
            noMutRemains[orgFpath] = []
            for mutNo in mutConts: # check for mutants in orgFpath
                if mutConts[mutNo].isEmpty(): # no appliedAt remains 
                    noMutRemains[orgFpath].append(mutNo)
        # saving 
        #for i,(orgFpath, mutConts) in enumerate(list(mutConts_pfile.items())):
        #    _dest = os.path.join(dest, str(i))
        #    os.makedirs(_dest, exist_ok=True)
        #    for mutNo, mutCont in mutConts.items():
        #        mutCont.saveAppliedAtInfo(_dest, commit_hash) # saving 
        _ = RMinerProcesser.saveAppliedAtInfoOfMuts(mutConts_pfile, commit_hash, dest)
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
        mutConts_pfile:Dict[str, Dict[int, Union[PitMutantCont, MutantCont]]], 
        commit_hash:str, 
        dest:str, 
    ) -> str:
        # pickle 
        #import gzip, pickle 
        combinedToSave = {}
        for orgFpath, mutConts in mutConts_pfile.items():
            #_dest = os.path.join(dest, str(i))
            #os.makedirs(_dest, exist_ok=True)
            combinedToSave[orgFpath] = {}
            for mutNo, mutCont in mutConts.items():
                #mutCont.saveAppliedAtInfo(_dest, commit_hash) # saving  
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
        #targetFiles:List[str], 
        commonGID:str, 
        targetCommits:List[str], 
        dest:str, 
        use_sdk:bool = False, 
        _testClassPatterns:str = None, 
        ant_or_mvn:str = 'mvn',
        which_mutant:str = 'major', 
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
        from mutants import mutationTool, pitMutationTool

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
                    # if prev_fpath not in new_fpaths 
                    if prev_fpath not in new_fpaths: # this can be risky .. in case where, simply copy and past occurs (but .. then it is not a refactoring ... so, maybe outside of our scope)
                        trackDict[org_fpath].remove(prev_fpath) 
                    #else: # pre
                    #    trackDict[org_fpath].extend(
                    #        [fpath for fpath in new_fpaths if fpath != prev_fpath]
                    #    ) # if alredy has, skip 
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
            # by rename
            for orgFpath, tracked in trackDict.items():
                _tracked = []
                for afile in tracked:
                    try:
                        new_afiles = renamedFiles[afile]
                    except KeyError:
                        new_afiles = [afile] 
                    #_tracked.append(new_afile)
                    _tracked.extend(new_afiles)
                trackDict[orgFpath] = _tracked 

        def updateMutContByDelete(
            mutConts_pfile:Dict[str, Dict],
            deletedFiles:List[str] 
        ):
            # here, deletion will be on MutantCont.appliedAts, 
            # as the pathes in deletedFiles are previous ones and it is before updateMutPos
            for mutConts in mutConts_pfile.values():
                for mutCont in mutConts.values(): # here, mutCont not yet-updated for currCommit 
                    mutCont.deleteNoLongerExistByDelFiles(deletedFiles) # delete from self.appliedAts 
        
        def updateSurvMutByDelete(
            survivedMuts:Dict[str,List[int]], 
            deletedFiles:List[str]
        ):
            # though the key of survivedMuts is original path, 
            # doesn't matter here, as if org_path is one of deleted fiels, then it is deleted 
            # if the path of the original,  
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
        #targetFilesTracking = {targetFile:[targetFile] for targetFile in targetFiles}
        targetFilesTracking = {targetFile:[targetFile] for targetFile in mutLRPairInfo_pfile.keys()}
        ## formulate mutLRPairInfos per file
        mutConts_pfile = RMinerProcesser.formulate_mutLRPairs(mutLRPairInfo_pfile, use_sdk = use_sdk, which_mutant = which_mutant)
        survivedMuts = {fpath:list(mutConts_pfile[fpath].keys()) for fpath in mutConts_pfile}
        revealedMuts = {targetFile:{} for targetFile in targetFilesTracking.keys()}
        #mutDeadAts = {fpath:{_no:None for _no in mutConts_pfile[fpath]} for fpath in mutConts_pfile}
        mutDeadAts = {(fpath,_no):None \
                      for fpath in mutConts_pfile for _no in mutConts_pfile[fpath]}

        refactoringOccurred = {}
        cnt_nothing_to_inspect = 0
        cnt_CONT_COMPILATION_FAIL_PASS = 0 
        #end = False
        print ("Start...")
        print (survivedMuts)
        for prevCommit, currCommit in tqdm(
            list(zip(targetCommits[:-1], targetCommits[1:]))
        ):
            ## check for early-end
            if sum([len(vs) for vs in survivedMuts.values()]) == 0: # all processsed
                print (f"All processed at {currCommit}")
                break 
            ## check whether this commit is our target: compared with the prev commit of targetCommit
            print (f"at {prevCommit}, {currCommit}")
            diffFiles, deletedFiles, renamedFiles = git_utils.getDiffFiles(
                workdir, 
                currCommit, 
                prevCommit
            ) # pathes are from prevCommit as they will be compared with the files from the last commi
            print ('Deleted',deletedFiles)
            # handle diffs
            filesToInspect = getFilesToInspect(targetFilesTracking, diffFiles) # k=prev_path, v=original path
            # handle deletion 
            ## exclude mutants that are in the deleted files: since one file can have more than one tracked, the file is deleted from the tracked
            # files
            updateFileTrackingByDelete(targetFilesTracking, deletedFiles) 
            # mutants
            updateMutContByDelete(mutConts_pfile, deletedFiles)
            updateSurvMutByDelete(survivedMuts, deletedFiles)
            ##
            if len(filesToInspect) == 0: # nothing to look
                print ("Nothing to inspect")
                print ("\t", sum([len(vs) for vs in survivedMuts.values()]))
                cnt_nothing_to_inspect += 1
                # still need to check renaming 
                updateFileTrackingByRename(targetFilesTracking, renamedFiles)
                continue # -> original code
            
            print ("yet-to-process: currently survived", sum([len(vs) for vs in survivedMuts.values()]))
            (
                dstDiffMapJSONFile, 
                dstRefactorJSONFile
            ) = RMinerProcesser.get_DefaultDiffMapAndRefactorFiles(dest, currCommit[:8])
            # now start
            java_utils.changeJavaVer(17, use_sdk = use_sdk) # for RMINER 
            run_status = RMinerProcesser.single_run(
                list(filesToInspect.keys()), # prev path, meaning if this is renamed, then it will be 
                workdir, currCommit, 
                dstDiffMapJSONFile, dstRefactorJSONFile
            )
            if not run_status: # stop
                print (f"Failed at {currCommit} and therby stop")
                break 
            
            print (f"At commit {currCommit}", workdir)
            #diffMaps = RMinerProcesser.readDiffFile(dstDiffMapJSONFile) # here, only filesToInpsect will be
            diffMaps = RMinerProcesser.readDiffFile_new(
                dstDiffMapJSONFile, workdir, currCommit, prevCommit)
            if len(diffMaps) == 0: # for case where no meaningful changes (e.g., only comments chaged), and thereby skipped, but will affect the position information 
                print ("files to inspect", filesToInspect, _testClassPatterns)
                ret_positions = RMinerProcesser.run_gumtree_v2(
                    workdir, currCommit, prevCommit, filesToInspect, renamedFiles
                ) # ret_positions -> only for those in filesToInspect
                # here, we only need to update the position, as there was no content changes 
                #print (ret_positions.keys())
                RMinerProcesser.updateMutPosByGumtree_v2(ret_positions, mutConts_pfile, survivedMuts)
                updateFileTrackingByRename(targetFilesTracking, renamedFiles)
                print ("in gumtree part")
                print ("Currently survived", sum([len(vs) for vs in survivedMuts.values()]))
                continue # here, no need to process further
            ####
            ## temp -7/1
            # update the files to track
            #fileUpdates = {} # local file update
            #for prev_fpath, curr_fpath in diffMaps.keys(): # prev_fpath -> one d, the filesToInspect
            #    try:
            #        fileUpdates[(prev_fpath, filesToInspect[prev_fpath])].append(curr_fpath)
            #    except KeyError:
            #        fileUpdates[(prev_fpath, filesToInspect[prev_fpath])] = [curr_fpath]
            ## extension & the files where all mutants were revealed will also be removed
            #updateFileTracking(targetFilesTracking, fileUpdates, survivedMuts)
            #continue ## skip to the next
            ######
            ####
            refactorings = RMinerProcesser.readRefactoringFile(dstRefactorJSONFile) # for later analysis ..?
            if refactorings.shape[0] > 0:
                refactoringOccurred[currCommit] = refactorings # for saving
            
            # now the mutant map here 
            ## prepare the repository 
            print (workdir, currCommit)
            git_utils.checkout(workdir, currCommit) 
            java_utils.changeJavaVer(8, use_sdk = use_sdk) # for testing
            ## temporary ##
            if 'lang' in workdir.lower():
                mvn_utils.preprocess_lang(workdir) # move TypeUtilTests
            ###############
            junit_ver, use_junit4 = None, None # since we no longer use major
            #junit_ver = mvn_utils.get_junit_version(mvn_utils.get_pom_file(workdir))
            #print (f'Checkout done, now in {currCommit[:8]} & running with junit {junit_ver}')
            #use_junit4 = mvn_utils.is_eqOrHigherThanJunit4(junit_ver)
            # also testing without any mutation 
            print ("Testing...")
            # filesToTest -> for the current commit, meaning based on RMiner result
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
                    test_classes_dir = os.path.abspath(os.path.join(
                        workdir, 
                        ant_d4jbased_utils.export_TestClassesDir_d4jbased(kwargs['d4j_home'], kwargs['project'], workdir)
                    ))
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
                if False: #ant_or_mvn == 'mvn':
                    (src_compiled, test_compiled, 
                    failed_testcases, error_testcases) = mutationTool.compile_and_run_test(
                        workdir, 
                        test_class_pat, 
                        use_junit4 = use_junit4, 
                        with_log = False
                    )
                else:
                    ### -> here, need to (dynamically? or may not needed) write property file and include it in defects4j.build.ext.xml
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
                exec_status = mutationTool.checkWhetherPass(src_compiled, test_compiled) 
                if exec_status == 0: # ok, nothing-to-do
                    skipTesting = False 
                    cnt_CONT_COMPILATION_FAIL_PASS = 0 # 
                elif exec_status == 1: # for this case, we will not reflect the test results
                    # e.g., the case where, nevethelss, 
                    skipTesting = True
                    cnt_CONT_COMPILATION_FAIL_PASS = 0 
                else: # exec_status = 2
                    cnt_CONT_COMPILATION_FAIL_PASS += 1
                    if cnt_CONT_COMPILATION_FAIL_PASS >= RMinerProcesser.MAX_CONT_COMPILATION_FAIL_PASS:
                        break # Stop goinng further
                    else:
                        skipTesting = True 

            #assert src_compiled and test_compiled, f"{src_compiled}, {test_compiled}" # b/c here, normal
            # revealedMuts: 
            #   key = orginal mutated fpath,
            #   value = a dictionary (key = mutNo) of where (loc at commit) and revealed by which tests 
            # here, mutCont.isEmpty is called, so that,nothing left to track (len(appleidAts) = 0) is reflectd
            print ('Start applying', currCommit)
            #stop = False
            _revealedMuts, noMutRemains = RMinerProcesser.applyMutations(
                workdir, 
                currCommit, 
                mutConts_pfile, # need to filter out those alraedy revealed
                which_mutant,
                diffMaps, # here, only the file that was modified will be included
                targetFilesTracking, # likely contain the files that not in diffMaps targets
                test_class_pat, 
                use_junit4, 
                (failed_testcases, error_testcases), # -> on the current commit 
                timeout, 
                dest,
                #compile_test = False, 
                skipTesting = skipTesting,
                ## temporary 
                #skipTesting = True if currCommit != '07611165' else False,
                # new 
                ant_or_mvn = ant_or_mvn, 
                **kwargs # d4j_home, project
            ) # currently not updated for diffMaps 
            #updateMutContNoRemain(mutConts_pfile, noMutRemains)
            if _revealedMuts is None: # in this case, noMutRemains will also None 
                # test compilation failed # stop at this point 
                # errors during the injection 
                print (f"Failed at {currCommit} due to test compilation error and therby stop")
                break 
            elif len(_revealedMuts) > 0:
                # update to the returned variable & save
                for orgFpath, pmutInfo in _revealedMuts.items(): 
                    revealedMuts[orgFpath].update(pmutInfo) # update this one
                revealInfoFile = os.path.join(dest, f"revealedAt.{currCommit[:8]}.json")
                #print (_revealedMuts)
                print (f"At commit {currCommit[:8]}, {len(_revealedMuts)} revealed")
                with open(revealInfoFile, 'w') as f:
                    import json 
                    f.write(json.dumps(_revealedMuts)) # 
            updateMutContNoRemain(mutConts_pfile, noMutRemains)
            # if all mutants in the original file are revealed, exclude that file from the inspectation
            ## get the surviving mutants
            print ("survived", survivedMuts)
            RMinerProcesser.updateSurvivedMutants(survivedMuts, _revealedMuts, noMutRemains)
            print ("aft survived", survivedMuts)
            mutConts_pfile = RMinerProcesser.getMutContOfSurvived(mutConts_pfile, survivedMuts)
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
            print ("==", fileUpdates.keys())
            print ("++", targetFilesTracking.keys())
            print ("--", noMutRemains.keys())
            print ("--", survivedMuts.keys())
            #### -> here. .. fileUpdates may contain something that should not be ?
            updateFileTracking(targetFilesTracking, fileUpdates, survivedMuts) # -> here ... if all mutants were processed, then targetFilesTracking -> should be updated here by survivedMuts
            #updateFileTrackingByRename(targetFilesTracking, renamedFiles) # -> diffMaps wil handle this
            for orgFpath in mutConts_pfile:
                for mutNo in mutConts_pfile[orgFpath]:
                    mutConts_pfile[orgFpath][mutNo].initTempAppliedAts() # empty tempApplyAts (i.e., set to {}), just in case
            print ("Currently survived", sum([len(vs) for vs in survivedMuts.values()]))
        print (
            f"Out of {len(targetCommits)}, nothings to inspect in {cnt_nothing_to_inspect} commits"
        )
        return revealedMuts, survivedMuts, mutDeadAts, refactoringOccurred
    


    @staticmethod
    def run_seq_refactor(
        workdir:str, 
        targetFiles:List[str], 
        targetCommits:List[str], 
        dest:str, 
        use_sdk:bool = False, 
    ) -> Dict:
        """
        """
        from tqdm import tqdm 
        # inspect commits in targetCommits 
        def getFilesToInspect(
            trackDict:Dict, diffFiles:List[str], #renamedFiles:Dict[str,str]
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
        
        def updateFileTrackingByDelete(
            trackDict:Dict[str,List[str]],
            deletedFiles:List[str]
        ):
            deletedFiles = set(deletedFiles)
            for orgFpath, tracked in trackDict.items():
                tracked = [afile for afile in tracked if afile not in deletedFiles] # excluded deleted files from tracking
                trackDict[orgFpath] = tracked  # here, if nothing left to be tracked, then 

        def updateFileTrackingByRename(
            trackDict:Dict[str,List[str]],
            renamedFiles:Dict[str,List[str]]  
        ):
            # if a file that is currently tracking is renamed, replace it with the new renamed one
            for orgFpath, tracked in trackDict.items():
                _tracked = []
                for afile in tracked:
                    try:
                        new_afiles = renamedFiles[afile] # since renamedFiles contain both renamed and copied ones
                    except KeyError:
                        new_afiles = [afile] 
                    _tracked.extend(new_afiles)
                trackDict[orgFpath] = list(set(_tracked)) 

        # set rename to max 
        git_utils.setDiffMergeRenameToMax(workdir)
        # start  
        targetFilesTracking = {targetFile:[targetFile] for targetFile in targetFiles}
        refactoringOccurred = {}
        cnt_nothing_to_inspect = 0
        print ("Start...")
        for prevCommit, currCommit in tqdm(
            list(zip(targetCommits[:-1], targetCommits[1:]))
        ):
            ## check whether this commit is our target: compared with the prev commit of targetCommit
            print (f"at {prevCommit}, {currCommit}")
            diffFiles, deletedFiles, renamedFiles = git_utils.getDiffFiles(
                workdir, currCommit, prevCommit
            ) 
            # handle diffs
            filesToInspect = getFilesToInspect(targetFilesTracking, diffFiles) # k=prev_path, v=original path
            if len(filesToInspect):
                print (f"In {currCommit}")
            # handle deletion 
            updateFileTrackingByDelete(targetFilesTracking, deletedFiles) 
            ### temp
            #updateFileTrackingByRename(targetFilesTracking, renamedFiles)
            #continue
            ###
            if len(filesToInspect) == 0: # nothing to look
                updateFileTrackingByRename(targetFilesTracking, renamedFiles)
                continue # -> original code
            (
                dstDiffMapJSONFile, 
                dstRefactorJSONFile
            ) = RMinerProcesser.get_DefaultDiffMapAndRefactorFiles(dest, currCommit[:8])
            java_utils.changeJavaVer(17, use_sdk=use_sdk) # for RMINER 
            run_status = RMinerProcesser.single_run(
                list(filesToInspect.keys()), # prev path, meaning if this is renamed, then it will be 
                workdir, 
                currCommit, 
                dstDiffMapJSONFile, dstRefactorJSONFile
            )
            if not run_status: # stop
                print (f"Failed at {currCommit} and therby stop")
                break 
            
            print (f"At commit {currCommit}", workdir)
            #diffMaps = RMinerProcesser.readDiffFile(dstDiffMapJSONFile) # here, only filesToInpsect will be
            refactorings = RMinerProcesser.readRefactoringFile(dstRefactorJSONFile) # for later analysis ..?
            if refactorings.shape[0] > 0:
                refactoringOccurred[currCommit] = refactorings # for saving
            # update the files to track 
            #fileUpdates = {} # local file update
            #for prev_fpath, curr_fpath in diffMaps.keys(): # prev_fpath -> one d, the filesToInspect
            #    try:
            #        fileUpdates[(prev_fpath, filesToInspect[prev_fpath])].append(curr_fpath)
            #    except KeyError:
            #        fileUpdates[(prev_fpath, filesToInspect[prev_fpath])] = [curr_fpath]
            # extension & the files where all mutants were revealed will also be removed
            #updateFileTracking(targetFilesTracking, fileUpdates)  
            updateFileTrackingByRename(targetFilesTracking, renamedFiles)  
        print (
            f"Out of {len(targetCommits)}, nothings to inspect in {cnt_nothing_to_inspect} commits"
        )
        return refactoringOccurred
