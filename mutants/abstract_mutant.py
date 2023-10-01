from typing import List, Dict, Tuple, Set
import os, sys 
import xml.etree.ElementTree as ET
import utils.file_utils as file_utils
import utils.java_utils as java_utils
import utils.gumtree as gumtree


# Mutant Container
class AbstractMutantCont:
    def __init__(self, 
        mutNo:int, 
        left:Dict, 
        right:Dict, 
        mutOp:str, 
        targeted,  
        mutated_fpath:str, # the original starting point 
    ):
        self.mutNo = mutNo
        self.mutOp = mutOp
        # initial info -> here, right is the one that contain the mutation information
        self.left, self.right = left, right 
        self.mutated_fpath = mutated_fpath 
        self.targeted = targeted
        ## need to initiate
        # dynamically "replaced" as the mutation can be propagated
        # -> can have more than one element if the mutation propagates e.g., by refactoring
        # ... this actually should be in a dictionary form or dataframe 
        # -> appliedAts -> mainly for comparison 
        # Although this is for one mutant, as the time passes, this mutant can be propagated
        # to other parts of code, and thereby, appliedAts and tempApplyAts contains the filepath (i.e.,
        # the propagated) and the new location of THE mutant at that file (filepath)
        self.mut_target_text = None 
        self.appliedAts:Dict = None
        self.tempApplyAts:Dict = None 
        self.start, self.end, self.start_chr_cnt, self.end_chr_loc = None, None, None, None 

    def addNewApplyAt(self, 
        prevFpath:str, newFpath:str, 
        content:str, 
        loc:Tuple[int,int]
    ):
        try:
            _ = self.tempApplyAts[prevFpath]
        except KeyError:
            self.tempApplyAts[prevFpath] = {} # init 
        finally:
            try:
                self.tempApplyAts[prevFpath][newFpath].append([content,loc])
            except KeyError:
                self.tempApplyAts[prevFpath][newFpath] = [[content,loc]]

    def applyMutation(self, newloc:Tuple[int, int], targetContent:str) -> str: 
        return None  

    # already have ... but, since there can be more, and this happen per mut 
    def injectAndTest(
        self,  
        workdir:str, 
        targetFileContent:str, # checkout & get -> this is the content of newPath
        prevPath:str, 
        newPath:str,
        use_junit4:bool,
        test_class_pat:str,  
        woMutFailed_or_Error_Tests:Tuple[Set[str], Set[str]], 
        timeout:int = None,
        with_test_compile:bool = False, 
        ant_or_mvn:str = 'mvn', 
        which_mutant:str = 'major',
        **kwargs, # e.g., d4j_home, project
    ) -> Dict:
        """
        workdir -> assumed to be already at the target commit that matches with newMutLoc
        return True if all still pass, and False if any of them (i.e.,variations) is failed, i.e., revealed
        => Currently, any revealed (failed when applied), then consider the mutatn as REVEALED
        """
        import mutants.mutationTool as mutationTool
        import mutants.pitMutationTool as pitMutationTool
        try:
            newApplyAts = self.tempApplyAts[prevPath][newPath] # a list of pairs of content and location for new mutant injection
        except KeyError: # the cases where no matching at all for this mutant for this pair of prevPath and newPath 
            return {} # return 

        def overwrite(original_content, fpath):
            file_utils.fileWrite(original_content, fpath)

        woMutFailedTests, woMutErrorTests = woMutFailed_or_Error_Tests
        no_longer_valid, revealdMutInfo = [], {}
        for i, (new_content, newMutLoc) in enumerate(newApplyAts):
            mutatedContent = self.applyMutation(newMutLoc, targetFileContent) # ...
            if mutatedContent is None: 
                # later... some loggiing (e.g., status of no_longer_valid)
                no_longer_valid.append(i) # failed to apply mutation -> will skip this one
                continue
            overwrite(mutatedContent, os.path.join(workdir, newPath))
            try:
                #if ant_or_mvn == 'mvn': # whether pit or not it iwll use this 
                if which_mutant == 'major':
                    out = mutationTool.compile_and_run_test(
                        workdir, 
                        test_class_pat, 
                        use_junit4 = use_junit4, 
                        with_test_compile = with_test_compile, # likely false
                        with_log = False, 
                        timeout = timeout 
                    )
                else:
                    out = pitMutationTool.compile_and_run_test(
                        workdir, 
                        test_class_pat, 
                        with_test_compile = with_test_compile,
                        timeout = timeout, 
                        ant_or_mvn = ant_or_mvn,
                        **kwargs
                    )
                print ("out len", len(out))
                (src_compiled, test_compiled, failed_testcases, error_testcases) = out
            except Exception as e: # can be timeout or CallledProcessedError
                # another option is to write targetFileContent to newPath
                overwrite(targetFileContent, os.path.join(workdir, newPath)) # restore
                print (e)
                # here, also can be time-out -> meaning killed 
                if type(e).__name__ == "TimeoutExpired": # consider as killed 
                    print (f"Timeout by mut: {self.mutNo}")
                    print (e.cmd)
                    print (e.stdout)
                    print (e.stderr)
                    revealdMutInfo = {
                        'allftests':"timeout", 
                        'ftests':[], 
                        'alletests':[], 
                        'etests':[], 
                        'mutLoc':newMutLoc, 
                        'oplt':new_content, # the targeted one
                        'mutant':self.mut_target_text, 
                        'mutOp':self.mutOp
                    } # enough to reconstruct or expect what is mutated
                    break 
                else:
                    # for other erors, meaning something unexpected that can affect the performance occurs -> stop runing b/c previously it was ok
                    print (f'Unexpected error occurs while running tests: {self.mutNo}')
                    no_longer_valid.append(i) # will not stop, b/c this error might be specific to injecting the mutant at this location
                    continue 
                
            overwrite(targetFileContent, os.path.join(workdir, newPath)) # restore 
            if not src_compiled: # meaning something wrong with mutant (will be identified as its index)
                no_longer_valid.append(i) # can continue with the others 
                print (f"Compilation failed: {self.mutNo}")
                print ("\tprev", prevPath)
                print ("\tnew", newPath)
            else:
                #assert test_compiled, test_class_pat    
                if not test_compiled: # here, return None, b/c in previous case, it was compilating
                    print (f'Unexpected error occurs while test compilation: {self.mutNo}')
                    return None # this will be rarely the case b/c in most cases,test compilation will be disabled
                
                failed_by_mut = [ft for ft in failed_testcases if ft not in woMutFailedTests]
                error_by_mut = [ft for ft in error_testcases if ft not in woMutErrorTests]
                if len(failed_by_mut) > 0 or len(error_by_mut) > 0: # revealed
                    print ("Failed bu mut", len(failed_by_mut))
                    revealdMutInfo = {
                        'allftests':list(failed_testcases), 
                        'ftests':failed_by_mut, 
                        'alletests':list(error_by_mut), 
                        'etests':error_by_mut, 
                        'mutLoc':newMutLoc, 
                        'oplt':new_content, # the targeted one
                        'mutant':self.mut_target_text, 
                        'mutOp':self.mutOp
                    } # enough to reconstruct or expect what is mutated
                    print ("failed tests", failed_by_mut)
                    print ("error test", error_by_mut)
                    # for currently the testing
                    break # based on the strict strategy, no need to process further  (broken => variation doesn't matter)
        # clean up 
        ## actually for this, there might be more that are invalid when compile ..., 
        # but for revealing, it means that this entire mutant will no-longer be investigated -> so, in fact, no meaning to further investigate
        # here, for this pair of (prevPath, newPath) -> no_longer_valid muts that were propgated from prevPath to newPath 
        ### for this mutnat and for this particular pair of files for particular no_longer_valid-index-identified (propagated) mutants
        self.deleteNoLongerValids(prevPath, no_longer_valid, inTemp = True, newfpath = newPath) 
        return revealdMutInfo
    
    # should be run after looking all diverged files (i.e., mappings)
    def updateAppliedAts(self, prevPath:str): 
        # replace appliedAts with tempApplyAt 
        try:
            newApplyAts_p_prevfpath = self.tempApplyAts[prevPath]
        except KeyError:
            newApplyAts_p_prevfpath = None # for this prev
            #print ('None...') # since this is per-mutant, meaning failed to match any for new: nothing added 
            del self.appliedAts[prevPath] 

        if newApplyAts_p_prevfpath is not None:
            del self.appliedAts[prevPath] 
            for newPath in newApplyAts_p_prevfpath:
                self.appliedAts[newPath] = newApplyAts_p_prevfpath[newPath] # be simple here for now -> this cannot be empty, b/c deleteNoLongerValids -> delete an emtpy temp
            del self.tempApplyAts[prevPath] # will be deleted -> meaning after looking through all prev-files in applyMutaiotns -> this wil be {}
            #print ('\tafter', self.appliedAts, prevFpath)
            
    def saveAppliedAtInfo(self, dest, commit_hash:str):        
        # save mutation information 
        mutinfo_file = os.path.join(
            dest, 
            f"appliedAt.mut{self.mutNo}.{commit_hash[:8]}.json") # the dest will tell the origin of mutation
        #for fpath, vs in self.appliedAts.items():
        #    self.appliedAts[fpath] = list(set(vs)) # just in case
        # save self.appliedAts 
        with open(mutinfo_file, 'w') as f:
            import json  
            f.write(json.dumps(self.appliedAts))

    # delete variations that are no longer valid: e.g., not compiled or doesn't exist anymore by changes in mutant context (no map)
    def deleteNoLongerValids(self, 
        fpath:str, indicesToNoValid:List[int], 
        inTemp:bool = False, newfpath:str =  None
    ):
        indicesToNoValid = set(indicesToNoValid) # for speed up
        if not inTemp:
            self.appliedAts[fpath] = [
                v for i,v in enumerate(self.appliedAts[fpath]) if i not in indicesToNoValid] 
        else: # here, 
            assert newfpath is not None
            prevPath = fpath
            self.tempApplyAts[prevPath][newfpath] = [
                v for i,v in enumerate(
                    self.tempApplyAts[prevPath][newfpath]) if i not in indicesToNoValid
            ] 
            if len(self.tempApplyAts[prevPath][newfpath]) == 0: # all mutants in newfpath killed
                del self.tempApplyAts[prevPath][newfpath] # -> as this pair doesn't work (length 0)
    
    def deleteNoLongerExistByDelFiles(self, deletedFiles:List[str]) -> None:
        for deletedFile in deletedFiles:
            try:
                del self.appliedAts[deletedFile]
                print (self.mutNo, "deletd", deletedFile)
            except KeyError:
                continue
    
    def isEmpty(self) -> bool:
        return len(self.appliedAts) == 0 # meaning, none remained to look at 

    def initTempAppliedAts(self):
        self.tempApplyAts = {}       
