"""
"""
import numpy as np 
from typing import List, Dict, Tuple, Set, Union
import os, sys
import pandas as pd 
from tqdm import tqdm 
sys.path.insert(0, "../")
from utils import file_utils, coverage
import time 

MAX_MUTANTS_NUM = 500 # only test for this number

class MutantInjector:
    def __init__(
        self, mml_fpath:str, work_dir:str = ".", major_home:str = None, 
        target_file:str = None,
        #mutant_context_fpath:str = "mutants.context", 
        mutant_log_fpath:str = "mutants.log", 
        major_log_fpath:str = "major.log", 
        classes_dir:str = None, 
        test_classes_dir:str = None 
    ):
        self.mml_fpath = mml_fpath
        self.major_home = major_home if major_home is not None else MAJOR_HOME
        self.work_dir = work_dir # assume to be the ..
        #self.src_dir = os.path.join(self.work_dir, 'src/main')
        #self.src_dir = os.path.join(self.work_dir, 'src/java')
        self.classes_dir = os.path.join(
            self.work_dir, 
            'target/classes' if classes_dir is None else classes_dir)
        self.test_classes_dir = os.path.join(
            self.work_dir, 
            'target/test-classes' if test_classes_dir is None else test_classes_dir)
        self.target_file = target_file
        #self.mutant_context_fpath = mutant_context_fpath # -> this infact, does not exist!
        #if not bool(os.path.dirname(mutant_context_fpath)):
        #    self.mutant_context_fpath = os.path.join(self.work_dir, self.mutant_context_fpath)
        self.mutant_log_fpath = mutant_log_fpath
        if not bool(os.path.dirname(mutant_log_fpath)):
            self.mutant_log_fpath = os.path.join(self.work_dir, self.mutant_log_fpath)
        self.major_log_fpath = major_log_fpath
        if not bool(os.path.dirname(major_log_fpath)):
            self.major_log_fpath = os.path.join(self.work_dir, self.major_log_fpath)

    def checkInjected(self):
        mutants_dir = os.path.join(self.work_dir, "mutants")
        return os.path.exists(mutants_dir)

    def inject(self, target_file:str = None) -> pd.DataFrame:
        """
        inject mutants
        """
        if target_file is None: target_file = self.target_file
        assert target_file is not None
        compile_cmd = "mvn compile"
        _ = run_cmd(compile_cmd, self.work_dir)
        class_file_saved = os.path.join(self.work_dir, "mutated_classes")
        os.makedirs(class_file_saved, exist_ok=True)
        cmd = f"{self.major_home}/major --mml {self.mml_fpath}"
        cmd += f" -d {class_file_saved}"
        cmd += f" -cp {self.classes_dir}" #-cp {self.test_classes_dir} {target_file}"
        cmd += f" {target_file}"
        cmd += " --export export.mutants" #--export strict.checks"

        print (f"Injecting: {cmd} at {self.work_dir}")
        output, err = run_cmd(cmd, self.work_dir) # 
        print (f"Injecting done!")
        if (err is not None) and not self.checkInjected():  # here, ... if mutants are injected, we will ignore this error
            print (output)
            print ("Error while injecting")
            print (err)
            sys.exit()
        self.target_file = target_file
        return self.get_mutant_infos()

    def write_mml(self, target_classes:List[str]):
        """<...>;"org.apache.commons.math.optimization.direct.BOBYQAOptimizer@bobyqb"
        find "Enable" key word and then ... 
        """
        pass 

    def _get_mutant_files(self, mutant_dir:str, base_name_of_file:str) -> List[str]:
        for root, _, files in os.walk(mutant_dir):
            for file in files:
                if file.endswith(base_name_of_file):
                    yield os.path.join(root, file)
                
    def get_mutant_files(self) -> Dict[int, str]:
        mutant_dir = os.path.join(self.work_dir, "mutants")
        base_name_of_file = os.path.basename(self.target_file)
        mutant_files = self._get_mutant_files(mutant_dir, base_name_of_file)
        mut_file_paths = {}
        for mutant_file in mutant_files:
            full_dirpath = os.path.dirname(mutant_file)
            sub_dirpath = full_dirpath[len(mutant_dir):]
            if sub_dirpath[0] == "/": sub_dirpath = sub_dirpath[1:]
            #print (full_dirpath, mutant_dir)
            #print (sub_dirpath)
            mut_id = int(sub_dirpath.split("/")[0]) # will be mutant number 
            mut_file_paths[mut_id] = mutant_file 
        return mut_file_paths         

    def get_major_log_info(self) -> pd.DataFrame:
        import re 
        # Logging mutant 704 
        cmd = f"grep 'Logging mutant' {self.major_log_fpath}"
        output, err = run_cmd(cmd, self.work_dir)
        mut_loc_info = {'mutantNo':[], 'lno':[], 'collno':[], 'cls_mth':[]}
        for line in output.split("\n"):
            line = line.strip()
            if not bool(line): continue
            matched = re.search("Logging\s+mutant\s+([0-9]+):.*line\s+([0-9]+),\s+col\s+([0-9]+)", line)
            assert matched is not None, line
            mutNo, lno, collno = list(map(int, matched.groups()))
            matched = re.search(".*:\s+([a-zA-Z\.<>0-9]+)\(line.*", line) # this can't catch init 
            assert matched is not None, line
            cls_mth_name = matched.group(1)

            mut_loc_info['mutantNo'].append(mutNo)
            mut_loc_info['lno'].append(lno)
            mut_loc_info['collno'].append(collno)
            mut_loc_info['cls_mth'].append(cls_mth_name)
        mut_loc_info = pd.DataFrame(mut_loc_info)
        #for output.split("\n"):
        #with open(self.major_log_fpath) as f:
        #    for line in f.readlines():
        #        line = line.strip()
        #        re.ma
        mut_loc_info = mut_loc_info.set_index('mutantNo')
        return mut_loc_info 
    
    def get_mutant_infos(self) -> pd.DataFrame:
        #mutant_info = pd.read_csv(self.mutant_log_fpath, delimiter = ":", header = None)
        with open(self.mutant_log_fpath) as f:
            lines = [line.strip() for line in f.readlines()]
        rows = []
        for line in lines:
            ts = line.split(":")
            ts_last = ":".join(ts[6:])
            rows.append(ts[:6] + [ts_last])
        mutant_info = pd.DataFrame(rows)
        mutant_info_columns = ['mutantNo', 'mutationOperatorGroup', 'org', 'new', 'target', 'lno', 'mutant']
        mutant_info.columns = mutant_info_columns
        ## parse mutant 
        mutant_info['mutant'] = mutant_info.mutant.apply(lambda v:tuple([tk.strip() for tk in v.split("|==>")]))
        mutant_info['fpath'] = [os.path.join(self.work_dir, self.target_file)] * len(mutant_info)
        mutant_info['mutantNo'] = mutant_info.mutantNo.apply(lambda v:int(v))
        mutant_info = mutant_info.set_index('mutantNo')
        mut_file_paths = self.get_mutant_files()
        mut_file_df = pd.DataFrame.from_dict(mut_file_paths.items())
        mut_file_df.columns = ['mutantNo', 'mut_fpath']
        mut_file_df = mut_file_df.set_index('mutantNo')
        mut_loc_info_df = self.get_major_log_info()
        complete_mutant_info = pd.merge(
            mutant_info, mut_file_df, how = 'inner', left_index=True, right_index=True)
        complete_mutant_info = pd.merge(
            complete_mutant_info, mut_loc_info_df, how = 'inner', left_index=True, right_index=True)
        return complete_mutant_info 

    def get_covered_mutants(self, 
        mutant_info:pd.DataFrame, file_line_coverage:Dict[int,int]
    ) -> pd.DateOffset:
        n = len(mutant_info)
        #print (file_line_coverage)
        #print (mutant_info.lno_y.apply(lambda v:bool(file_line_coverage[v])))
        #mutant_info = mutant_info.loc[mutant_info.lno_y.apply(
        #    lambda v:bool(file_line_coverage[v])).values]
        covered_lnos = set([lno for lno, covered in file_line_coverage.items() if bool(covered)])
        mutant_info = mutant_info.loc[mutant_info.lno_y.isin(covered_lnos)]
        n_covered = len(mutant_info)
        print (f"Out of {n} muants, {n_covered} covered")
        return mutant_info

    def parse_context(self, context_file = "mutants.context"):
        """
        Index(['mutantNo', 'mutationOperatorGroup', 'mutationOperator',
       'nodeTypeBasic', 'nodeTypeDetailed', 'nodeContextBasic',
       'astContextBasic', 'astContextDetailed', 'parentContextBasic',
       'parentContextDetailed', 'parentStmtContextBasic',
       'parentStmtContextDetailed', 'hasLiteralChild', 'hasVariableChild',
       'hasOperatorChild'])
        hasVariableChild -> the ..../
        """
        context_file = os.path.join(self.work_dir, os.path.basename(context_file))
        mutant_context = pd.read_csv(context_file)
        ... 

    def test_mutant(
        self, 
        mutant_info:pd.DataFrame, 
        targetMutNos:List[int],
        test_class_pat:str, # temporary 
        testcase_name:str, # temporary
        woMut_failed_error_testcases:Tuple[Set[str],Set[str]], 
        use_junit4:bool = True, 
        dest:str = None,
        timeout:int = None
    ) -> Tuple[List[int], List[int]]:
        """
        """
        import shutil 
        dest = "." if dest is None else dest
        compile_cmd = "mvn compile" 
        #test_compile_cmd = "mvn test-compile -DskipTests=true"
        #_, test_compiler_err = run_cmd(test_compile_cmd, self.work_dir) # for tests, need to compile only once
        #if test_compiler_err is not None:
            #print (test_compiler_err)
            #sys.exit(0)
        # prepare mutation testing 
        # compile
        os.makedirs(dest, exist_ok=True)
        print ("Testing...")
        tempdir = os.path.join(self.work_dir, "temp")
        os.makedirs(tempdir, exist_ok=True)
        woMut_failed_testcases, woMut_error_testcases = woMut_failed_error_testcases
        compile_failed = []
        not_killed, _not_killed = [], []
        mutant_info_of_targets = mutant_info.loc[targetMutNos] if bool(targetMutNos) else mutant_info
        #saved_files = [] # likely just one org_fpath as we work on per-file mutation
        saved_files, org_file_contents = set(), dict()
        try:
            for mutNo, mut_info in tqdm(list(mutant_info_of_targets.iterrows())): 
                # replace the original file with the mutated 
                org_fpath = mut_info.fpath # this actully store the absolute path 
                mut_fpath = mut_info.mut_fpath # this actully store the absolute path 
                if org_fpath not in saved_files: # b/c already saved
                    #shutil.copyfile(org_fpath, os.path.join(tempdir, os.path.basename(org_fpath)))
                    with open(org_fpath) as f:
                        org_file_contents[org_fpath] = "".join(f.readlines())
                    saved_files.add(org_fpath)
                shutil.copyfile(
                    mut_fpath, 
                    os.path.join(os.path.dirname(org_fpath), os.path.basename(mut_fpath))
                )
                #print ("Mut path", mut_fpath, self.work_dir)
                #print ("org path", os.path.join(os.path.dirname(org_fpath), os.path.basename(mut_fpath)))
                #mut_content = file_utils.readFile(mut_fpath)
                #file_utils.fileWrite(
                #    os.path.join(os.path.dirname(org_fpath), os.path.basename(mut_fpath)))
                #file_utils.fileWrite()
                #print ('for mut', mut_fpath, os.path.join(os.path.dirname(org_fpath), os.path.basename(mut_fpath)))
                # compile 
                compile_output, compiler_err = run_cmd(compile_cmd, self.work_dir) # compile source code .. how about test??
                if compiler_err is not None:
                    compile_failed.append(mutNo)
                    continue #  
                # execute
                test_run_output, test_run_err, _test_class_pat = run_test(
                    self.work_dir, test_class_pat, testcase_name, 
                    use_junit4 = use_junit4,
                    timeout = timeout)
                if _test_class_pat is not None:
                    test_class_pat = _test_class_pat
                if test_run_err is not None: # time-out will also be catched here
                    #if check_timeout(test_run_err):
                    #    # killed ...
                    #else:
                    print (f"Error while running test for {mutNo}")
                    print (test_run_err) #  -> currently, set to ignore the error
                    continue # considered as killed
                # ... get the status of mutant killed 
                failed_testcases, error_testcases = parse_test_output(
                    self.work_dir, test_class_pat, with_log = False) # temporay
                if len(failed_testcases) + len(error_testcases) == 0: # passing (nothing to check here)
                    not_killed.append(mutNo)
                    _not_killed.append(mutNo)
                    print (f"Not killed: {mutNo}") 
                else: # further check 
                    failed_by_mut_tcs = [ft for ft in failed_testcases 
                                         if ft not in woMut_failed_testcases]
                    error_by_mut_tcs = [ft for ft in error_testcases 
                                         if ft not in woMut_error_testcases]
                    if len(failed_by_mut_tcs) + len(error_by_mut_tcs) == 0: # meaning no additional test failure/error by the mutant, meaning hidden
                        not_killed.append(mutNo)
                        _not_killed.append(mutNo)
                        print (f"killed or error but the same!: {mutNo}") 
                ## early saving 
                if len(_not_killed) == 5:
                    mutant_info.loc[_not_killed].to_pickle(
                        os.path.join(dest, f"will_inspect_{int(len(not_killed)/5)}.pkl"))
                    _not_killed = [] # reset 
                #if len(not_killed) > 20: # for early end
                #    break   
            # save the remaining in _not_killed
            if len(_not_killed) > 0:
                mutant_info.loc[_not_killed].to_pickle(
                    os.path.join(dest, f"will_inspect_{int(len(not_killed)/5)+1}.pkl"))
            # restore to unmutated file after 
            self.restore_to_no_mut(self.work_dir, org_file_contents)
        except Exception as e:
            print (e)
            # clean up
            self.restore_to_no_mut(self.work_dir, org_file_contents)
        return not_killed, compile_failed
    
    def run_ex(self, target_file:str, test_class_pat:str):
        """
        an example for injecting and testing the mutants to get the surviving ones
        """
        mutant_info = self.inject(target_file)
        mutant_info.to_pickle("all_mutinfo.pkl")
        #mutant_info = pd.read_pickle('all_mutinfo.pkl')
        not_killed, compile_failed = self.test_mutant(
            mutant_info, 
            None, # all 
            test_class_pat,
            None,
            use_junit4 = True
        )
        n_total = len(mutant_info)
        n_not_killed = len(not_killed)
        print (f"Out of {n_total} mutants, {n_not_killed} survived and {len(compile_failed)} failed to be compiled")
        mutant_info.loc[not_killed].to_pickle('will_inspect.pkl')
        return mutant_info.loc[not_killed]

    def selectMutant(self, mutant_info:pd.DataFrame, mutSelMth:str) -> List[int]:
        targetMutNos = []
        if mutSelMth == 'all':
            targetMutNos = mutant_info.index.values
        elif mutSelMth == "fixed":# e.g., randomly sample 10%
            all_targetMutNos = mutant_info.index.values
            import numpy as np 
            #targetMutNos = np.random.choice(
            #    all_targetMutNos, int(len(all_targetMutNos)*0.25), replace = False)
            if len(all_targetMutNos) > MAX_MUTANTS_NUM:
                num_sel = MAX_MUTANTS_NUM
            else:
                num_sel = len(all_targetMutNos)
            np.random.seed(0)
            targetMutNos = np.random.choice(
                all_targetMutNos, num_sel, replace = False)
        else:
            print (f"Invalid mut selection strategy: {mutSelMth}")
        print(len(targetMutNos))
        return targetMutNos.tolist()

    #def restore_to_no_mut(self, filesToRestore:List[str], tempdir:str):
        #import shutil
        #print ("Cleaning...") 
        #for fileToRestore in filesToRestore:
            #temp_file = os.path.join(tempdir, os.path.basename(fileToRestore))
            #shutil.copyfile(temp_file, fileToRestore) # restore back to the orignal dir
            #print (f"\t{temp_file} -> {fileToRestore}")
        #shutil.rmtree(tempdir) # clean-up
    def restore_to_no_mut(self, work_dir:str, org_file_contents:Dict[str,str]):
        print ("Cleaning...") 
        for fileToRestore, fileContent in org_file_contents.items():
            file_utils.fileWrite(fileContent, os.path.join(work_dir, fileToRestore))

    def run(self, 
        target_files:List[str], mutSelMth:str = "all", 
        testClassPatterns:Union[Dict[str,str], str] = None, 
        testcase_name:str = None,
        dest:str = "outputs/inter", 
        use_junit4:bool = True, 
    ) -> Dict[int, Tuple[str, pd.DataFrame]]:
        """
        an example for injecting and testing the mutants to get the surviving ones
        """
        mutant_info_pfile = {}
        ## get the initial test results without any mutation
        t1 = time.time()
        (src_compiled, test_compiled, 
         woMut_failed_testcases, woMut_error_testcases) = compile_and_run_test(
            self.work_dir, 
            ",".join(
                list(testClassPatterns.values())
                ) if isinstance(testClassPatterns, Dict) else testClassPatterns, 
            use_junit4 = use_junit4, 
            with_log = False, # here, the timeout is None
            with_coverage=True, 
        ) # here, no timeout (or timeout as None, b/c it is the frst run)
        t2 = time.time()
        #timeout = 2 * (t2 - t1)
        timeout = (t2 - t1) * 1.1
        #assert src_compiled and test_compiled, f"{src_compiled}, {test_compiled}" # b/c here, normal
        if not src_compiled or not test_compiled:
            print (f"Src: {src_compiled}, Test: {test_compiled}")
            return None
        print ('Original', woMut_failed_testcases, woMut_error_testcases)
        covgfile = coverage.get_covgfile(self.work_dir)
        # compile tests
        test_compile_cmd = "mvn test-compile -DskipTests=true"
        test_output, test_compiler_err = run_cmd(test_compile_cmd, self.work_dir) # for tests, need to compile only once
        if test_compiler_err is not None:
            print (test_output)
            print ("Test compilation error:\n", test_compiler_err)
            assert False
        #
        for i, target_file in enumerate(target_files):
            dstMutInfo_file = os.path.join(dest, f"all_mutinfo_file{i}.pkl")
            mutant_info = self.inject(target_file)
            ## get covered mutants 
            #print (coverage.get_line_coverage(covgfile, target_file))
            file_line_coverage = list(
                coverage.get_line_coverage(covgfile, target_file).values())[0]
            #print("bfr", type(mutant_info), len(mutant_info))
            mutant_info = self.get_covered_mutants(mutant_info, file_line_coverage)
            #print("aft", type(mutant_info), len(mutant_info))
            ### 
            mutant_info.to_pickle(dstMutInfo_file)
            # all is right here, as we don't know how many will survive
            #print (len(mutant_info))
            targetMutNos = self.selectMutant(mutant_info, mutSelMth) # some selection mechanism 
            not_killed, compile_failed = self.test_mutant(
                mutant_info, 
                targetMutNos, 
                #testClassPatterns[target_file], # test class pat for the current target_file
                testClassPatterns[target_file] if isinstance(testClassPatterns, Dict) else testClassPatterns,
                testcase_name,
                (woMut_failed_testcases, woMut_error_testcases),
                use_junit4 = use_junit4, 
                dest = os.path.join(dest, str(i)), 
                timeout = timeout
            )
            n_total = len(mutant_info)
            n_not_killed = len(not_killed)
            print (
                f"Out of {n_total} mutants, {n_not_killed} survived and {len(compile_failed)} failed to be compiled")
            # store
            mutant_info.loc[not_killed].to_pickle(
                os.path.join(dest, f'all_will_inspect_file{i}.pkl'))
            mutant_info_pfile[i] = (target_file, mutant_info.loc[not_killed])
        
        merged_file = os.path.join(dest, f"all_will_inspect_file_pfile.pkl") 
        with open(merged_file, 'wb') as f:
            import pickle 
            pickle.dump(mutant_info_pfile, f)
        print (f"Combined to-inspect info saved in {merged_file}")
        # move the generated mutants to the dest 
        import shutil 
        mutant_dir = os.path.join(self.work_dir, "mutants")
        ## needed to be zip
        new_mutant_zip_file = os.path.join(dest, "mutants")
        shutil.make_archive(new_mutant_zip_file, 'zip', mutant_dir)
        print (f"..mutants now saved in {new_mutant_zip_file}")
        assert os.path.exists(new_mutant_zip_file + ".zip"), new_mutant_zip_file + ".zip"
        shutil.rmtree(mutant_dir)
        new_mutant_log_fpath = os.path.join(dest, os.path.basename(self.mutant_log_fpath))
        shutil.move(self.mutant_log_fpath, new_mutant_log_fpath)
        new_major_log_fpath = os.path.join(dest, os.path.basename(self.major_log_fpath))
        shutil.move(self.major_log_fpath, new_major_log_fpath)
        print ('done!')
        return mutant_info_pfile
    
