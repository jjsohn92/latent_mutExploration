"""
EXPERIMENTAL_ARGUMENT_PROPAGATION,FALSE_RETURNS,TRUE_RETURNS,CONDITIONALS_BOUNDARY,
CONSTRUCTOR_CALLS,EMPTY_RETURNS,INCREMENTS,INLINE_CONSTS,INVERT_NEGS,MATH,NEGATE_CONDITIONALS,
NON_VOID_METHOD_CALLS,NULL_RETURNS,PRIMITIVE_RETURNS,REMOVE_CONDITIONALS_EQUAL_IF,
REMOVE_CONDITIONALS_EQUAL_ELSE,REMOVE_CONDITIONALS_ORDER_IF,REMOVE_CONDITIONALS_ORDER_ELSE,
VOID_METHOD_CALLS,EXPERIMENTAL_BIG_DECIMAL,EXPERIMENTAL_BIG_INTEGER,EXPERIMENTAL_MEMBER_VARIABLE,
EXPERIMENTAL_NAKED_RECEIVER,REMOVE_INCREMENTS,EXPERIMENTAL_SWITCH,EXPERIMENTAL_BIG_DECIMAL,
EXPERIMENTAL_BIG_INTEGER


=> here, this will cover up to matcher 
"""

import os, sys 
from typing import List, Dict, Tuple 
import xml.etree.ElementTree as ET
from tqdm import tqdm 
sys.path.insert(0, "..")
import utils.mvn_utils as mvn_utils
import utils.file_utils as file_utils
import utils.java_utils as java_utils
import utils.git_utils as git_utils
import utils.ant_d4jbased_utils as ant_d4jbased_utils
import utils.ant_mvn_utils as ant_mvn_utils
import utils.gumtree as gumtree
import re

def register_all_namespaces(filename):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

def addMVNPitPlugin(pom_file:str, junit_ver:str = "4.11"):
    """
    1. change to junit 4.11 -> due to errors when running coverage -> not sure whether this will be the same if we use "the right" version
    2. add pitest-command-line dependency
    """
    register_all_namespaces(pom_file)
    tree, root, tag_to_root = mvn_utils.get_tree_root_and_root_tag(pom_file)
    tag_to_dependencies = "{}dependencies".format(tag_to_root)
    tag_to_dependency = "{}dependency".format(tag_to_root)
    tag_to_artifactId = "{}artifactId".format(tag_to_root)
    tag_to_version = "{}version".format(tag_to_root)
    # modify to junit have junit_ver
    for depds_node in root.iter(tag_to_dependencies):
         for depd_node in depds_node.iter(tag_to_dependency):
            artifact_nodes = depd_node.findall(tag_to_artifactId)
            if len(artifact_nodes) == 0:
                continue 
            else:
                artifact_node = artifact_nodes[0]
                if artifact_node.text == 'junit':
                    version_nodes = depd_node.findall(tag_to_version)
                    #assert len(version_nodes) > 0, depd_node
                    if len(version_nodes) == 1:
                        junit_version = version_nodes[0].text 
                        if junit_version != junit_ver:
                            version_nodes[0].text = junit_ver
                    else:
                        version_node_str = f"<version>{junit_ver}</version>"
                        version_node = ET.fromstring(version_node_str)
                        depd_node.append(version_node)
                    break 
                else:
                    continue
    # add pitest plugin
    tag_to_build = "{}build".format(tag_to_root)
    build_node = tree.find(tag_to_build)
    tag_to_plugins = "{}plugins".format(tag_to_root)
    plugins_node = build_node.find(tag_to_plugins)
    #plugins_node = tree.find(tag_to_plugins)
    pitest_plugin_str = "    <plugin> \n\
            <groupId>org.pitest</groupId>\n\
            <artifactId>pitest-maven</artifactId>\n\
            <version>1.7.4</version>\n\
            <configuration>\n\
                <skipFailingTests>true</skipFailingTests>\n\
            </configuration>\n\
        </plugin>"
    pitest_plugin_node = ET.fromstring(pitest_plugin_str)
    plugins_node.append(pitest_plugin_node)

    import shutil
    shutil.move(pom_file, pom_file + ".bak") # for backup 
    # overwrite the current pom file
    tree.write(
        pom_file, 
        method='xml',
        xml_declaration=True
    )

def restore_pomfile(pom_file:str):
    import shutil 
    if os.path.exists(pom_file + ".bak"):
        shutil.move(pom_file, pom_file + ".pitest")
        shutil.move(pom_file + ".bak", pom_file)

def isAt(start_end_pos_pline:Dict, targetLno:int, index:int) -> bool:
    pos = index + 1
    start, end = start_end_pos_pline[targetLno]
    return (pos >= start) and (pos <= end)

def isBelow(start_end_pos_pline:Dict, targetLno:int, index:int) -> bool:
    pos = index + 1
    _, end = start_end_pos_pline[targetLno]
    return pos > end

def isBlock(target_e:ET.Element) -> bool:
    try:
        _type = target_e.attrib['type']
        return _type == 'Block'
    except Exception:
        return False 

def hasBlock(target_e:ET.Element) -> bool:
    for e in target_e:
        try:
            _type = e.attrib['type']
        except KeyError:
            continue 
        if (_type == 'Block') and (_type != 'ReturnStatement'): 
            return True 
    return False

def findSpecificTypeElements(
    start_e:ET.Element, targetTypes:List[str],
    lno_start_pos:int, lno_end_pos:int, 
    only_direct_child:bool = False, 
    check_itself:bool = True 
) -> List[Tuple[ET.Element, str]]:
    targets = []
    to_lookat = list(start_e.iter()) if not only_direct_child else list(start_e)
    if check_itself: to_lookat += [start_e]
    for e in to_lookat:
        try:
            _type = e.attrib['type']
        except KeyError:
            continue   
        if _type in set(targetTypes):
            pos = int(e.attrib['pos']) + 1
            if lno_start_pos is None: # meanig the position does not matter 
                targets.append([e, _type])
            elif (pos >= lno_start_pos) and (pos <= lno_end_pos):
                targets.append([e, _type])
    return targets


def findSpecificTypeElements_recur(
    start_e:ET.Element,
    targetTypes:List[str],
    lno_start_pos:int, lno_end_pos:int, 
    targets:List[Tuple[ET.Element, str]],
    is_parent_block:bool = False, 
    label:str = None
) -> Tuple[List[Tuple[ET.Element, str]], bool]:
    _type = start_e.attrib['type']
    pos = int(start_e.attrib['pos']) + 1
    if _type == 'Block':
        if (pos >= lno_start_pos) and (pos <= lno_end_pos):
            is_parent_block = True  
        else: # this block is out of the range
            return targets, is_parent_block
    elif _type in set(targetTypes):
        start_e_label = None 
        if label is not None:
            try:
                start_e_label = start_e.attrib['label']
            except Exception:
                start_e_label = None 
        if _type == 'QualifiedName':
            start_e_label = start_e_label.split(".")[-1]
        if start_e_label == label: # if label = None, then start_e_label is always None  
            if not is_parent_block:
                targets.append([start_e, _type])
            else: # is_parent_block = True
                if (pos >= lno_start_pos) and (pos <= lno_end_pos): # further check 
                    targets.append([start_e, _type])
                else: # and now, it is out of the line range, meaning completely starts from new line 
                    return targets, is_parent_block # no need to further look this branch 
    # further look 
    for e in start_e: # check for its child 
        findSpecificTypeElements_recur(
            e, targetTypes,
            lno_start_pos, lno_end_pos, 
            targets,
            is_parent_block = is_parent_block, 
            label = label
        )        
    return targets, is_parent_block

def findElement(
    codeTree:ET.ElementTree, 
    start_end_pos_pline:Dict[int, Tuple[int,int]], 
    targetLno:int, 
    targetTypes:List[str], 
    label:str = None
) -> List[Tuple[ET.Element, str]]:
    # 
    ret_es = []
    targetTypes = set(targetTypes)
    for e in codeTree.iter():
        try:
            index = int(e.attrib['pos'])
        except KeyError:
            continue 
        if isAt(start_end_pos_pline, targetLno, index):
            ret_es.append(e)
        elif isBelow(start_end_pos_pline, targetLno, index):
            break 
    if len(ret_es) == 0:
        return None 
    else:
        targets, covered = [], set([])
        lno_start_pos, lno_end_pos = start_end_pos_pline[targetLno]
        for ret_e in ret_es:  
            _targets = []
            _targets, _ = findSpecificTypeElements_recur(
                ret_e, targetTypes, 
                lno_start_pos, lno_end_pos, 
                _targets, 
                is_parent_block = isBlock(ret_e), 
                label = label
            )
            for t, t_type in _targets:
                k = t.attrib['pos'] + "-" + t.attrib['length']
                if k in covered:
                    continue
                else:
                    targets.append([t, t_type])
                    covered.add(k)
        return targets 

def findElement_revised(
    codeTree:ET.ElementTree, 
    start_end_pos_pline:Dict[int, Tuple[int,int]], 
    targetLno:int, 
    targetTypes:List[str], 
    label:str = None
) -> List[Tuple[ET.Element, str]]:
    # for this, the "direct" parent should start at targeted line
    ret_es:List[ET.Element] = []
    targetTypes = set(targetTypes)
    for e in codeTree.iter():
        try:
            index = int(e.attrib['pos'])
        except KeyError:
            continue 
        if isAt(start_end_pos_pline, targetLno, index):
            ret_es.append(e)
        elif isBelow(start_end_pos_pline, targetLno, index):
            break 
    if len(ret_es) == 0:
        return None 
    else:
        targets = []
        for ret_e in ret_es:
            ret_e_type = ret_e.attrib['type']
            if ret_e_type in targetTypes:
                if label is not None:
                    ret_e_label = ret_e.attrib['label']
                    if ret_e_type == 'QualifiedName':
                        ret_e_label = ret_e_label.split(".")[-1]
                    if ret_e_type == 'NumberLiteral':
                        try:
                            _ = eval(ret_e_label)
                        except Exception: # e.g., 0.0F
                            ret_e_label = ret_e_label[:-1]
                        if eval(ret_e_label) == eval(label):
                            targets.append(ret_e)
                            continue
                    else:
                        if ret_e_label == label:
                            targets.append(ret_e)
                            continue
                else:
                    targets.append(ret_e)
                    continue
            for targetType in targetTypes:
                if label is None:
                    child_pat = f"./*[@type = '{targetType}']"
                    matched_es = ret_e.findall(child_pat)
                    targets.extend(matched_es)
                else:
                    if targetType == 'QualifiedName':
                        child_pat = f"./*[@type = '{targetType}']"
                        _matched_es = ret_e.findall(child_pat)
                        matched_es = []
                        for c in _matched_es: # direct child 
                            c_label = c.attrib['label'].split(".")[-1]
                            if c_label == label:
                                matched_es.append(c)
                    elif targetType == 'NumberLiteral':
                        child_pat = f"./*[@type = '{targetType}']"
                        _matched_es = ret_e.findall(child_pat)
                        matched_es = []
                        for c in _matched_es: # direct child 
                            c_label = c.attrib['label']
                            try:
                                _ = eval(c_label) 
                            except Exception: # e.g., 0.0F
                                c_label = c_label[:-1]
                            if eval(c_label) == eval(label):
                                matched_es.append(c)
                    else:
                        child_pat = f"./*[@type = '{targetType}'][@label = '{label}']"
                        matched_es = ret_e.findall(child_pat)
                    targets.extend(matched_es)
        targets = list(set(targets))
        # add type
        ret_targets = []
        for e in targets:
            e_type = e.attrib['type']
            ret_targets.append([e, e_type])
        return ret_targets 

def getElementByEncounterOrder(
    targets:List[ET.Element], 
    mutant:ET.Element, 
    groupedMutantsByLandM:Dict[Tuple[int, str, str], List[ET.Element]],
    ret_lengths:bool = False
) -> Tuple[ET.Element, int]:
    # targets -> from codeTree (meaning the output of source code parsing)
    # groupedMutantsByLandM -> contain all mutants regardless of their status
    mutatedClass = PitMutantProcessor.getMutatedClass(mutant)
    lineNumber = PitMutantProcessor.getLineNumber(mutant)
    mutator_op = PitMutantProcessor.getMutator(mutant)
    description = PitMutantProcessor.getDescription(mutant)    
    #  
    k = (mutatedClass, lineNumber, mutator_op, description)
    mutantsWithSameKey = groupedMutantsByLandM[k] # the same line, the same operator
    n_same_key = len(mutantsWithSameKey)
    n_targets = len(targets)
    if n_same_key != n_targets: # meaning something is missing here, and don't have further information to differentiate
        if not ret_lengths:
            return None, None
        else:
            return None, None, (n_same_key, n_targets)
    # get the appearance index of target_index
    index_of_target_mut = PitMutantProcessor.getIndex(mutant)
    idx_to_same_appOrder = 0
    for mut in mutantsWithSameKey:
        idx = PitMutantProcessor.getIndex(mut)
        if idx < index_of_target_mut:
            idx_to_same_appOrder += 1 
    target = targets[idx_to_same_appOrder]
    if not ret_lengths:
        return target, idx_to_same_appOrder
    else:
        return target, idx_to_same_appOrder, (n_same_key, n_targets)

def getStartAndEndCharCnt(e:ET.Element) -> Tuple[int, int]:
    # start from 1 
    start = int(e.attrib['pos']) + 1 # start fr
    end = start + int(e.attrib['length']) - 1# can be same as start
    return start, end 

def parse_test_output(
    work_dir:str, test_class_pat:str, with_log:bool = True
) -> Tuple[List[str], List[str]]:
    """
    return failed test cases
    """
    report_dir = os.path.join(work_dir, "target/surefire-reports")
    import glob
    if ',' in test_class_pat: # combined ones
        report_files = []
        for _pat in test_class_pat.split(","):
            report_files.extend(glob.glob(os.path.join(report_dir, f"TEST-{_pat}.xml")))
    else:
        report_files = glob.glob(
            os.path.join(report_dir, f"TEST-{test_class_pat}.xml"))
    import xml.etree.ElementTree as ET
    n_testcases = 0
    failed_testcases, error_testcases, failure_msgs, error_msgs = [], [], [], []
    print (f"test reports: {len(report_files)}")
    for report_file in tqdm(report_files):
        #print ('Report ', report_file)
        tree = ET.parse(report_file)
        root = tree.getroot()
        testcase_nodes = root.findall('testcase')
        n_testcases += len(testcase_nodes)
        for testcase_node in testcase_nodes:
            # check for failure 
            failure_nodes = testcase_node.findall('failure')
            fail_node, err_node = None, None 
            if len(failure_nodes) > 0:
                fail_node = failure_nodes[0]
                try:
                    test_class = testcase_node.attrib['classname'] 
                except KeyError:
                    test_class = "None" # e.g., Lang50
                testcase_name = testcase_node.attrib['name']
                failure_msg = fail_node.text
                failure_msgs.append(failure_msg.strip())
                failed_testcases.append(f"{test_class}#{testcase_name}")
            else: # further check for error
                error_nodes = testcase_node.findall('error')
                if len(error_nodes) > 0:
                    err_node = error_nodes[0]
                    test_class = testcase_node.attrib['classname'] 
                    testcase_name = testcase_node.attrib['name']
                    error_msg = err_node.text
                    error_msgs.append(error_msg.strip())
                    error_testcases.append(f"{test_class}#{testcase_name}")
    n_failed, n_error = len(failed_testcases), len(error_testcases)
    n_passed = n_testcases - n_failed - n_error
    if with_log:
        print (f"Out of {n_testcases}, pass: {n_passed}, fail: {n_failed}, error: {n_error}")
    return failed_testcases, error_testcases

def getFailingTests(work_dir:str, ant_or_mvn:str, testPat:str) -> List[str]:
    if ant_or_mvn == 'ant_d4j':
        failing_tests_file = os.path.join(work_dir, "failing_tests")
        failed_testcases = []
        if not os.path.exists(failing_tests_file):
            return []
        else:
            with open(failing_tests_file) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not bool(line): continue 
                    if line.startswith("---"):
                        line = line[len("---"):].strip()
                        fulltestcase = line.replace("::", "#")
                        failed_testcases.append(fulltestcase)
            # clean up 
            os.remove(failing_tests_file)
            return list(set(failed_testcases))
    else:
        failed_testcases, error_testcases = parse_test_output(
            work_dir, testPat, with_log = False)
        return failed_testcases + error_testcases
    
def compile_and_run_test(
    work_dir:str, 
    test_class_pats:str, 
    with_test_compile:bool = True, 
    timeout:int = None,  # ----> need to set
    ant_or_mvn:str = 'mvn', 
    **kwargs
):
    """
    test_class_pats: expect multiple test class patterns to be joined by ","
    """
    import subprocess 
    src_compiled, test_compiled = True, True
    if ant_or_mvn == 'ant_d4j':
        src_compiled, _compile_cmd = ant_d4jbased_utils.compile(
            kwargs['d4j_home'], kwargs['project'], work_dir)
        if with_test_compile:
            test_compiled, _tst_compile_cmd = ant_d4jbased_utils.test_compile(
                kwargs['d4j_home'], kwargs['project'], work_dir)
    else: # mvn,
        src_compiled, ant_or_mvn, _compile_cmd = ant_mvn_utils.compile(
            work_dir, prefer = ant_or_mvn)
        if with_test_compile:
            test_compiled, ant_or_mvn, _tst_compile_cmd = ant_mvn_utils.test_compile(
                work_dir, prefer = ant_or_mvn)

    if not src_compiled:
        print ("Compile Erorr:")
        import subprocess 
        out = subprocess.check_output("java -version", shell = True)
        print (out)
        print (f"While compiling srcs at {work_dir}")
        print (list(kwargs))
        print (_compile_cmd)
        return (False, None, None, None) 

    if not test_compiled:
        print ("Test compile Erorr:")
        print (f"While addressing {test_class_pats} at {work_dir}")
        print (list(kwargs))
        print (_tst_compile_cmd)
        return (True, False, None, None) # True, False, None, None
    
    # run tests
    #print ('Ant or mvn', ant_or_mvn)
    if ant_or_mvn == 'ant_d4j':
        targetTestClassesFile = java_utils.writeTargetTests(work_dir, test_class_pats)
        failingTestFile = os.path.join(work_dir, "failing_tests")
        tested, _run_test_cmd = ant_d4jbased_utils.run_tests(
            kwargs['d4j_home'], kwargs['project'], 
            work_dir, 
            timeout, 
            f"-DOUTFILE={failingTestFile}", 
            f"-propertyfile {targetTestClassesFile}"
        ) 
    else:
        if ant_or_mvn == 'mvn':
            tested, _run_test_cmd = ant_mvn_utils.mvn_call(
                work_dir, "test", timeout, 
                f"-Dtest={test_class_pats}", "-Dmaven.test.failure.ignore=true"
            )
        else:
            print ("Not implemented yet")
            _run_test_cmd = None

    if not bool(tested): # error while testing:
        from subprocess import CalledProcessError
        raise CalledProcessError(
            returncode = 1, #.returncode,
            cmd = _run_test_cmd, 
            stderr = "Error while testing"
        )
    # get failing and error tests 
    failed_or_errored_tests = getFailingTests(work_dir, ant_or_mvn, test_class_pats) 
    failed_or_errored_tests = set(failed_or_errored_tests)
    return src_compiled, test_compiled, set(failed_or_errored_tests), set([])

def checkWhetherPass(src_compiled:bool, test_compiled:bool) -> int:
    if not bool(src_compiled):
        return 2 
    elif bool(src_compiled) and bool(test_compiled): # True and True
        return 0
    elif bool(src_compiled) and not bool(test_compiled): # True and False
        return 1
    else:
        print ("Shoudn't be reached", src_compiled, test_compiled)
        sys.exit()

class PitMutantProcessor():
    """
    default operators:
        1) conditional
    """
    def __init__(self) -> None:
        pass
    # currently only the default ones
    MUTATORS = {
        'defaults':[
            "CONDITIONALS_BOUNDARY",
            "INCREMENTS",
            "INVERT_NEGS",
            "MATH",
            "NEGATE_CONDITIONALS",
            "VOID_METHOD_CALLS", 
            "EMPTY_RETURNS",
            "FALSE_RETURNS",
            "TRUE_RETURNS",
            "NULL_RETURNS",
            "PRIMITIVE_RETURNS", 
        ]
    } 
    @staticmethod
    def getCmd(
        targetClasses:str, targetTests:str, is_mvn:bool = True,
        mutators_config:str = 'defaults', **kwargs
    ):
        if is_mvn: # currently disabled
            # the output will be target/pit-reports
            cmd = "mvn test-compile org.pitest:pitest-maven:mutationCoverage" 
            cmd += f" -DtargetClasses={targetClasses}"
            cmd += f" -DtargetTests={targetTests}"
            cmd += " -DoutputFormats=xml"
            mutator_list = ",".join(PitMutantProcessor.MUTATORS[mutators_config]) 
            cmd += f" -Dmutators={mutator_list}"
            cmd += " -DtimestampedReports=false"
            cmd += " -DtimeoutFactor=2 -DtimeoutConst=300"
        else:
            # command-line: ant will
            assert kwargs is not None
            cmd = "java -jar " + kwargs['pit_jar_path'] \
                + " --classPath " + kwargs['classPath'] \
                + " --reportDir " + kwargs['reportDir'] \
                + " --targetClasses " + targetClasses \
                + " --targetTests " + targetTests \
                + " --sourceDirs " + kwargs['sourceDirs'] \
                + " --mutators " + ",".join(PitMutantProcessor.MUTATORS[mutators_config]) \
                + " --outputFormats XML" \
                + " --timestampedReports=false" \
                + " --timeoutFactor 2" \
                + " --timeoutConst 300" \
                + " --skipFailingTests=true" # by setting this to true, failing tests are ignored from computing the status
                #+ " --fullMutationMatrix=true" # 
                #+ " --maxMutationsPerClass 1000" \
        return cmd

    @staticmethod
    def getGroupedLivedMutants(mutation_file:str) -> Dict[str, List[ET.Element]]:
        #print ("mutation file", mutation_file)
        tree, _, _ = mvn_utils.get_tree_root_and_root_tag(mutation_file)
        mutants = tree.findall('mutation')
        lived_mutants = {}
        cnt = 0
        for mutant in mutants:
            status = mutant.attrib['status']
            if status == 'LIVED' or status == 'SURVIVED':
                fpath = mutant.find("sourceFile").text
                _fpath_basename = fpath.split(".")[0]
                mutatedClass = PitMutantProcessor.getMutatedClass(mutant)         
                ts = mutatedClass.split(".")
                mutatedFpath = None 
                for i, t in enumerate(ts[::-1]):
                    if t == _fpath_basename:
                        mutatedFpath = "/".join(ts[:len(ts) - i]) + ".java"
                        break 
                assert mutatedFpath is not None, f"{fpath}, {mutatedClass}"
                try:
                    _ = lived_mutants[mutatedFpath]
                except KeyError as e:
                    lived_mutants[mutatedFpath] = []
                lived_mutants[mutatedFpath].append(mutant)
                cnt += 1
        print (f"Out of {len(mutants)}, {cnt} are covered but not-killed")
        return lived_mutants    
    
    @staticmethod
    def getGroupedByLineAndMutator(mutation_file:str) -> Dict[Tuple[int, str, str], List[ET.Element]]:
        """
        group mutants based on the locator and the line
        """
        tree, _, _ = mvn_utils.get_tree_root_and_root_tag(mutation_file)
        mutants = tree.findall('mutation')
        grouped_mutants = {} 
        for mutant in mutants:
            mutatedClass = PitMutantProcessor.getMutatedClass(mutant)
            lineNumber = PitMutantProcessor.getLineNumber(mutant)
            mutator_op = PitMutantProcessor.getMutator(mutant)
            description = PitMutantProcessor.getDescription(mutant)      
            k = (mutatedClass, lineNumber, mutator_op, description)
            try:
                grouped_mutants[k].append(mutant)
            except KeyError:
                grouped_mutants[k] = [mutant]
        return grouped_mutants  

    @staticmethod
    def prepare(work_dir:str, file:str) -> Tuple[ET.ElementTree, str, Dict[int, Tuple[int,int]]]:
        codeTree = gumtree.processFile_gumtree_by_file(work_dir, file, file)
        file_content = file_utils.readFile(file)
        codeTree = gumtree.addMissingInfixOps(codeTree, file_content, file_content)
        start_end_pos_pline = file_utils.compute_start_end_pos_pline(file_content)
        return (codeTree, file_content, start_end_pos_pline)

    @staticmethod 
    def getMutationReportDir(work_dir:str, is_mvn:bool = True):
        if is_mvn:
            return os.path.join(work_dir, "target/pit-reports")
        else:
            return os.path.join(work_dir, "pit-reports")
        
    @staticmethod 
    def getMutationResultFile(work_dir:str, is_mvn:bool = True):
        report_dir = PitMutantProcessor.getMutationReportDir(work_dir, is_mvn = is_mvn)
        return os.path.join(report_dir, "mutations.xml")

    @staticmethod
    def runMutation(
        work_dir:str, 
        targetFiles:str, 
        targetClasses:str, 
        targetTests:str, 
        is_mvn:bool = True, 
        mutators_config:bool = 'defaults', 
        **kwargs
    ):
        """
        1. run mutatin (pitest)
        2. get the mutation results (e.g., mutations.xml)
        """
        import subprocess
        reportDir = PitMutantProcessor.getMutationReportDir(work_dir, is_mvn = is_mvn)
        cmd = PitMutantProcessor.getCmd(
            targetClasses, targetTests, is_mvn = is_mvn, 
            reportDir = reportDir, 
            mutators_config = mutators_config, 
            **kwargs
        )
        if is_mvn: addMVNPitPlugin(os.path.join(work_dir, "pom.xml"))
        mutation_file = PitMutantProcessor.getMutationResultFile(work_dir, is_mvn = is_mvn) 
        # run mutation 
        if not os.path.exists(mutation_file):
            # timeout may occur
            out = subprocess.run(
                cmd,
                shell = True, 
                cwd = work_dir, 
                capture_output = True, 
                timeout = (60 * 60 * 4) # if run more than 6 hours, time-out
            )
            if out.returncode != 0:
                print (f"Error while mutating {targetClasses} with {targetTests} at {work_dir}")
                import traceback, logging
                logging.error(traceback.format_exc())
                # cleanup
                if is_mvn:
                    restore_pomfile(os.path.join(work_dir, "pom.xml"))
                assert False
        # process raw mutation file to obtain source-code level mutation information 
        ret_mut_lr_pairs = PitMutantProcessor.genMutLRPairs(
            work_dir, mutation_file, targetFiles, further = kwargs['further'])
        if is_mvn: restore_pomfile(os.path.join(work_dir, "pom.xml"))
        return ret_mut_lr_pairs
        
    @staticmethod
    def genMutLRPairs(
        work_dir:str, 
        mutation_file:str,
        fullPathMutatedFiles:List[str], 
        further:bool = False
    ) -> Dict:
        # full_filePath : e.g.. src/main/...
        def getFullPath(fullPaths:List[str], file:str) -> str:
            for fullPath in fullPaths:
                if fullPath.endswith(file):
                    return fullPath
            return None 
        def get_core_path(full_filePath:str, work_dir:str) -> str:
            if work_dir in full_filePath:
                full_filePath = full_filePath[len(work_dir):]
                if full_filePath[0] == '/':
                    full_filePath = full_filePath[1:]
            return full_filePath
        
        groupedMutants = PitMutantProcessor.getGroupedLivedMutants(mutation_file)
        if further:
            groupedMutantsByLandM = PitMutantProcessor.getGroupedByLineAndMutator(mutation_file)
        else:
            groupedMutantsByLandM = None
        ret_mut_lr_pairs = {}
        for i, (filePath, mutants_infile) in enumerate(groupedMutants.items()):
            idx_mut = 0
            if len(mutants_infile) == 0: continue 
            full_filePath = getFullPath(fullPathMutatedFiles, filePath)
            if work_dir not in full_filePath:
                full_filePath = os.path.join(work_dir, full_filePath)
            # drop common dirpath
            core_filePath = get_core_path(full_filePath, work_dir)
            ret_mut_lr_pairs[core_filePath] = {}
            codeTree, file_content, start_end_pos_pline = PitMutantProcessor.prepare(work_dir, full_filePath)
            from tqdm import tqdm 
            failed_to_locate = {}
            for i, mutant in enumerate(tqdm(mutants_infile)):
                formatted_mut = PitMutantProcessor.process(
                    codeTree, 
                    file_content, start_end_pos_pline, 
                    mutant, 
                    groupedMutantsByLandM = groupedMutantsByLandM
                )   
                if formatted_mut is None: 
                    op = mutant.find("mutator").text
                    op_name = op.split(".")[-1]
                    try:
                        failed_to_locate[op_name].append(mutant)
                    except KeyError:
                        failed_to_locate[op_name] = [mutant]
                    idx_mut += 1
                    continue
                mut_target_text = file_content[formatted_mut['pos'][1] - 1:formatted_mut['pos'][2]]
                formatted_mut['text'] = mut_target_text 
                if ('isNegConstant' in formatted_mut.keys()) and formatted_mut['isNegConstant']:
                    formatted_mut['text'] = ("-", formatted_mut['text']) 
                ret_mut_lr_pairs[core_filePath][idx_mut] = formatted_mut
                idx_mut += 1 
            # logging 
            print (f"Out of {idx_mut} mutants, failed to process:")
            cnt_total_failed = 0
            for k,v in failed_to_locate.items():
                #print(f"\t{k}: {len(v)}")
                cnt_total_failed += len(v)
            print (f"\tTotal: {cnt_total_failed}")
            # if it is more than 
            if cnt_total_failed > len(mutants_infile) * 0.8 and len(mutants_infile) > 10: 
                print (f"{cnt_total_failed} vs {len(mutants_infile)}: something is wrong")
                print ("\t", failed_to_locate)
                #assert False # will keep goining anyway
            if len(ret_mut_lr_pairs[core_filePath]) == 0:
                del ret_mut_lr_pairs[core_filePath]
        return ret_mut_lr_pairs

    @staticmethod
    def process(
        codeTree:ET.ElementTree, 
        file_content:str, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ) -> Dict:
        """
        return formatted 
        """
        op = mutant.find("mutator").text
        op_name = op.split(".")[-1] # e.g., 'org.pitest.mutationtest.engine.gregor.mutators.MathMutator':
        #print (op, op_name)
        if op_name == 'MathMutator':
            return PitMutantProcessor.math(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'ConditionalsBoundaryMutator':
            return PitMutantProcessor.conditionalBoundary(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'IncrementsMutator':
            return PitMutantProcessor.increments(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'InvertNegsMutator':
            return PitMutantProcessor.invertNegatives(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'NegateConditionalsMutator':
            return PitMutantProcessor.negateConditionals(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'VoidMethodCallMutator':
            return PitMutantProcessor.voidMethodCalls(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'PrimitiveReturnsMutator':
            return PitMutantProcessor.primitiveReturns(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'EmptyObjectReturnValsMutator':
            return PitMutantProcessor.emptyReturns(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'BooleanFalseReturnValsMutator':
            return PitMutantProcessor.falseReturns(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'BooleanTrueReturnValsMutator':
            return PitMutantProcessor.trueReturns(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'NullReturnValsMutator':
            return PitMutantProcessor.nullReturns(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
    
    @staticmethod
    def getMutator(mutant:ET.Element) -> str:
        return mutant.find('mutator').text
    
    @staticmethod
    def getDescription(mutant:ET.Element) -> str:
        return mutant.find('description').text

    @staticmethod 
    def getLineNumber(mutant:ET.Element) -> int:
        return int(mutant.find('lineNumber').text)

    @staticmethod
    def getMutatedClass(mutant:ET.Element) -> str:
        return mutant.find('mutatedClass').text

    @staticmethod 
    def getIndex(mutant:ET.Element) -> int:
        """
        return block and index
        """
        index = int(mutant.find('index').text)
        return index

    @staticmethod 
    def getParentAndLeftAndRight(
        start_e:ET.Element, target_pos:int, target_length:int, 
        parent_e:ET.Element, 
        idx_to_start:int, 
        has_found:bool
    ) -> Tuple[ET.Element, ET.Element, ET.Element, ET.Element]:  
        try:
            pos = int(start_e.attrib['pos'])
            length = int(start_e.attrib['length'])
            if pos == target_pos and length == target_length:
                has_found = True 
        except Exception: 
            pass 

        if has_found:
            if idx_to_start > 0:
                left_e = parent_e[idx_to_start - 1]
            else:
                left_e = None 
            if len(parent_e) > idx_to_start + 1:
                right_e = parent_e[idx_to_start + 1]
            else:
                right_e = None     
            return (parent_e, start_e, left_e, right_e) # the previous one (the parent) amnd 
        else:
            ret = None
            for i, e in enumerate(start_e):
                ret = PitMutantProcessor.getParentAndLeftAndRight(
                    e, target_pos, target_length, 
                    start_e, 
                    i, 
                    has_found)
                if ret is not None: # found! -> no need to look more
                    break 
            return ret 

    @staticmethod
    def conditionalBoundary(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        """
        find the location of the targeted relational operators and generate the 
        formatted for the replacement
        -> targeted opereators: <, <=, >, >=
        """
        # CONDITIONALS_BOUNDARY
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        targeted_ops = set(['<', '<=', '>', '>='])

        # find the location
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            ##
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR"]
            )
            if found_es is None: return None
        #print ("found", found_es, lineNumber)
        targets = []
        for found_e, e_type in found_es:
            _op_str = found_e.attrib['label']
            if _op_str in targeted_ops:
                targets.append((found_e, e_type))
        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                target, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: CONDITIONALS_BOUNDARY")
                    return None 
        elif len(targets) == 1:
            target, _ = targets[0]
        else: # len(targets) == 0: 
            print (f"failed due to no-match...(CONDITIONALS_BOUNDARY)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        
        org_op = target.attrib['label']
        if org_op == '<':
            new_op = "<="
        elif org_op == '<=':
            new_op = "<"
        elif org_op == ">":
            new_op = ">="
        elif org_op == ">=":
            new_op = ">"
        else:
            print (f"Something is wrong: {org_op}: {lineNumber}", target)
            return None 
                   
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target) # target => to be switched
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('CONDITIONALS_BOUNDARY', description)
        return formatted 

    @staticmethod 
    def increments(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        # INCREMENTS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "Changed increment from ([-0-9]+) to ([-0-9]+)\s*$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: INCREMENTS: {description}")
            return None 
        else:
            org_op = int(matched.groups()[0]) # -1 or 1
        should_be = "++" if org_op > 0 else "--"
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["POSTFIX_EXPRESSION_OPERATOR", 'PREFIX_EXPRESSION_OPERATOR'], 
            label = should_be
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["POSTFIX_EXPRESSION_OPERATOR", 'PREFIX_EXPRESSION_OPERATOR'], 
                label = should_be   
            )
            if found_es is None: return None
        targets = found_es
        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                target, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: INCREMENTS")
                    return None 
        elif len(targets) == 1:
            target, _ = targets[0]
        else:
            print (f"failed due to no-match...(INCREMENTS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
    
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target) # increment or decrement operator
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt) 
        formatted['left'] = should_be
        formatted['right'] = "--" if should_be == "++" else "--" 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('INCREMENTS', description)
        return formatted

    @staticmethod
    def invertNegatives(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        # INVERT_NEGS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["PREFIX_EXPRESSION_OPERATOR"] # since 
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["PREFIX_EXPRESSION_OPERATOR"] # since 
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                target, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: INVERT_NGES")
                    return None
        elif len(found_es) == 1:
            target, _ = found_es[0]
        else:
            print (f"failed due to no-match...(INVERT_NGES)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None
        #
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = '-'
        formatted['right'] = "" # nop
        formatted['targeted'] = target 
        formatted['mutOp'] = ('INVERT_NEGS', description)
        return formatted

    @staticmethod
    def math(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "Replaced ([a-zA-Z0-9\s]+) with ([a-zA-Z0-9\s]+)\s*$"
        matched = re.match(pat, description)
        op_str_dict = {
            'addition':"+", 
            'subtraction':"-", 
            'multiplication':"*", 
            'division':"/", 
            'modulus':"%",
            'OR': "|", 'AND':"&", 'XOR':"^", 
            'Shift Left': "<<", 'Shift Right':">>", 'Unsigned Shift Right':">>>"
        }
        def convert(op_str) -> Tuple[str,str]:
            op_ts = op_str.split(" ")
            if len(op_ts) == 2:
                datatype, _op_str = op_ts 
                try:
                    op = op_str_dict[_op_str]
                except KeyError: # Shift Left, Shift Right
                    op = op_str_dict[op_str]
            else:# len(op_ts) == 1: # XOR, Unsigned Shift Right
                datatype = None
                op = op_str_dict[op_str] # 
            return op, datatype

        def checkAndGetOpsAndNewOps(found_es, org_op, new_op):
            targets = [] # the same line, the same operator
            org_op_new_op_pairs = []
            for found_e, e_type in found_es:
                _op_str = found_e.attrib['label']
                if e_type == "ASSIGNMENT_OPERATOR": # e.g., += 
                    if org_op + "=" == _op_str:
                        targets.append(found_e)
                        org_op_new_op_pairs.append([org_op + "=", new_op + "="])
                elif e_type == 'INFIX_EXPRESSION_OPERATOR':
                    if _op_str == org_op:
                        targets.append(found_e)
                        org_op_new_op_pairs.append([org_op, new_op])
                else: # POSTFIX_EXPRESSION_OPERATOR -> either ++ or --
                    if _op_str[0] == org_op:
                        targets.append(found_e)
                        org_op_new_op_pairs.append([org_op + org_op, new_op + new_op])
            return targets, org_op_new_op_pairs
        
        if not bool(matched):
            return None 
        else:
            org = matched.groups()[0]
            org_op, datatype = convert(org)
            new = matched.groups()[1]
            new_op, _ = convert(new)

        # find the location 
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
        )
        if found_es is None: return None
        targets, org_op_new_op_pairs = checkAndGetOpsAndNewOps(found_es, org_op, new_op)
        if len(targets) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
            )
            if found_es is None: return None
            targets, org_op_new_op_pairs = checkAndGetOpsAndNewOps(found_es, org_op, new_op)
        if len(targets) > 1: # then, we cannot process this 
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                target, idx_to_target = getElementByEncounterOrder(
                    targets,
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: MATH")
                    return None
                org_op_new_op_pair = org_op_new_op_pairs[idx_to_target]
        elif len(targets) == 1:
            target = targets[0]
            org_op_new_op_pair = org_op_new_op_pairs[0]
        else:
            print (f"failed due to no-match...(MATH)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op_new_op_pair[0]
        formatted['right'] = org_op_new_op_pair[1] 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('MATH', description)
        return formatted 

    @staticmethod
    def negateConditionals(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        """
        from Doc: "This mutator overlaps to a degree with the conditionals boundary mutator, 
        but is less stable i.e these mutations are generally easier for a test suite to detect."
        """
        # NEGATE_CONDITIONALS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        targeted_ops = set(['==', '!=', '<=', '>=', '<', ">"])
        # find the location 
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR"]
            )
            if found_es is None: return None
        targets = []
        for found_e, e_type in found_es:
            _op_str = found_e.attrib['label']
            if _op_str in targeted_ops:
                targets.append((found_e, e_type))

        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                target, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: NEGATE_CONDITIONALS")
                    print (f"LineNumer: {lineNumber}")
                    print (f"Descrip: {description}")
                    return None
        elif len(targets) == 1:
            target, _ = targets[0]
        else:
            print (f"...failed due to no-match...(NEGATE_CONDITIONALS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        org_op = target.attrib['label']
        if org_op == '==':
            new_op = "!="
        elif org_op == '!=':
            new_op = "=="
        elif org_op == "<=":
            new_op = ">"
        elif org_op == ">=":
            new_op = "<"
        elif org_op == "<":
            new_op = ">="
        elif org_op == ">":
            new_op = "<="
        else:
            print (f"Something is wrong, not sure: {org_op}: {lineNumber}", target)
            return None
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('NEGATE_CONDITIONALS', description)
        return formatted
    
    @staticmethod
    def voidMethodCalls(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str, 
        groupedMutantsByLandM:Dict = None 
    ):
        # VOID_METHOD_CALLS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "removed call to ([a-zA-Z0-9\/]+)::(.*)$" # originally from MethodCallVisitor 
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: VOID_METHOD_CALLS: {description}")
            return None 
        else:
            method_name = matched.groups()[1]

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["MethodInvocation"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["MethodInvocation"]
            )
            if found_es is None: return None
        # compare with tge method name of each found es
        targets = []
        for e, e_type in found_es:
            simpleName_node = e.find("*[@type = 'SimpleName']")
            if simpleName_node is None: continue
            e_method_name_e = simpleName_node.attrib['label']
            if e_method_name_e == method_name:
                targets.append([e, e_type])
        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                mthInvoc_e, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if mthInvoc_e is None:
                    print ("Mismatch after checking ordering: VOID_METHOD_CALLS")
                    return None
        elif len(targets) == 1:
            mthInvoc_e, _ = targets[0]
        else:
            print (f"failed due to no-match...(VOID_METHOD_CALLS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None         
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(mthInvoc_e)
        method_call_str = file_content[start_chars_cnt - 1:end_chars_cnt] # will not include ';'
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = method_call_str
        formatted['right'] = "" # nothing as it is a remove 
        formatted['targeted'] = mthInvoc_e 
        formatted['mutOp'] = ('VOID_METHOD_CALLS', description)
        return formatted 

    @staticmethod
    def emptyReturns(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element,
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        """
        Integer, Short, Long, Chracter, Float, Double -> 0
        List -> Collections.emptyList()
        java.util.Optional -> Optional.empty()
        java.lang.String -> “”
        java.util.Set -> Collections.emptySet()
        """
        # EMPTY_RETURNS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "replaced ([a-zA-Z]*)\s*return value with ([a-zA-Z0\.\"\'\s]+) for"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: EMPTY_RETURNS {description}")
            return None 
        dataType, replacedVal = matched.groups()
        if not bool(dataType): # i.e., replaced return value with 
            if replacedVal in set(
                ['Collections.emptyList', 'Collections.emptyMap', 'Collections.emptySet', 
                 'Optional.empty', 'Stream.empty']
            ):
                replacedVal += "()"
        
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ReturnStatement"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ReturnStatement"]
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                retStmt_e, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if retStmt_e is None:
                    print ("Mismatch after checking ordering: EMPTY RETURNS")
                    return None 
        elif len(found_es) == 1:
            retStmt_e, _ = found_es[0]
        else:
            print (f"failed due to no-match...(EMPTY RETURNS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        target = retStmt_e[0] # the returned one
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        org_op = file_content[start_chars_cnt - 1:end_chars_cnt] # target.attrib['label']
        new_op = replacedVal
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target 
        formatted['mutOp'] = ('EMPTY_RETURNS', description) 
        return formatted
    

    @staticmethod
    def falseReturns(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element,
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # FALSE_RETURNS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        replacedVal = "false"
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ReturnStatement"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ReturnStatement"]
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                retStmt_e, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if retStmt_e is None:
                    print ("Mismatch after checking ordering: FALSE_RETURNS")
                    return None 
        elif len(found_es) == 1:
            retStmt_e, _ = found_es[0]
        else:
            print (f"failed due to no-match...(FALSE_RETURNS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None
        target = retStmt_e[0]
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        org_op = file_content[start_chars_cnt-1:end_chars_cnt] # target.attrib['label']
        new_op = replacedVal
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target 
        formatted['mutOp'] = ('FALSE_RETURNS', description) 
        return formatted

    @staticmethod
    def trueReturns(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element,
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # TRUE_RETURNS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        replacedVal = "true"

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ReturnStatement"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ReturnStatement"]
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                retStmt_e, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if retStmt_e is None:
                    print ("Mismatch after checking ordering: TRUE_RETURNS")
                    return None 
        elif len(found_es) == 1:
            retStmt_e, _ = found_es[0]
        else:
            print (f"failed due to no-match...(TRUE_RETURNS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        target = retStmt_e[0]
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        org_op = file_content[start_chars_cnt - 1:end_chars_cnt] #target.attrib['label']
        new_op = replacedVal
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target 
        formatted['mutOp'] = ('TRUE_RETURNS', description) 
        return formatted

    @staticmethod
    def nullReturns(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element,
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # NULL_RETURNS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        replacedVal = "null"
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ReturnStatement"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ReturnStatement"]
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                retStmt_e, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if retStmt_e is None:
                    print ("Mismatch after checking ordering: NULL RETURNS")
                    return None 
        elif len(found_es) == 1:
            retStmt_e, _ = found_es[0]
        else:
            print (f"failed due to no-match...(NULL RETURNS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        target = retStmt_e[0]
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        org_op = file_content[start_chars_cnt - 1:end_chars_cnt] #target.attrib['label']
        new_op = replacedVal
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target 
        formatted['mutOp'] = ('NULL_RETURNS', description) 
        return formatted 

    @staticmethod
    def primitiveReturns(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element,
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        """
        Replaces int, short, long, char, float and double return values with 0.
        """
        # PRIMITIVE_RETURNS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "replaced.*with ([a-zA-Z0\.]+) for"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: PRIMITIVE_RETURNS: {description}")
            return None 
        replacedVal = matched.groups()[0]
    
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ReturnStatement"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ReturnStatement"]
            )
            if found_es is None: return None
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                retStmt_e, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if retStmt_e is None:
                    print ("Mismatch after checking ordering: PRIMITIVE_RETURNS")
                    return None 
        elif len(found_es) == 1:
            retStmt_e, _ = found_es[0]
        else:
            print (f"failed due to no-match...(PRIMITIVE_RETURNS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        target = retStmt_e[0]
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        org_op = file_content[start_chars_cnt - 1:end_chars_cnt] #target.attrib['label']
        new_op = replacedVal
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target 
        formatted['mutOp'] = ('PRIMITIVE_RETURNS', description)
        return formatted 