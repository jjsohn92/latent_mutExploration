from typing import List, Dict, Tuple 
import os 

def changeJavaVer(ver:int, use_sdk:bool = False): # better to be always false
    from subprocess import CalledProcessError
    try:
        if not use_sdk:
            os.environ['JAVA_HOME'] = os.getenv(f"JAVA_{ver}_HOME")
        else:
            import subprocess
            if ver == 8:
                cmd = "sdk default java 8.0.362-tem"
                _ = subprocess.run(cmd, shell= True)
            elif ver == 17:
                cmd = "sdk default java 17.0.7-tem"
                _ = subprocess.run(cmd, shell= True)
            else:
                print (f"Wrong version: {ver}")
                assert False 
    except (Exception, CalledProcessError) as e:
        print (e)
        print (f"Set JAVA_{ver}_HOME variable")
        assert False

def getTestClassPats(targetFiles:List, commonGID:str) -> Dict[str,str]:
    """
    Assume to follow the naming convention of java for both source an test files
    commonGID: e.g., org.apache.commons, org.joda, com.google (... )
    """
    testClassPatterns = {}
    for targetFile in targetFiles:
        subdirs = os.path.dirname(targetFile).replace("/", ".")
        firstE = commonGID.split(".")[0]
        parent = subdirs[subdirs.index(firstE):]      
        classname = os.path.basename(targetFile)[:-len(".java")]
        targetClass = parent + "." + classname 
        testClassPat = targetClass + "*"
        testClassPatterns[targetFile] = testClassPat # single
    return testClassPatterns

def getTestClassPats_d4j(d4j_home:str, project:str, bid:int) -> str:
    """
    -> reference defects4j's already-idenentified relevant test list 
    """
    relevant_test_file = os.path.join(
        d4j_home, f"framework/projects/{project}/relevant_tests/{bid}"
    )
    testClassPatterns = []
    with open(relevant_test_file) as f:
        for test in f.readlines():
            test = test.strip()
            if not bool(test): continue 
            # will take away the test part 
            test = "*" + test.split(".")[-1] #+ "*"
            testClassPatterns.append(test)
    merged_testClassPatterns = ",".join(testClassPatterns)
    return merged_testClassPatterns

def getFullTestClasses(testPat:str, test_classes_dir:str) -> str:
    # test_classes_dir -> assume to be absolute
    from pathlib import Path
    testclasses = []
    if test_classes_dir.endswith("/"): test_classes_dir = test_classes_dir[:-1]
    for a_testpat in testPat.split(","):
        if a_testpat.endswith("*"): a_testpat = a_testpat[:-1]
        if a_testpat.startswith("*"): a_testpat = a_testpat[1:]
        if not bool(a_testpat): continue 
        for path in Path(test_classes_dir).rglob(f"{a_testpat}.class"):
            path = str(path)
            path = path.replace(test_classes_dir, "**")
            if path.startswith("/"): path = path[1:]
            testclasses.append(path)
    return ",".join(testclasses)

#def writeTargetTests(
    #work_dir:str, testPat:str, use_full_path:bool = False, test_dir:str = None
#) -> str:
    ## testPat: expected to be concatentated with comma 
    #targetTestClassesFile = os.path.join(work_dir, "temp.test.properties")
    #to_write = []
    #for atestPat in testPat.split(","):
        #if not use_full_path:
            #atestPat = os.path.basename(atestPat) 
            #full_atestPat = os.path.join("**", atestPat)
        #else:
            #atestPat = atestPat.replace(".", "/") + '.class'
            #if test_dir.endswith("/"): test_dir = test_dir[:-1]
            #full_atestPat = atestPat.replace(test_dir, "**") # **/org/.... *.class
        #to_write.append(full_atestPat)
    #with open(targetTestClassesFile, 'w') as f:
        #to_write_str = ",".join(to_write)
        #f.write(f"target.test.classes={to_write_str}")
    #return targetTestClassesFile 
def writeTargetTests(work_dir:str, testPat:str) -> str:
    # testPat: expected to be concatentated with comma 
    targetTestClassesFile = os.path.join(work_dir, "temp.test.properties")
    with open(targetTestClassesFile, 'w') as f:
        f.write(f"target.test.classes={testPat}")
    return targetTestClassesFile 

def getTargetClasses_d4j(
    d4j_home:str, project:str, bid:int, targetedFiles:List[str]
) -> str:
    """
    -> reference defects4j's already-idenentified relevant test list 
    """
    relevant_test_file = os.path.join(
        d4j_home, f"framework/projects/{project}/modified_classes/{bid}.src"
    )
    targetClasses = []
    basenames = [os.path.basename(afile)[:-5] for afile in targetedFiles]
    def is_target(basenames, _class):
        ts = set(_class.split("."))
        for basename in basenames:
            if basename in ts:
                return True 
        return False 
    
    with open(relevant_test_file) as f:
        for _class in f.readlines():
            _class = _class.strip()
            if not bool(_class): continue
            if not is_target(basenames, _class): # not in one of the targeted files
                continue
            _class.replace('$', "\$")
            targetClasses.append(_class)
    merged_targetClasses = ",".join(targetClasses)
    return merged_targetClasses


# Try to skip those related to locale and time -> we will likely get additional errors due to them
def selectTargetFiles():
    pass 

def getJavaVersion():
    return os.environ['JAVA_HOME']

def setJavaHome(java_home):
    os.environ['JAVA_HOME'] = java_home
