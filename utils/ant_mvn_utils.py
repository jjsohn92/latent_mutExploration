import subprocess
from subprocess import CompletedProcess, TimeoutExpired, CalledProcessError
from typing import List, Dict, Tuple 
import os, sys, glob
import shutil 

ANT_HOME = os.getenv("ANT_HOME")

def add_reportdir_to_batchtest():
    pass

def check_runstate(out:CompletedProcess, surpress:bool = False) ->bool:
    if out.returncode == 0:
        return True 
    elif surpress:
        print (f"Stderr: {out.stderr}")
        return True 
    else:
        print (f"Stderr: {out.stderr}")
        return False

def mvn_call(repo_path:str, cmd:str, timeout:int, *args):
    cmd = ["mvn", cmd] + list(args)
    cmd = " ".join(cmd)
    import time 
    t1 = time.time()
    out = subprocess.run(
        cmd, 
        shell = True, 
        cwd = repo_path, 
        capture_output = True, 
        timeout = timeout
    )
    t2 = time.time()
    #if out.returncode != 0:
    if (timeout is not None) and ((t2 - t1) > timeout):
        raise TimeoutExpired(
            cmd = cmd, 
            timeout = timeout, 
            output = out.stdout, 
            err = out.stderr 
        )
    #else:
        #raise CalledProcessError(
            #returncode = out.returncode,
            #cmd = cmd, 
            #stdout = out.stdout, 
            #stderr = out.stderr
        #)
    return out, cmd

### need to complete
def ant_call(repo_path:str, cmd:str, timeout:int, *args) -> bool:
    #ant_file = os.path.join(repo_path, "build.xml")
    cmd = [f"{ANT_HOME}/ant", cmd] + list(args)
    cmd = " ".join(cmd)
    import time 
    t1 = time.time()
    out = subprocess.run(
        cmd, 
        shell = True, 
        cwd = repo_path, 
        capture_output = True, 
        timeout = timeout)
    t2 = time.time()
    #if bool(out.stderr) or out.returncode != 0:
    if (timeout is not None) and ((t2 - t1) > timeout):
        raise TimeoutExpired(
            cmd = cmd, 
            timeout = timeout, 
            output = out.stdout, 
            err = out.stderr 
        )
    #else:
        #raise CalledProcessError(
            #returncode = out.returncode,
            #cmd = cmd, 
            #stdout = out.stdout, 
            #stderr = out.stderr
        #)
    return out, cmd

## below should be modified
def compile(repo_path:str, prefer:str = None, strict:bool = False) -> bool:
    # strict = True -> if failed to compile with prefer, consider as failure
    mvn_supported = os.path.exists(os.path.join(repo_path, "pom.xml"))
    ant_supported = os.path.exists(os.path.join(repo_path, "build.xml"))
    used = prefer
    cmd = None
    compiled = None
    if mvn_supported and ant_supported:
        if (prefer is None) or prefer == 'mvn': # meaning by default mvn 
            compiled, cmd = mvn_call(repo_path, "compile", None)
            if not strict and compiled.returncode != 0:
                compiled, cmd = ant_call(repo_path, "compile", None)
                used = 'ant'
        elif prefer == 'ant':
            compiled, cmd = ant_call(repo_path, "compile", None)
            if not strict and compiled.returncode != 0:
                compiled, cmd = mvn_call(repo_path, "compile", None)
                used = 'mvn'
        else:
            print (f"Cannot find either mvn or ant: {repo_path}")
            return False, None 
    elif mvn_supported:
        if not strict or (prefer == 'mvn'): 
            compiled, cmd = mvn_call(repo_path, "compile", None)
            used = 'mvn'
        else: # mvn supported but prefer == 'ant' and strict = True
            return False, prefer, None
    elif ant_supported:
        if not strict or (prefer == 'ant'):
            compiled, cmd = ant_call(repo_path, "compile", None)
            used = 'ant'
        else: # ant supported but prefer == 'mvn' and strict = True
            return False, prefer, None
    return compiled.returncode == 0, used, cmd

def test_compile(repo_path:str, prefer:str = None, strict:bool = False) -> bool:
    mvn_supported = os.path.exists(os.path.join(repo_path, "pom.xml"))
    ant_supported = os.path.exists(os.path.join(repo_path, "build.xml"))
    used = prefer
    cmd = None
    compiled = None
    if mvn_supported and ant_supported:
        if (prefer is None) or prefer == 'mvn':
            compiled, cmd = mvn_call(repo_path, "test-compile", None)
            if not strict and compiled.returncode != 0:
                compiled, cmd = ant_call(repo_path, "compile.tests", None)
                used = 'ant'
        elif prefer == 'ant':
            compiled, cmd = ant_call(repo_path, "compile.tests", None)
            if not strict and compiled.returncode != 0:
                compiled, cmd = mvn_call(repo_path, "test-compile", None)
                used = 'mvn'
        else:
            print (f"Cannot find either mvn or ant: {repo_path}")
            return False, None
    elif mvn_supported:
        if not strict or (prefer == 'mvn'): 
            compiled, cmd = mvn_call(repo_path, "test-compile", None)
        else: # mvn supported but prefer == 'ant' and strict = True
            return False, prefer, None
    elif ant_supported:
        if not strict or (prefer == 'ant'): 
            compiled, cmd = ant_call(repo_path, "compile.tests", None)
        else: # ant supported but prefer == 'mvn' and strict = True
            return False, prefer, None
    return compiled.returncode == 0, used, cmd

def export_SourceDirs(repo_path:str,  prefer:str = None) -> str:
    mvn_supported = os.path.exists(os.path.join(repo_path, "pom.xml"))
    ant_supported = os.path.exists(os.path.join(repo_path, "build.xml"))
    used = prefer
    cmd = None
    if mvn_supported and ant_supported:
        if (prefer is None) or prefer == 'mvn':
            compiled, cmd = mvn_call(repo_path, "test-compile", None)
            if compiled.returncode != 0:
                compiled, cmd = ant_call(repo_path, "compile.tests", None)
                used = 'ant'
        elif prefer == 'ant':
            compiled, cmd = ant_call(repo_path, "compile.tests", None)
            if compiled.returncode != 0:
                compiled, cmd = mvn_call(repo_path, "test-compile", None)
                used = 'mvn'
        else:
            print (f"Cannot find either mvn or ant: {repo_path}")
            return False, None
    elif mvn_supported:
        compiled, cmd = mvn_call(repo_path, "test-compile", None)
    elif ant_supported:
        compiled, cmd = ant_call(repo_path, "compile.tests", None)
    return compiled.returncode == 0, used, cmd


def addExportTestClassPath_ant(repo_path:str) -> str:
    # test.classpath -> property to search 
    pom_file = "..."
    pass 

def export_SourceDirs_mvn(repo_path:str) -> str:
    # either src/java or src/main/java 
    if os.path.exists(os.path.join(repo_path, "src/main/java")):
        return "src/main/java"
    elif os.path.exists(os.path.join(repo_path, "src/java")):
        return "src/java" 

def export_SourceDirs_ant(repo_path:str) -> str:
    # the most common 
    if os.path.exists(os.path.join(repo_path, "src/main/java")):
        return "src/main/java"
    elif os.path.exists(os.path.join(repo_path, "src/java")):
        return "src/java" 
    else:
        pass 
