import subprocess
from subprocess import TimeoutExpired
import time 
import os 

ANT_HOME = os.getenv("ANT_HOME")

def ant_d4jbased_call(
    d4j_home:str, project:str, 
    repo_path:str, cmd:str, timeout:int, 
    *args
):
    """
    all pathes should be absolute path 
    """
    build_fpath = os.path.join(d4j_home, "framework/projects/defects4j.build.ext.xml")
    full_cmd = f"{ANT_HOME}/ant -f {build_fpath} -Dd4j.home={d4j_home} -Dd4j.dir.projects={d4j_home}/framework/projects" 
    full_cmd += f" -Dd4j.project.id={project} -Dbasedir={repo_path}"
    t1 = time.time()
    if args is not None:
        args = list(args)
    else:
        args = []
    full_cmd += " " + cmd + " " + " ".join(list(args))
    print (full_cmd)
    out = subprocess.run(
        full_cmd,
        shell = True, 
        cwd = repo_path, 
        capture_output=True, # will automatically surpress rasing exception
        timeout = timeout
    )
    t2 = time.time()
    if (timeout is not None) and ((t2 - t1) > timeout):
        print (f"Timeout while running {cmd}")
        raise TimeoutExpired(
            cmd = full_cmd, 
            timeout = timeout, 
            output = out.stdout, 
            err = out.stderr 
        )
    return out, full_cmd

def export_property(
    d4j_home:str, project:str, repo_path:str, cmd:str
) -> str:
    exported, _ = ant_d4jbased_call(d4j_home, project, repo_path, cmd, None)
    output = exported.stdout.decode('utf-8', 'backslashreplace')
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("[echo]"):
            outs = line.split("[echo]")[1].strip().replace(":", ",")
            return outs  
    return None 

def export_SourceDirs_d4jbased(d4j_home:str, project:str, repo_path:str) -> bool:
    outs = export_property(d4j_home, project, repo_path, "export.source.home")
    return outs 

def check_set_SourceDirs_d4jbased(d4j_home:str, project:str, repo_path:str) -> bool:
    outs = export_property(d4j_home, project, repo_path, "has.source.home")
    return True if outs == 'true' else False 

def export_TestDir_d4jbased(d4j_home:str, project:str, repo_path:str) -> bool:
    outs = export_property(d4j_home, project, repo_path, "export.test.home")
    return outs 

def check_set_TestDir_d4jbased(d4j_home:str, project:str, repo_path:str) -> bool:
    outs = export_property(d4j_home, project, repo_path, "has.test.home")
    return True if outs == 'true' else False 

def export_TestClassesDir_d4jbased(d4j_home:str, project:str, repo_path:str) -> bool:
    outs = export_property(d4j_home, project, repo_path, "export.test.classes.dir")
    return outs 

def export_compileClassPath_d4jbased(d4j_home:str, project:str, repo_path:str) -> str:
    outs = export_property(d4j_home, project, repo_path, "export.compile.classpath")
    return outs 

def export_compileTestClassPath_d4jbased(d4j_home:str, project:str, repo_path:str) -> str:
    outs = export_property(d4j_home, project, repo_path, "export.test.classpath")
    return outs 

def compile(d4j_home:str, project:str, repo_path:str, *args) -> bool:
    compiled, cmd = ant_d4jbased_call(d4j_home, project, repo_path, "compile", None, *args)
    return compiled.returncode == 0, cmd

def test_compile(d4j_home:str, project:str, repo_path:str, *args) -> bool:
    test_compiled, cmd = ant_d4jbased_call(d4j_home, project, repo_path, "compile.tests", None, *args)
    return test_compiled.returncode == 0, cmd

def run_tests(d4j_home:str, project:str, repo_path:str, timeout:int, *args) -> bool:
    running_tests, cmd = ant_d4jbased_call(d4j_home, project, repo_path, "run.dev.tests", timeout, *args)
    #return (running_tests.returncode == 0, running_tests)
    return running_tests.returncode == 0, cmd


def run_all_tests(d4j_home:str, project:str, repo_path:str, timeout:int, *args) -> bool:
    running_tests, cmd = ant_d4jbased_call(d4j_home, project, repo_path, "run.all.dev.tests", timeout, *args)
    #return (running_tests.returncode == 0, running_tests)
    return running_tests.returncode == 0, cmd
