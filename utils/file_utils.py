from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
import os

def readFile(fpath:str) -> str:
    content = ""
    with open(fpath, encoding ='UTF-8', errors = 'backslashreplace') as f:
        #for line in f.readlines():
        content += "".join(f.readlines())
    return content

def compute_start_end_pos_pline(content:str) -> Dict:
    """
    Return (dict):
        key = a line number,
        value = (start_of_line, end_of_line)
    start from 1
    """
    lines = content.split("\n")
    start_end_pos_pline = {}
    pos = 1 # starting from 1 (so, position, not index)
    for i, line in enumerate(lines):
        lno = i + 1
        l_size = len(line) #+ 1
        start_end_pos_pline[lno] = (pos, pos + l_size - 1)
        pos += l_size + 1 # update (+1 due to \n)
    return start_end_pos_pline

def readTargetFiles(targetFile:str):
    afiles = []
    with open(targetFile) as f:
        for afile in f.readlines():
            afile = afile.strip()
            if bool(afile): afiles.append(afile)
    return afiles

def copydir(src:str, dest:str):#, work_dir:str):
    import subprocess
    from subprocess import CalledProcessError

    if os.path.exists(dest):
        import shutil  
        shutil.rmtree(dest) # delete 
    cmd = f"cp -Rf {src} {dest}" # the safest 
    try:
        _ = subprocess.run(cmd, shell=True)# cwd = work_dir)
    except CalledProcessError as e:
        print (e)
        assert False 
    
def fileWrite(content:str, filePath:str):
    with open(filePath, 'w') as f:
        f.write(content)

def getSrcDir(repo_path:str) -> Tuple[str, str]: 
    # will be used for those not puresly d4j-based
    root_dir = None
    for root, dirs, _ in os.walk(repo_path):
        for dir in dirs:
            if dir == 'src':
                root_dir = root 
                break 
    assert root_dir is not None, repo_path
    if os.path.exists(os.path.join(root_dir, "src/main/java")):
        return root_dir, "src/main/java"
    elif os.path.exists(os.path.join(root_dir, "src/java")):
        return root_dir, "src/java"
    else:
        print (f"Something is wrong ...: {root_dir}")
        assert False


#start_attrib = '{http://www.srcML.org/srcML/position}start'
#end_attrib = '{http://www.srcML.org/srcML/position}end'
#
#def getPos(e:ET.Element):
    #e.attrib[start_attrib]
    #e.attrib[end_attrib] 
    #pass 
#
#def parse(fileContent:str, workdir:str):
    #import subprocess 
    #from subprocess import CalledProcessError
#    
    #with open("temp.java", 'w') as f:
        #f.write(fileContent)
    #outputfile = "temp.xml"
    #cmd = f"srcml --position -l Java -X temp.java > " + outputfile
    #try:
        #_ = subprocess.run(cmd, cwd = workdir, shell = True)
        ##output = subprocess.check_output(
        ##    cmd, cwd = work_dir, shell = True).decode('utf-8', 'backslashreplace')
    #except CalledProcessError as e: # ... this actually will also catch a simple test failure
        #print (cmd)
        #print (e)
        #return None
    #tree = ET.parse(outputfile)
    #for elem in tree.iter():
        #_, _, elem.tag = elem.tag.rpartition('}') # strip ns
    #import shutil
    #shutil.rmtree('temp.java')
    #shutil.rmtree('temp.xml')
    #return tree
#
#def genAbstractedCode(startElement:ET.Element):
    #"""
    #generate 
    #"""
    #tag = startElement.tag
    #if not isinstance(tag, str) and tag is not None:
        #return
    #t = startElement.text
    #if t: yield t
    #for e in startElement:
        #yield from genAbstractedCode(e)
        #t = e.tail 
        #if t:
            #yield t
#
#def initTreeAndRootSRCML(inputfile:str, work_dir:str) -> Tuple[ET.ElementTree, Dict]:
    #"""
    #run srcml & preprocessing of output xml
    #both self.xmlTree and self.root are initialised here. 
    #"""
    #import subprocess 
    #from subprocess import CalledProcessError
    #inputfile = inputfile.replace('$', "\$")
    #if os.path.dirname(inputfile) == work_dir:
        #inputfile = os.path.basename(inputfile)
    #outputfile = os.path.join(work_dir, inputfile) + ".xml"
    #cmd = f"srcml --position -l Java -X {inputfile} > {outputfile}"
    #try:
        ##output = subprocess.check_output(
        ##    cmd, cwd = work_dir, shell = True).decode('utf-8', 'backslashreplace')
        #_ = subprocess.run(cmd, cwd = work_dir, shell = True)
    #except CalledProcessError as e: # ... this actually will also catch a simple test failure
        #print (cmd)
        #print (e)
        #return None
#        
    #tree = ET.parse(outputfile.replace("\$", "$"))
    #for elem in tree.iter():
        #_, _, elem.tag = elem.tag.rpartition('}') # strip ns
    #inputfile = os.path.join(work_dir, inputfile)
    #start_end_pos_pline = compute_start_end_pos_pline(readFile(inputfile))
    #return tree, start_end_pos_pline
