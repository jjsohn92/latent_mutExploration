"""
Here, check whether the changes are made on non-semantic part or not 
"""
import os, sys
from typing import List, Dict, Tuple
from utils.java_utils import changeJavaVer, getJavaVersion, setJavaHome
import xml.etree.ElementTree as ET

### new ####
def compareFile(fileA:str, fileB:str) -> bool:
    # used to filter out commit with no meaningfull changes
    contentA = process(fileA) 
    #contentA = [tk for tk in contentA if ]
    contentB = process(fileB)
    # for comparision 
    import re
    contentA = re.sub(r'\s+', ' ', "\n".join(contentA))
    contentB = re.sub(r'\s+', ' ', "\n".join(contentB))
    return contentA == contentB

def compareLine(
    fileA:str, lnoA:int, fileB:str, lnoB:int, 
    computeSimScore:bool = False
):
    # used to compare the lines without 
    #lineA = "".join(process(fileA, lnoA))
    #lineB = "".join(process(fileB, lnoB)) 
    lineA = " ".join(process(fileA, lnoA))
    lineB = " ".join(process(fileB, lnoB)) 
    if computeSimScore:
        import jellyfish
        score = jellyfish.levenshtein_distance(lineA, lineB)
        return (lineA == lineB, score)
    else:
        return lineA == lineB

def process(file:str, lno:int = None) -> List[str]:
    """
    remove all comments and annotation
    """
    tree = initTreeAndRootSRCML(file)
    if lno is None:
        tree = remove_comments_and_annotations(tree)
        tks = list(genAbstractedCode(tree.getroot()))
    else:
        tree = remove_comments_and_annotations(tree)
        tks = list(genAbstractedCodeLine(tree.getroot(), lno))
    ret_lines = postprocess(tks)
    ret_lines = [l for l in ret_lines if bool(l.strip())]
    return ret_lines

def initTreeAndRootSRCML(inputfile:str) -> Tuple[ET.ElementTree, Dict]:
    """
    run srcml & preprocessing of output xml
    both self.xmlTree and self.root are initialised here. 
    """
    import subprocess 
    from subprocess import CalledProcessError
    inputfile = inputfile.replace('$', "\$")
    cmd = f"srcml --position -l Java -X {inputfile}"
    try:
        output = subprocess.check_output(
            cmd, shell = True).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e: 
        print (cmd)
        print (e)
        return None
    tree = ET.ElementTree(ET.fromstring(output))
    for elem in tree.iter():
        _, _, elem.tag = elem.tag.rpartition('}') # strip ns
    return tree

def getElementsAtLnoInOrder(startElement:ET.Element, lno:int):
    start_attrib = '{http://www.srcML.org/srcML/position}start'
    end_attrib = '{http://www.srcML.org/srcML/position}end'
    tag = startElement.tag
    if not isinstance(tag, str) and tag is not None:
        return
    if start_attrib in startElement.attrib.keys():
        start_lno = int(startElement.attrib[start_attrib].split(":")[0])
        end_lno = int(startElement.attrib[end_attrib].split(":")[0])
        if start_lno == lno: # or end_lno == lno:
            yield startElement
    for e in startElement:
        yield from getElementsAtLnoInOrder(e, lno)
    #tag = startElement.tag
    #if not isinstance(tag, str) and tag is not None:
        #return
    #t = startElement.text
    #if t:
        #yield t
    #for e in startElement:
        #yield from genAbstractedCode(e)
        #t = e.tail
        #if t:
            #yield t


def genAbstractedCodeLine(
    startElement:ET.Element, lno:int
):
    start_attrib = '{http://www.srcML.org/srcML/position}start'
    tag = startElement.tag
    if not isinstance(tag, str) and tag is not None:
        return
    t = startElement.text
    if start_attrib in startElement.attrib.keys():
        start_lno = int(startElement.attrib[start_attrib].split(":")[0])
        the_target = start_lno == lno
    else:
        the_target = False
    if t and the_target:
        yield t
    for e in startElement:
        yield from genAbstractedCodeLine(e, lno)
        t = e.tail
        if t and the_target:
            yield t

def remove_comments_and_annotations(tree:ET.ElementTree) -> ET.ElementTree:
    for elem in tree.iter():
        _, _, elem.tag = elem.tag.rpartition('}') # strip ns
    for elem in tree.iter():
        for child in list(elem):
            if child.tag in set(['comment', 'annotation']):
                elem.remove(child)
    return tree

def remove_comments_and_annotations_byList(
    elems:List[ET.Element]
) -> ET.ElementTree:
    for elem in elems:
        _, _, elem.tag = elem.tag.rpartition('}') # strip ns
    ret = []
    for elem in elems:
        if elem.tag not in set(['comment', 'annotation']):
            ret.append(elem)
    return ret

def genAbstractedCode(
    startElement:ET.Element,
):
    tag = startElement.tag
    if not isinstance(tag, str) and tag is not None:
        return
    t = startElement.text
    if t:
        yield t
    for e in startElement:
        yield from genAbstractedCode(e)
        t = e.tail
        if t:
            yield t

def genAbstractedCode_byList(elems:List[ET.Element]) -> List[str]:
    ret = []
    for e in elems:
        t = e.text
        if t: 
            ret.append(t)
            #print ("t", t, 'end')
        tail = e.tail 
        if tail: 
            #print ("tail", tail, 'end')
            ret.append(tail)
    return ret
            
def postprocess(tks:List[str]) -> List[str]:
    """
    used before comparing the processed lines
    """
    import re
    raw_out = "".join(tks)
    ret_lines = []
    for line in raw_out.split("\n"):
        line = line.strip()
        if not bool(line):
            continue
        line = re.sub(r'\s+', ' ', line)
        ret_lines.append(line)
    return ret_lines 


