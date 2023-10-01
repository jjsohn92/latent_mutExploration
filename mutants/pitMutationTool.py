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
from typing import List, Dict, Tuple, Union 
import xml.etree.ElementTree as ET
sys.path.insert(0, "..")
import utils.mvn_utils as mvn_utils
import utils.file_utils as file_utils
import utils.java_utils as java_utils
import utils.git_utils as git_utils
import utils.ant_d4jbased_utils as ant_d4jbased_utils
import utils.ant_mvn_utils as ant_mvn_utils
import utils.gumtree as gumtree
import re
#
import pandas as pd 
# 

def register_all_namespaces(filename):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])

## will-be-deleted
def addMVNPitDependency(pom_file:str, junit_ver:str = "4.11"):
    """
    1. change to junit 4.11 -> due to errors when running coverage
    2. add pitest-command-line dependency
    """
    register_all_namespaces(pom_file)
    tree, root, tag_to_root = mvn_utils.get_tree_root_and_root_tag(pom_file)
    tag_to_dependencies = "{}dependencies".format(tag_to_root)
    tag_to_dependency = "{}dependency".format(tag_to_root)
    tag_to_artifactId = "{}artifactId".format(tag_to_root)
    tag_to_version = "{}version".format(tag_to_root)
    dependencies_node = tree.find(tag_to_dependencies)
    # modify to junit to have junit_ver
    for depds_node in root.iter(tag_to_dependencies):
         for depd_node in depds_node.iter(tag_to_dependency):
            artifact_nodes = depd_node.findall(tag_to_artifactId)
            if len(artifact_nodes) == 0:
                continue 
            else:
                artifact_node = artifact_nodes[0]
                if artifact_node.text == 'junit':
                    version_nodes = depd_node.findall(tag_to_version)
                    assert len(version_nodes) > 0, depd_node
                    junit_version = version_nodes[0].text 
                    if junit_version != junit_ver:
                        version_nodes[0].text = junit_ver
                    break 
                else:
                    continue
    
    pitest_depedency_str = "    <dependency> \n\
            <groupId>org.pitest</groupId>\n\
            <artifactId>pitest-command-line</artifactId>\n\
            <version>1.14.2</version>\n\
        </dependency>"
    pitest_depedency_node = ET.fromstring(pitest_depedency_str)
    dependencies_node.append(pitest_depedency_node)

    import shutil
    shutil.move(pom_file, pom_file + ".bak") # for backup 
    # overwrite the current pom file
    tree.write(
        pom_file, 
        method='xml',
        xml_declaration=True
    )

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
    print (tag_to_dependencies)
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
        if (_type == 'Block') and (_type != 'ReturnStatement'): # the latter one due to specifically targeting this
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
    #print ("targetetsts", start_e, lno_start_pos, lno_end_pos, only_direct_child, check_itself)
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
    ##
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


def findSpecificTypeElements_v2(
    start_e:ET.Element, targetTypes:List[str],
    lno_start_pos:int, lno_end_pos:int, 
    targets:List
) -> List[Tuple[ET.Element, str]]:
    try:
        _type = start_e.attrib['type']
        if _type in set(targetTypes):
            pos = int(e.attrib['pos']) + 1
            if lno_start_pos is None: # meanig the position does not matter 
                targets.append([e, _type])
            elif (pos >= lno_start_pos) and (pos <= lno_end_pos):
                targets.append([e, _type])
    except KeyError:
        pass    
    
    for i, e in enumerate([start_e] + list(start_e)):
        check_pos = False
        try:
            _type = e.attrib['type']
            check_pos = True 
        except KeyError:
            check_pos = False
        if check_pos:    
            if _type in set(targetTypes):
                pos = int(e.attrib['pos']) + 1
                if lno_start_pos is None: # meanig the position does not matter 
                    targets.append([e, _type])
                elif (pos >= lno_start_pos) and (pos <= lno_end_pos):
                    targets.append([e, _type])
    return targets


def findElement_v1(
    codeTree:ET.ElementTree, 
    start_end_pos_pline:Dict[int, Tuple[int,int]], 
    targetLno:int, 
    targetTypes:List[str], 
) -> List[Tuple[ET.Element, str]]:
    ret_e = None
    targetTypes = set(targetTypes)
    for e in codeTree.iter():
        try:
            index = int(e.attrib['pos'])
        except KeyError:
            continue 
        if isAt(start_end_pos_pline, targetLno, index): # the first met
            ret_e = e
            break 
    targets = []
    if ret_e is None:
        return targets
    else:
        lno_start_pos, lno_end_pos = start_end_pos_pline[targetLno]
        targets = findSpecificTypeElements(
            ret_e, targetTypes, lno_start_pos, lno_end_pos
        )
    return targets 

def findElement_v2(
    codeTree:ET.ElementTree, 
    start_end_pos_pline:Dict[int, Tuple[int,int]], 
    targetLno:int, 
    targetTypes:List[str], 
) -> List[Tuple[ET.Element, str]]:
    # 
    ret_es = []
    targetTypes = set(targetTypes)
    for e in codeTree.iter():
        try:
            index = int(e.attrib['pos'])
        except KeyError:
            continue 
        if isAt(start_end_pos_pline, targetLno, index) and (
            not hasBlock(e)) and (
            not isBlock(e)
        ):
            ret_es.append(e)
        elif isBelow(start_end_pos_pline, targetLno, index):
            break 

    if len(ret_es) == 0:
        return None 
    else:
        #for e in ret_es:
        #    print ("++", e.attrib['type'], hasBlock(e), isBlock(e), e.attrib['pos'], e.attrib['length'])
        targets, covered = [], set([])
        for ret_e in ret_es:
            _targets = findSpecificTypeElements(
                ret_e, targetTypes, None, None, only_direct_child = True 
            )
            for t, t_type in _targets:
                k = t.attrib['pos'] + "-" + t.attrib['length']
                if k in covered:
                    continue
                else:
                    targets.append([t, t_type])
                    covered.add(k)
        return targets 

def findElement_v3(
    codeTree:ET.ElementTree, 
    start_end_pos_pline:Dict[int, Tuple[int,int]], 
    targetLno:int, 
    targetTypes:List[str], 
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
        #for e in ret_es:
        #    print ("++++", e.attrib['type'], hasBlock(e), isBlock(e), e.attrib['pos'], e.attrib['length'])
        targets, covered = [], set([])
        lno_start_pos, lno_end_pos = start_end_pos_pline[targetLno]
        for ret_e in ret_es:
            if isBlock(ret_e): # if it is a block, then need to look whether the child elements 
                #print ("From this", ret_e, ret_e.attrib['type'])
                _targets = findSpecificTypeElements(
                    ret_e, targetTypes, 
                    lno_start_pos, lno_end_pos, 
                    only_direct_child = False, #True, 
                    check_itself = False # for block, we skip 
                )
            else:
                #print ("From this", ret_e, ret_e.attrib['type'])
                _targets = findSpecificTypeElements(
                    ret_e, targetTypes, 
                    None, None, 
                    only_direct_child = False, #True, 
                    check_itself = True
                )
            for t, t_type in _targets:
                k = t.attrib['pos'] + "-" + t.attrib['length']
                if k in covered:
                    continue
                else:
                    targets.append([t, t_type])
                    covered.add(k)
        return targets 
    

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
        #for e in ret_es:
        #    print ("++++", e.attrib['type'], hasBlock(e), isBlock(e), e.attrib['pos'], e.attrib['length'])
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
        #for ret_e in ret_es:
        #    print(ret_e.attrib)
        #print (f"./*[@type = '{list(targetTypes)[0]}']")
        #print (f"./*[@type = '{list(targetTypes)[0]}'][@label = '{label}']")
        targets = []
        for ret_e in ret_es:
            ret_e_type = ret_e.attrib['type']
            if ret_e_type in targetTypes:
                # here, additional label check, -> currently disable more b/c of unhandled issues of "-1" (for inline)
                ##
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
                    #else:
                        #pat = f"[@label = '{label}']" 
                        #if ret_e == ret_e.find(pat):
                            #targets.append(ret_e)
                            #continue # no need to look for the children of ret_e 
                else:
                    targets.append(ret_e)
                    continue
            #else:
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
                    #targets.extend(ret_e.findall(child_pat))
                    targets.extend(matched_es)
            #for c in ret_e: # direct child 
            #    c_type = c.attrib['type']
            #    if c_type in targetTypes:
            #        targets.append(c)
        targets = list(set(targets))
        #print ("in", len(targets))
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
    #print ('length', n_same_key, n_targets)
    if n_same_key != n_targets: # meaning something is missing here, and don't have further information to differentiate
        print (k)
        print ('length', n_same_key, n_targets)
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


#def getParent(
    #startElement:ET.Element, 
    #target_pos:int, target_length:int, target_type:str
#):
    #ret = None
    #for e in startElement:
        #pos = int(e.attrib['pos'])
        #length = int(e.attrib['length'])
        #t_type = e.attrib['type']
        #if pos + length > target_pos + target_length: # out of our scop[e]
            #return None
        #if (t_type == target_type) and (
            #pos == target_pos) and (
            #length == target_length
        #):
            #ret = startElement
            #break # found 
#    
    #if ret is not None:
        #return ret 
    #else:
        ## rec
        #for e in startElement:
            #ret = getParent(e, target_pos, target_length, target_type)
            #if ret is not None:
                #return ret 
        #return ret 

def getChildToParentMaps(codeTree:ET.ElementTree) -> Dict[ET.Element, ET.Element]:
    return {c:p for p in codeTree.iter() for c in p}

def handleNegativeConstant(codeTree:ET.ElementTree, target_e:ET.Element) -> bool:
    # for this we need to check whether its direct preceeding element is PREFIX_EXPRESSION_OPERATOR and label = "-"
    target_e_type = target_e.attrib['type']
    assert target_e_type == 'NumberLiteral', f"This is not NumberLiteral: {target_e.attrib}"
    childAndParentPairs = getChildToParentMaps(codeTree)
    parent_of_target_e = childAndParentPairs[target_e]
    all_child_nodes = list(parent_of_target_e)
    idx_of_target_e = None
    for i,e in enumerate(all_child_nodes):
        if e == target_e:
            idx_of_target_e = i
            break 
    assert idx_of_target_e  is not None, idx_of_target_e.attrib 
    if idx_of_target_e == 0:
        return False 
    else:
        prior_sibling_e = all_child_nodes[i-1]
        is_valid_type = prior_sibling_e.attrib['type'] == 'PREFIX_EXPRESSION_OPERATOR'
        if is_valid_type: # if it is a prefix expression operator
            is_valid_label = prior_sibling_e.attrib['label'] == "-"
            return is_valid_label 
        else:
            return False
            
#def handlePredefConstant(target_e:ET.Element):
    #target_e_type = target_e.attrib['type']
    #assert target_e_type == 'QualifiedName', f"This is not QualifiedName: {target_e.attrib}"
    #pass 

#def checkForMissingInfixOp(
    #file_content:str, 
    #target:ET.Element, 
    #childToParentMaps:Dict[ET.Element, ET.Element]
#) -> ET.Element:
    ## 
    #parent = childToParentMaps[target]
    #p_type = parent.type 
    #assert p_type == 'InfixExpression', f"{p_type} vs InfixExpression"
    #ret_idx = None
    #es = list(parent)
    #for i, e in enumerate(es):
        #if e == target:
            #ret_idx = i
            #break 
    #assert ret_idx is not None, parent 
    #num_es = len(es) # in most cases, will be three 
    #num_following_es = num_es - ret_idx - 1
    #if num_following_es > 1: 
        #missing_infix_ops = []
        #for i in range(num_following_es + 1, num_es, 2):
            #prev_i = i - 1  
            #curr_i = i 
            #_, end_of_prev = getStartAndEndCharCnt(es[prev_i])
            #end_of_prev += 1
            #start_of_curr, _ = getStartAndEndCharCnt(es[curr_i]) 
            #start_of_curr -= 1 # as index 
#            
            #infix_op = file_content[end_of_prev:start_of_curr]
            #prefixes = []
            #for j,c in enumerate(infix_op):
                #if bool(c.strip()): break 
                #prefixes.append(end_of_prev + j)
            #start_of_infix_op = max(prefixes)
            #postfixes = []
            #for j,c in enumerate(infix_op[::-1]):
                #if bool(c.strip()): break 
                #postfixes.append(start_of_curr - j -1)
            #end_of_infix_op = min(postfixes) - 1 # index 
            #length_of_infix_op = end_of_infix_op - start_of_infix_op + 1
            #infix_op_txt = infix_op.strip()
            #n = len(infix_op)
            #msg = f"{infix_op} (vs {file_content[start_of_infix_op:end_of_infix_op + 1]}): {n} vs {length_of_infix_op}"
            #assert n == length_of_infix_op, msg 
            #e_str = f'<tree type="INFIX_EXPRESSION_OPERATOR" label="{infix_op_txt}" pos="{start_of_infix_op}" length="{length_of_infix_op}"></tree>'
            #infix_op_e = ET.fromstring(e_str)
            #missing_infix_ops.append(infix_op_e)
        #return missing_infix_ops
    #else:
        #return None # nothing is missing 

#def getElementByEncounterOrder_new(
    #targets:List[ET.Element], 
    #mutant:ET.Element, 
    #groupedMutantsByLandM:Dict[Tuple[int, str, str], List[ET.Element]],
    #codeTree:ET.ElementTree = None
#) -> Tuple[ET.Element, int]:
    ## targets -> from codeTree (meaning the output of source code parsing)
    ## groupedMutantsByLandM -> contain all mutants regardless of their status
    #mutatedClass = PitMutantProcessor.getMutatedClass(mutant)
    #lineNumber = PitMutantProcessor.getLineNumber(mutant)
    #mutator_op = PitMutantProcessor.getMutator(mutant)
    #description = PitMutantProcessor.getDescription(mutant)    
    ##  
    #k = (mutatedClass, lineNumber, mutator_op, description)
    ### new for missing infix ops
    #for target in targets:
        #t_type = target.attrib['type']
        #if t_type == 'INFIX_EXPRESSION_OPERATOR':
            #...
            #pass 
    #### end new for missing infix ops 
    ##print (k)
    #mutantsWithSameKey = groupedMutantsByLandM[k] # the same line, the same operator
    ##print ("need to look", mutantsWithSameKey)
    #n_same_key = len(mutantsWithSameKey)
    #n_targets = len(targets)
    ##print ('length', n_same_key, n_targets)
    #if n_same_key != n_targets: # meaning something is missing here, and don't have further information to differentiate
        #print (k)
        #print ('length', n_same_key, n_targets)
        #return None, None
    ## get the appearance index of target_index
    #index_of_target_mut = PitMutantProcessor.getIndex(mutant)
    #idx_to_same_appOrder = 0
    #for mut in mutantsWithSameKey:
        #idx = PitMutantProcessor.getIndex(mut)
        #if idx < index_of_target_mut:
            #idx_to_same_appOrder += 1 
    #target = targets[idx_to_same_appOrder]
    #return target, idx_to_same_appOrder


def getStartAndEndCharCnt(e:ET.Element) -> Tuple[int, int]:
    # start from 1 
    start = int(e.attrib['pos']) + 1 # start fr
    end = start + int(e.attrib['length']) - 1# can be same as start
    return start, end 

## new 
# ... -> acutaklly for thi,s only have to consider for ant-d4j and ant (mvn will use the one before)
# -> here, this one is also used to set the timeout -> its first running time 
def getFailingTests(work_dir:str, ant_or_mvn:str, testPat:str) -> List[str]:
    from mutants.mutationTool import parse_test_output 
    if ant_or_mvn == 'ant':
        print ("Not implemented yet") # here, always need to check whether the task "test" have report dir or modify to have one as the attribute in batchtest
        pass 
    elif ant_or_mvn == 'ant_d4j':
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
): # -> List[str]:
    """
    test_class_pats: expect multiple test class patterns to be joined by ","
    """
    import subprocess 
    from subprocess import TimeoutExpired

    src_compiled, test_compiled = True, True
    if ant_or_mvn == 'ant_d4j':
        src_compiled, _compile_cmd = ant_d4jbased_utils.compile(
            kwargs['d4j_home'], kwargs['project'], work_dir)
        if with_test_compile:
            test_compiled, _tst_compile_cmd = ant_d4jbased_utils.test_compile(
                kwargs['d4j_home'], kwargs['project'], work_dir)
    else: # mvn, ant (but for now, mostly mvn)
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
    print ('Ant or mvn', ant_or_mvn)
    if ant_or_mvn == 'ant_d4j':
        #test_dir = ant_d4jbased_utils.export_TestDir_d4jbased(kwargs['d4j_home'], kwargs['project'], work_dir)
        #testClasses = java_utils.getFullTestClasses(test_class_pats, test_dir)
        # here, only timeout exception can be raised
        targetTestClassesFile = java_utils.writeTargetTests(work_dir, test_class_pats)
        failingTestFile = os.path.join(work_dir, "failing_tests")
        tested, _run_test_cmd = ant_d4jbased_utils.run_tests(
            kwargs['d4j_home'], kwargs['project'], 
            work_dir, 
            timeout, 
            f"-DOUTFILE={failingTestFile}", 
            f"-propertyfile {targetTestClassesFile}"
            #"-Dtest.entry.class", test_class_pats, 
        ) 
        #_cmd = "run.dev.tests " + " ".join(["-Dtest.entry.class", test_class_pats])
    else:
        if ant_or_mvn == 'mvn':
            tested, _run_test_cmd = ant_mvn_utils.mvn_call(
                work_dir, "test", timeout, 
                f"-Dtest={test_class_pats}", "-Dmaven.test.failure.ignore=true"
            )
            #_cmd = "test " + " ".join([f"-Dtest={test_class_pats}", "-Dmaven.test.failure.ignore=true"])
        else:
            print ("Not implemented yet")
            #out = ant_mvn_utils.ant_call(work_dir, "test", ["-Dtest.entry="]s)
            _run_test_cmd = None
            pass
    if not bool(tested): # error while testing:
        from subprocess import CalledProcessError
        raise CalledProcessError(
            returncode = 1, #.returncode,
            cmd = _run_test_cmd, 
            stderr = "Error while testing"
        )
    
    # get failing and error tests 
    failed_or_errored_tests = getFailingTests(work_dir, ant_or_mvn, test_class_pats) # test_class_pats -> for mvn, but likely will not be used 
    failed_or_errored_tests = set(failed_or_errored_tests)
    return src_compiled, test_compiled, set(failed_or_errored_tests), set([])


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
            "VOID_METHOD_CALLS", ## complete deletion 
            "EMPTY_RETURNS",
            "FALSE_RETURNS",
            "TRUE_RETURNS",
            "NULL_RETURNS",
            "PRIMITIVE_RETURNS", 
            #"INLINE_CONSTS"
        ], 
        'all':[
            "CONDITIONALS_BOUNDARY",
            "INCREMENTS",
            "INVERT_NEGS",
            "MATH",
            "NEGATE_CONDITIONALS",
            "VOID_METHOD_CALLS", ## complete deletion 
            "EMPTY_RETURNS",
            "FALSE_RETURNS",
            "TRUE_RETURNS",
            "NULL_RETURNS",
            "PRIMITIVE_RETURNS", 
            # 
            #"REMOVE_CONDITIONALS", #- skipped b/c it completely remove the condition and thereby will always end up to the mutated commit, 
            "INLINE_CONSTS", 
            "CONSTRUCTOR_CALLS", # replace with null (not sure) 
            #"NON_VOID_METHOD_CALLS", # temporary excluded as this require to parse a java file to get the replaced value
            "REMOVE_INCREMENTS",
            #"ABS", # negation  -> currently require to furhter process var_cnt (counted per type) to locate the exact mutated location -> skip it for npw
            "AOD", # AOD_1: first one, AOD_2: second one 
            "CRCR", # // overlap with inline
            "OBBN", 
            #"OBBN_1", # reverse the operators: target & and |
            #"OBBN_2", # replae with first memebr
            #"OBBN_3", # replaec with second member
            "ROR", # , 
            "AOR"
            #"UOI" # insertion 
        ]
    } 
    # excluded: 
    #   Remove Conditionals
    #   RETURNS -> might better to be excluded ..
    #   void method calls -> remove entire call .. but not sure 
    #   constructor call mutator 

    #<configuration> \n\
      #<outputFormats>XML</outputFormats> \n\
      #<maxMutationsPerClass>{maxMutationsPerClass}</maxMutationsPerClass> \n\
      #<mutationUnitSize>{mutationUnitSize}</mutationUnitSize> \n\
      #<timeoutConstant>{timeoutConstant}</timeoutConstant>\n\
      #<reportsDirectory>{localReportsDirectory}</reportsDirectory>\n\
      #<threads>{threads}</threads>\n\
      #<maxDependencyDistance>{maxDependencyDistance}</maxDependencyDistance>\n\
    #</configuration> \n\
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
            #cmd += " -DskipFailingTests=true" # -> in configuraton 
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
                + " --skipFailingTests=true" # by setting this to true, failing tests are ignored from computing the status, which is what we wanted
                #+ " --fullMutationMatrix=true" # -> technically, not needed, b
                #+ " --maxMutationsPerClass 1000" \
        return cmd

    @staticmethod
    def getGroupedLivedMutants(mutation_file:str) -> Dict[str, List[ET.Element]]:
        print ("mutation file", mutation_file)
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
        ###
        codeTree = gumtree.addMissingInfixOps(codeTree, file_content, file_content)
        ###
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
        ( compile sources and tests -> expected to be finished)
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
        print (f"Cmd: {cmd}")
        if is_mvn:
            addMVNPitPlugin(os.path.join(work_dir, "pom.xml"))
        mutation_file = PitMutantProcessor.getMutationResultFile(work_dir, is_mvn = is_mvn) # need to save this one
        #print (cmd)
        # run mutation 
        if not os.path.exists(mutation_file):
            ### temporary
            import re
            matched = re.match("^([a-zA-Z]+)([0-9]+)$", os.path.basename(work_dir))
            assert bool(matched), os.path.basename(work_dir)
            _project, _bid = matched.groups()
            _bid = int(_bid)
            active_bugfile = os.path.join(
                os.getenv("D4J_HOME"), f"framework/projects/{_project}/active-bugs.csv")
            df = pd.read_csv(active_bugfile)
            _rev = df.loc[df['bug.id'] == _bid]['revision.id.fixed'].values[0][:8] 
            _mutfile = f"output/pit/final/{_project}/inter/{_rev}/mutations.xml"
            if os.path.exists(_mutfile):
                import shutil
                dirOfMutationFile = os.path.dirname(mutation_file)
                os.makedirs(dirOfMutationFile, exist_ok = True)
                shutil.copyfile(_mutfile, mutation_file)
            ### tempoaray end 
            else:
                # here, timeout may occur
                out = subprocess.run(
                    cmd,
                    shell = True, 
                    cwd = work_dir, 
                    capture_output=True, 
                    timeout = (60 * 60 * 4) # if run more than 6 hours, time-out
                )
                if out.returncode != 0:
                    print (f"Error while mutating {targetClasses} with {targetTests} at {work_dir}")
                    print (out.stdout)
                    print (out.stderr) 
                    import traceback, logging
                    logging.error(traceback.format_exc())
                    # cleanup
                    if is_mvn:
                        restore_pomfile(os.path.join(work_dir, "pom.xml"))
                    assert False
                    #sys.exit()
        # get mutation file 
        # run and get the results
        ret_mut_lr_pairs = PitMutantProcessor.genMutLRPairs(
            work_dir, mutation_file, targetFiles, further = kwargs['further'])
        if is_mvn:
            restore_pomfile(os.path.join(work_dir, "pom.xml"))
        #sys.exit()
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
            #basename = os.path.basename(file)
            for fullPath in fullPaths:
                #_basename = os.path.basename(fullPath)
                #if fullPath.endswith(file) and basename == _basename:
                #    return fullPath
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
        #print (fullPathMutatedFiles)
        #print (groupedMutantsByLandM.keys())
        #sys.exit()
        ret_mut_lr_pairs = {}
        for i, (filePath, mutants_infile) in enumerate(groupedMutants.items()):
            idx_mut = 0
            if len(mutants_infile) == 0: print ("None!!", filePath); continue 
            full_filePath = getFullPath(fullPathMutatedFiles, filePath)
            #full_filePath = filePath
            print ("Full file path", full_filePath, filePath)
            if work_dir not in full_filePath:
                full_filePath = os.path.join(work_dir, full_filePath)
            # get core 
            core_filePath = get_core_path(full_filePath, work_dir)
            #print ("Core", core_filePath)
            ret_mut_lr_pairs[core_filePath] = {}
            codeTree, file_content, start_end_pos_pline = PitMutantProcessor.prepare(work_dir, full_filePath)
            from tqdm import tqdm 
            failed_to_locate = {}
            for i, mutant in enumerate(tqdm(mutants_infile)):
                #if i != 33:
                #    continue 
                formatted_mut = PitMutantProcessor.process(
                    codeTree, 
                    file_content, start_end_pos_pline, 
                    mutant, 
                    groupedMutantsByLandM = groupedMutantsByLandM
                )   
                if formatted_mut is None: 
                    #failed_to_locate += 1
                    op = mutant.find("mutator").text
                    op_name = op.split(".")[-1]
                    try:
                        failed_to_locate[op_name].append(mutant)
                    except KeyError:
                        failed_to_locate[op_name] = [mutant]
                    idx_mut += 1
                    continue

                mut_target_text = file_content[formatted_mut['pos'][1] - 1:formatted_mut['pos'][2]]
                formatted_mut['text'] = mut_target_text ### ------------------------------> check whether this apply for every
                if ('isNegConstant' in formatted_mut.keys()) and formatted_mut['isNegConstant']: # CRCR, INLINE constant
                    formatted_mut['text'] = ("-", formatted_mut['text']) ## 8/19. NOT SURE
                #ret_mut_lr_pairs[full_filePath][idx_mut] = formatted_mut
                ret_mut_lr_pairs[core_filePath][idx_mut] = formatted_mut
                idx_mut += 1 
            
            # logging 
            print (f"Out of {idx_mut} mutants, failed to process:")
            cnt_total_failed = 0
            for k,v in failed_to_locate.items():
                print(f"\t{k}: {len(v)}")
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
        #elif op_name == 'MethodCallMethodVisitor'
        elif op_name == 'NegateConditionalsMutator':
            return PitMutantProcessor.negateConditionals(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'VoidMethodCallMutator': # -> THIS SHOULD BE THE TARGET AS THEN, WHENEVER A MUTANTIS REVEALED.. IT SHOULD GO TO THE MTUATEDAT ....
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
        ## above are defaults
        elif op_name == 'InlineConstantMutator':
            return PitMutantProcessor.inlineConstant(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'ConstructorCallMutator':
            return PitMutantProcessor.constructorCall(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'NonVoidMethodCallMutator': # temporary excluded
            return PitMutantProcessor.nonVoidMethodCall(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name == 'RemoveIncrementsMutator':
            return PitMutantProcessor.removeIncrements(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name in ['AOD1Mutator', 'AOD2Mutator']: 
            return PitMutantProcessor.arithmeticOpDelete(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name in ['OBBN1Mutator', 'OBBN2Mutator', 'OBBN3Mutator']:
            return PitMutantProcessor.bitwiseOperator(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name in ['CRCR1Mutator', 'CRCR2Mutator', 'CRCR3Mutator', 'CRCR4Mutator', 'CRCR5Mutator', 'CRCR6Mutator']:
            return PitMutantProcessor.constantReplace(codeTree, start_end_pos_pline, mutant, file_content, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name in ['ROR1Mutator', 'ROR2Mutator', 'ROR3Mutator', 'ROR4Mutator', 'ROR5Mutator']:
            return PitMutantProcessor.relationalOpReplace(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)
        elif op_name in ['AOR1Mutator', 'AOR2Mutator', 'AOR3Mutator', 'AOR4Mutator']:
            return PitMutantProcessor.arithmeticOpReplace(codeTree, start_end_pos_pline, mutant, groupedMutantsByLandM = groupedMutantsByLandM)

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
    #def getIndexAndBlock(mutant:ET.Element) -> Tuple[int,int]:
    def getIndex(mutant:ET.Element) -> int:
        """
        return block and index
        """
        index = int(mutant.find('index').text)
        #block = int(mutant.find('block').text)
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
        ## new
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
                    #print ('===found es', found_es)
                    #for e, _ in found_es:
                    #    print (e, e.attrib['type'], e.attrib['pos'], e.attrib['length'])
                    #print (f"Fiaield due to encountering order") ### HERE, also many
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
        ## here targeted are either ++ or --
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
            #new_op = int(matched.groups()[1]) # 1 or -1 
        should_be = "++" if org_op > 0 else "--"
        ## new
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["POSTFIX_EXPRESSION_OPERATOR", 'PREFIX_EXPRESSION_OPERATOR'], 
            label = should_be
        )
        if found_es is None: return None
        #_found_es = []
        #for e, e_type in found_es:
        #    _op = e.attrib['label']
        #    if _op == should_be: 
        #        _found_es.append([e, e_type])
        #found_es = _found_es
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
        #targets = []
        #for e, e_type in found_es:
        #    _op = e.attrib['label']
        #    if _op == should_be: 
        #        targets.append([e, e_type])
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
        #print (found_es)
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
    ): # MATH
        #63:        "Replaced integer addition with subtraction"));
        #65:        "Replaced integer subtraction with addition"));
        #67:        "Replaced integer multiplication with division"));
        #69:        "Replaced integer division with multiplication"));
        #71:        "Replaced bitwise OR with AND"));
        #73:        "Replaced bitwise AND with OR"));
        #75:        "Replaced integer modulus with multiplication"));
        #77:        "Replaced XOR with AND"));
        #79:        "Replaced Shift Left with Shift Right"));
        #81:        "Replaced Shift Right with Shift Left"));
        #83:        "Replaced Unsigned Shift Right with Shift Left"));
        #88:        "Replaced long addition with subtraction"));
        #90:        "Replaced long subtraction with addition"));
        #92:        "Replaced long multiplication with division"));
        #94:        "Replaced long division with multiplication"));
        #96:        "Replaced bitwise OR with AND"));
        #98:        "Replaced bitwise AND with OR"));
        #100:        "Replaced long modulus with multiplication"));
        #102:        "Replaced XOR with AND"));
        #104:        "Replaced Shift Left with Shift Right"));
        #106:        "Replaced Shift Right with Shift Left"));
        #108:        "Replaced Unsigned Shift Right with Shift Left"));
        #112:        "Replaced float addition with subtraction"));
        #114:        "Replaced float subtraction with addition"));
        #116:        "Replaced float multiplication with division"));
        #118:        "Replaced float division with multiplication"));
        #120:        "Replaced float modulus with multiplication"));
        #124:        "Replaced double addition with subtraction"));
        #126:        "Replaced double subtraction with addition"));
        #128:        "Replaced double multiplication with division"));
        #130:        "Replaced double division with multiplication"));
        #132:        "Replaced double modulus with multiplication"));

        ##
        # mutNo, 
        # left, 
        # right,
        # pos 
        # mutOp 
        # targeted  
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        #print (description)
        #print (lineNumber)
        # e.g., "Replaced double division with multiplication"
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
        # if org_op in ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>', '>>>']:
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
            # POSTFIX_EXPRESSION_OPERATOR -> a tricky one ... this.xx ++/--; -> this.xx = this.xx -1/+1; 
        )
        if found_es is None: return None
        targets, org_op_new_op_pairs = checkAndGetOpsAndNewOps(found_es, org_op, new_op)
        #if len(found_es) == 0:
        if len(targets) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
                # POSTFIX_EXPRESSION_OPERATOR -> a tricky one ... this.xx ++/--; -> this.xx = this.xx -1/+1; 
            )
            if found_es is None: return None
            targets, org_op_new_op_pairs = checkAndGetOpsAndNewOps(found_es, org_op, new_op)
        #targets = [] # the same line, the same operator
        #org_op_new_op_pairs = []
        #for found_e, e_type in found_es:
            #_op_str = found_e.attrib['label']
            #if e_type == "ASSIGNMENT_OPERATOR": # e.g., += 
                #if org_op + "=" == _op_str:
                    #targets.append(found_e)
                    #org_op_new_op_pairs.append([org_op + "=", new_op + "="])
            #elif e_type == 'INFIX_EXPRESSION_OPERATOR':
                #if _op_str == org_op:
                    #targets.append(found_e)
                    #org_op_new_op_pairs.append([org_op, new_op])
            #else: # POSTFIX_EXPRESSION_OPERATOR -> either ++ or --
                #if _op_str[0] == org_op:
                    #targets.append(found_e)
                    #org_op_new_op_pairs.append([org_op + org_op, new_op + new_op])
        #print ("among these", targets)
        if len(targets) > 1: # then, we cannot process this 
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                #print ("In!")
                target, idx_to_target = getElementByEncounterOrder(
                    targets,
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    #print ('herer ....', targets)
                    print ("Mismatch after checking ordering: MATH")
                    return None
                org_op_new_op_pair = org_op_new_op_pairs[idx_to_target]
        elif len(targets) == 1:
            target = targets[0]
            org_op_new_op_pair = org_op_new_op_pairs[0]
        else: # len(targets) == 0
            print (f"failed due to no-match...(MATH)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        #print ("Target", target)
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op_new_op_pair[0]
        formatted['right'] = org_op_new_op_pair[1] 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('MATH', description)
        #print ("Normal")
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

        -> currently "?" and the case where the entire condition (e.g., if (xxx)) are specified as targets, -> cannot handler 
        """
        # NEGATE_CONDITIONALS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        #targeted_ops = set(['==', '!=', '<=', '>=', '<', ">"])
        targeted_ops = set(['==', '!=', '<=', '>=', '<', ">"])#, "&&", "||"]) # yes, this will further drop mutants, but will be less or no False Positivies for the results
        #print (description)
        #print (lineNumber)
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
                #print ('hereree')
                #for e, e_type in targets:
                #    print (e, e_type, e.attrib['label'])
                ###
                target, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    #print ('here') ### !!!! => many ...!!!!
                    print ("Mismatch after checking ordering: NEGATE_CONDITIONALS")
                    print (f"LineNumer: {lineNumber}")
                    print (f"Descrip: {description}")
                    #print (targets)
                    #print (target)
                    return None
        elif len(targets) == 1:
            target, _ = targets[0]
        else:
            print (f"...failed due to no-match...(NEGATE_CONDITIONALS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            print (f"LineNumer: {lineNumber}")
            print (f"Descrip: {description}")
            #sys.exit()
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
        #elif org_op == "&&":
        #    new_op = "||"
        #elif org_op == "||":
        #    new_op = "&&"
        else:
            print (f"Something is wrong, not sure: {org_op}: {lineNumber}", target)
            return None
            
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('NEGATE_CONDITIONALS', description)
        #print ("Normal")
        return formatted
    
    @staticmethod
    def voidMethodCalls(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str, 
        groupedMutantsByLandM:Dict = None 
    ):
        """
        """
        # VOID_METHOD_CALLS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        # e.g., 
        # removed call to org/apache/commons/math/ode/nonstiff/EmbeddedRungeKuttaIntegrator::sanityChecks
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
        ###
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
        ## additional check
        #lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #cands, _ = findSpecificTypeElements_recur(mthInvoc_e, ['SimpleName'], lno_start_pos, lno_end_pos, [], False)
        #method_name_e = None
        #for cand,_ in cands:
        #    _method_name = cand.attrib['label']
        #    if method_name == _method_name:
        #        method_name_e = cand 
        #        break 
        #if method_name_e is None: 
        #    return None         
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(mthInvoc_e)
        method_call_str = file_content[start_chars_cnt - 1:end_chars_cnt] # will not include ';'
        #if file_content[end_chars_cnt] == ";":
        #    method_call_str += ";"
        #    end_chars_cnt += 1 # to include ";"
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
        # e.g., 
        # "replaced double return with 0.0d for org/apache/commons/math/ode/nonstiff/EmbeddedRungeKuttaIntegrator::integrate"
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
        
        #if len(retStmt_e) == 1: # the number of child nodes under retStmt_e
        target = retStmt_e[0] # the returned one
        #else:
        #    #lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #    #cands = findSpecificTypeElements(retStmt_e, ['NullLiteral'], lno_start_pos, lno_end_pos
        #    #)
        #    #if len(cands) != 1:  return None 
        #    #target = cands[0][0]
        #    print ("Should be w")
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
        #if len(retStmt_e) == 1: 
        target = retStmt_e[0]
        #else:
        #    lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #    cands = findSpecificTypeElements(retStmt_e, ['BooleanLiteral'], lno_start_pos, lno_end_pos)
        #    if len(cands) != 1: return None 
        #    target = cands[0][0]
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
        #if len(retStmt_e) == 1: 
        target = retStmt_e[0]
        #else:
        #    lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #    cands = findSpecificTypeElements(retStmt_e, ['BooleanLiteral'], lno_start_pos, lno_end_pos)
        #    if len(cands) != 1: return None 
        #    target = cands[0][0]
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
        # new
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
        #if len(retStmt_e) == 1: 
        target = retStmt_e[0]
        #else:
        #    return None 
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
        # e.g., 
        # "replaced double return with 0.0d for org/apache/commons/math/ode/nonstiff/EmbeddedRungeKuttaIntegrator::integrate"
        #pat = "replaced.*with (.+) for (.+)::(.*)"
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
        #lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #cands = findSpecificTypeElements(retStmt_e, ['SimpleName', 'NumberLiteral'], lno_start_pos, lno_end_pos)
        #if len(cands) != 1:
        #    return None 
        #else:
        #target = cands[0][0]
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

    ## from the below is no longer by default
    #@staticmethod
    #def removeConditional(
    # codeTree:ET.ElementTree, 
    # start_end_pos_pline:Dict[int, Tuple[int,int]], 
    # mutant:ET.Element
    #): 
    # # stronger group & not suited for our case, as its purpose is mostly about 
    # the evaluation of test suite rather than developers making faults
    #    pass
    #@staticmethod
    #def experimentalSwitch(
        #codeTree:ET.ElementTree, 
        #start_end_pos_pline:Dict[int, Tuple[int,int]], 
        #mutant:ET.Element, 
        #groupedMutantsByLandM:Dict = None 
    #):
        ## VOID_METHOD_CALLS
        #formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        #description = PitMutantProcessor.getDescription(mutant)
        #lineNumber = PitMutantProcessor.getLineNumber(mutant)
        #pass 

    @staticmethod 
    def inlineConstant(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        """
        For this, the target is literal
        inline constant: a literal valueassigned to a non-final variable

        for booleean case, will cover only the changes betwen 0 or 1 (for the others, will skip)
        """
        # INLINE_CONSTS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        pat = "Substituted ([0-9\.-a-zA-Z]+) with ([0-9\.-a-zA-Z]+)\s*$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: INLINE_CONSTS: {description}")
            return None 
        else:
            org_constant = matched.groups()[0]
            new_constant = matched.groups()[1]       
        #if org_constant in set(['true', 'false']): # -> this won't happend b/c automated change to 1 and 0
            #targeTypes = ['BooleanLiteral']
            #neg_org_constant = False
            #isPreDefConstant = False
        #else:
        try:
            neg_org_constant = eval(org_constant) < 0
            if neg_org_constant: 
                targeTypes = ['NumberLiteral']
            else:
                if eval(org_constant) > 1:
                    targeTypes = ['NumberLiteral']
                else: # 0 and 1 (can be false and true)
                    if (org_constant in set(['0', '1'])) and (new_constant in set(['0', '1'])):
                        targeTypes = ['NumberLiteral', 'BooleanLiteral']
                    else:
                        targeTypes = ['NumberLiteral']
            isPreDefConstant = False
        except Exception as e: # e.g., NaN
            targeTypes = ['QualifiedName']
            neg_org_constant = False 
            isPreDefConstant = True

        if neg_org_constant: # if orginal is negative value
            org_constant = org_constant[1:] 
            assert eval(org_constant) >= 0
        can_label_set = len(targeTypes) == 1 

        #print ("Params", can_label_set, neg_org_constant, isPreDefConstant, org_constant)
        ### CURRENTLY THE ISSUES IS -1 -> THIS IS represented as - as prefix and 1 as value .. 
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            targeTypes, 
            label = org_constant if can_label_set else None # due to true and false referred as 1 and 0
        )
        if found_es is None: return None
        #print ("found", found_es)
        # new
        #_found_es = []
        # original code
        #for e, e_type in found_es:
            #if not isPreDefConstant and e_type != 'BooleanLiteral':
                #is_neg_constant = handleNegativeConstant(codeTree, e)
                #if neg_org_constant == is_neg_constant: # either both false or both true
                    #_found_es.append([e, e_type])
            #else:
                #_found_es.append([e, e_type])
        # original end
        _found_es = []
        for e, e_type in found_es:
            if can_label_set: # true, meaning no possibitliy for boolean one. All the qualified name case will be here and the labels have been checked
                if not isPreDefConstant: # NumberLiteral
                    is_neg_constant = handleNegativeConstant(codeTree, e)
                    if neg_org_constant == is_neg_constant: # either both false or both true
                        _found_es.append([e, e_type])
                else: # QualifiedName
                    _found_es.append([e, e_type])
            else: # only the case of ['NumberLiteral', 'BooleanLiteral']
                assert not isPreDefConstant
                e_label = e.attrib['label'] 
                e_type = e.attrib['type']
                if e_type == 'BooleanLiteral': # for this case 
                    e_label = "1" if e_label == 'true' else "0"
                # compare the label value
                if e_label == org_constant: # can be numerical comparision or boolean one 
                    _found_es.append([e, e_type])
        found_es = _found_es
        #print ("found2", found_es)
        ## check 
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                targeTypes, 
                label = org_constant if can_label_set else None # due to true and false referred as 1 and 0
            )
            if found_es is None: return None
            #if neg_org_constant: 
            #_found_es = []
            #for e, e_type in found_es:
                #if not isPreDefConstant and e_type != 'BooleanLiteral':
                    #is_neg_constant = handleNegativeConstant(codeTree, e)
                    ##if is_neg_constant:
                    #if neg_org_constant == is_neg_constant:
                        #_found_es.append([e, e_type])
                #else:
                    #_found_es.append([e, e_type])
            _found_es = []
            for e, e_type in found_es:
                if can_label_set: # true, meaning no possibitliy for boolean one. All the qualified name case will be here and the labels have been checked
                    if not isPreDefConstant: # NumberLiteral
                        is_neg_constant = handleNegativeConstant(codeTree, e)
                        if neg_org_constant == is_neg_constant: # either both false or both true
                            _found_es.append([e, e_type])
                    else: # QualifiedName
                        _found_es.append([e, e_type])
                else: # only the case of ['NumberLiteral', 'BooleanLiteral']
                    assert not isPreDefConstant
                    e_label = e.attrib['label'] 
                    e_type = e.attrib['type']
                    if e_type == 'BooleanLiteral': # for this case 
                        e_label = "1" if e_label == 'true' else "0"
                    # compare the label value
                    if e_label == org_constant: # can be numerical comparision or boolean one 
                        _found_es.append([e, e_type])
            found_es = _found_es
        #
        #print ("found3", found_es)
        ## now the final ##
        if len(found_es) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                #target, _, (n_same_key, n_targets)= getElementByEncounterOrder(
                target, _ = getElementByEncounterOrder(
                    [found_e[0] for found_e in found_es],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    #if n_same_key > n_targets:
                    print ("Mismatch after checking ordering: INLINE CONSTS")
                    return None
        elif len(found_es) == 1:
            target, _ = found_es[0]
        else:
            print (f"failed due to no-match...(INLINE CONSTS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        ## convert 
        if target.attrib['type'] == 'BooleanLiteral':
            new_constant = 'true' if new_constant == '1' else 'false'
            
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = file_content[start_chars_cnt - 1:end_chars_cnt] #org_constant  # since QualifiedName name is added, 
        formatted['right'] = new_constant 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('INLINE_CONSTS', description)
        formatted['isNegConstant'] = neg_org_constant
        #print (formatted)
        return formatted 


    @staticmethod
    def constructorCall(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str, 
        groupedMutantsByLandM:Dict = None 
    ):
        # CONSTRUCTOR_CALLS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        #mutatedClass = PitMutantProcessor.getMutatedClass(mutant)
        pat = "removed call to ([a-zA-Z0-9\/]+)::(.*)$" # originally from MethodCallVisitor 
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: CONSTRUCTOR_CALLS: {description}")
            return None 
        else:
            class_name = matched.groups()[0]
            class_name = class_name.replace("/", ".")
            core_class_name = class_name.split(".")[-1]

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["ClassInstanceCreation"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["ClassInstanceCreation"]
            )
            if found_es is None: return None
        targets = []
        constructor_call_strs = []
        for e, e_type in found_es:
            start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(e)
            constructor_call_str = file_content[start_chars_cnt - 1:end_chars_cnt]
            #if constructor_call_str.startswith('this'):
            #    constructor_call_str = mutatedClass.split(".")
            if constructor_call_str == core_class_name:
                targets.append([e, e_type])
                constructor_call_strs.append(constructor_call_str)
        
        target_constructor_call_str = None
        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                constructor_call_e, idx = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if constructor_call_e is None:
                    print ("Mismatch after checking ordering: CONSTRUCTOR_CALLS")
                    return None 
                target_constructor_call_str = constructor_call_strs[idx]
        elif len(targets) == 1:
            constructor_call_e, _ = targets[0]
            target_constructor_call_str = constructor_call_strs[0]
        else:
            print (f"failed due to no-match...(CONSTRUCTOR_CALLS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None
    
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = target_constructor_call_str
        formatted['right'] = "null" # nothing as it is a remove 
        formatted['targeted'] = constructor_call_e 
        formatted['mutOp'] = ('CONSTRUCTOR_CALLS', description)
        return formatted

    @staticmethod
    def nonVoidMethodCall(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str, 
        groupedMutantsByLandM:Dict = None 
    ):
        # NON_VOID_METHOD_CALLS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        # e.g., 
        # removed call to org/apache/commons/math/ode/nonstiff/EmbeddedRungeKuttaIntegrator::sanityChecks
        pat = "removed call to ([a-zA-Z0-9\/]+)::(.*)$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: NON_VOID_METHOD_CALLS: {description}")
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
        ##
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
                    print ("Mismatch after checking ordering: NON_VOID_METHOD_CALLS")
                    return None
        elif len(targets) == 1:
            mthInvoc_e, _ = targets[0]
        else:
            print (f"failed due to no-match...(NON_VOID_METHOD_CALLS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        #lno_start_pos, lno_end_pos = start_end_pos_pline[lineNumber]
        #cands, _ = findSpecificTypeElements_recur(mthInvoc_e, ['SimpleName'], lno_start_pos, lno_end_pos, [], False)
        #method_name_e = None
        #for cand,_ in cands:
        #    _method_name = cand.attrib['label']
        #    if method_name == _method_name:
        #        method_name_e = cand 
        #        break 
        #if method_name_e is None: 
        #    return None             
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(mthInvoc_e)
        method_call_str = file_content[start_chars_cnt - 1:end_chars_cnt]
        #if file_content[end_chars_cnt] == ";":
        #    method_call_str += ";"
        #    end_chars_cnt += 1 # to include ";"
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = method_call_str
        formatted['right'] = "" # nothing as it is a remove
        formatted['targeted'] = mthInvoc_e 
        formatted['mutOp'] = ('NON_VOID_METHOD_CALLS', description)
        return formatted 

    ## currently skipped
    @staticmethod
    def negation(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        # ABS
        ## This mutator replace any use of a numeric variable (local variable, field, array cell) with its negation
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["PREFIX_EXPRESSION_OPERATOR", "SimpleName"] # either remove negation or add negation to variable
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["PREFIX_EXPRESSION_OPERATOR", "SimpleName"] # either remove negation or add negation to variable
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
                    print ("Mismatch after checking ordering: ABS (Negation)")
                    return None
        elif len(found_es) == 1:
            target, _ = found_es[0]
        else:
            print (f"failed due to no-match...(ABS - Negation)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None

        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = ""
        formatted['right'] = ""
        formatted['targeted'] = target 
        formatted['mutOp'] = ('ABS', description)
        return formatted
    
    @staticmethod 
    def removeIncrements(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # REMOVE_INCREMENTS
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)       
        # e.g., Removed increment 1, Removed increment -1, 
        pat = "Removed increment\s+([-0-9]+)$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: REMOVE_INCREMENTS: {description}")
            return None 
        else:
            org_increment = int(matched.groups()[0]) # -1 or 1
        should_be = "++" if org_increment > 0 else "--"
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["POSTFIX_EXPRESSION_OPERATOR", 'PREFIX_EXPRESSION_OPERATOR'], 
            label = should_be
        )
        if found_es is None: return None
        #
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["POSTFIX_EXPRESSION_OPERATOR", 'PREFIX_EXPRESSION_OPERATOR'], 
                label = should_be
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
                    print ("Mismatch after checking ordering: REMOVE_INCREMENTS")
                    return None 
        elif len(found_es) == 1:
            target, _ = found_es[0]
        else:
            print (f"failed due to no-match...(REMOVE_INCREMENTS)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target) # increment or decrement operator
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt) 
        formatted['left'] = should_be #file_content[start_chars_cnt - 1:end_chars_cnt]
        formatted['right'] = "" # nothing: e.g., i++ -> i, i-- -> i 
        formatted['targeted'] = target 
        formatted['mutOp'] = ('REMOVE_INCREMENTS', description)
        return formatted


    @staticmethod
    def arithmeticOpDelete(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # AOD: AOD_1, AOD_2
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        mutator = PitMutantProcessor.getMutator(mutant).split(".")[-1]
        #print (lineNumber, description)
        #pat = "Replaced ([a-zA-Z0-9\s]+) with operation with first member"
        #matched = re.match(pat, description)
        ops = set(['+', '-', '*', '/', '%'])
        def checkAndGetOps(found_es, ops):
            targets = []
            for found_e, e_type in found_es:
                _op = found_e.attrib['label']
                if e_type == "ASSIGNMENT_OPERATOR":
                    #_op = _op[0] # e.g., += -> +
                    #if _op in ops:
                    #    targets.append([found_e, e_type])
                    _op = _op.strip()
                    if _op in [f"{op}=" for op in ops]:
                        targets.append([found_e, e_type])
                elif e_type == 'INFIX_EXPRESSION_OPERATOR':
                    if _op in ops:
                        targets.append([found_e, e_type])
                else: # POSTFIX_EXPRESSION_OPERATOR and PREFIX_EXPRESSION_OPERATOR 
                    #_op = _op[0] # e.g., ++ -> +
                    #if _op in ops:
                    #    targets.append([found_e, e_type]) 
                    _op = _op.strip()
                    if _op in ["++", "--"]:
                        targets.append([found_e, e_type])
            return targets 
        #all_es = list(codeTree.iter())
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", 
             "POSTFIX_EXPRESSION_OPERATOR", "PREFIX_EXPRESSION_OPERATOR"]
        )
        if found_es is None: return None
        found_es = checkAndGetOps(found_es, ops)
        #if lineNumber == 519:
            #print("found", found_es)
            #for e,_ in found_es:
                #print (e.attrib)
            #sys.exit()
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", 
                "POSTFIX_EXPRESSION_OPERATOR", "PREFIX_EXPRESSION_OPERATOR"]
            )
            if found_es is None: return None
            found_es = checkAndGetOps(found_es, ops)
        targets = found_es
        #targets = []
        #for found_e, e_type in found_es:
            #_op = found_e.attrib['label']
            #if e_type == "ASSIGNMENT_OPERATOR":
                #_op = _op[0] # e.g., += -> +
                #if _op in ops:
                    #targets.append([found_e, e_type])
            #elif e_type == 'INFIX_EXPRESSION_OPERATOR':
                #if _op in ops:
                    #targets.append([found_e, e_type])
            #else: # POSTFIX_EXPRESSION_OPERATOR and PREFIX_EXPRESSION_OPERATOR 
                #_op = _op[0] # e.g., ++ -> +
                #if _op in ops:
                    #targets.append([found_e, e_type])         
        if len(targets) > 1:
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                #for e, e_type in targets:
                #    print (e, e_type, e.attrib['pos'], e.attrib['length'], e.attrib['label'])
                target, _ = getElementByEncounterOrder(
                    [target[0] for target in targets],
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: AOD")
                    return None
        elif len(targets) == 1:
            target,_ = targets[0]
        else:
            print (f"failed due to no-match...(AOD)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None  

        # get index of the found target 
        ## target -> currently, an operator 
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        pos = start_chars_cnt - 1
        length = end_chars_cnt - pos
        parent_e, target_e, left_e, right_e = PitMutantProcessor.getParentAndLeftAndRight(
            codeTree, pos, length, 
            None, 0, False) 
        assert target_e == target, f"{target_e} vs {target}"
        
        parent_start_chars_cnt, parent_end_chars_cnt = getStartAndEndCharCnt(parent_e)
        targeted_start_chars_cnt = parent_start_chars_cnt
        targeted_end_chars_cnt = parent_end_chars_cnt
        target_type = target.attrib['type']
        org_op = file_content[targeted_start_chars_cnt - 1:targeted_end_chars_cnt]
        new_op = None
        if target_type in 'INFIX_EXPRESSION_OPERATOR':
            ## here should be +, -, /, *, %
            assert left_e is not None and right_e is not None, f"{left_e}, {right_e}"
            left_start_chars_cnt, left_end_chars_cnt = getStartAndEndCharCnt(left_e)
            right_start_chars_cnt, right_end_chars_cnt = getStartAndEndCharCnt(right_e)
            #org_op = file_content[left_start_chars_cnt - 1:right_end_chars_cnt]
            if mutator == 'AOD1Mutator': 
                new_op = file_content[left_start_chars_cnt - 1:left_end_chars_cnt]
            else:
                new_op = file_content[right_start_chars_cnt - 1:right_end_chars_cnt]
        elif target_type == 'ASSIGNMENT_OPERATOR':
            ## here should be +=, -=, /=, *=, %=
            assert left_e is not None and right_e is not None, f"{left_e}, {right_e}"
            left_start_chars_cnt, left_end_chars_cnt = getStartAndEndCharCnt(left_e)
            right_start_chars_cnt, right_end_chars_cnt = getStartAndEndCharCnt(right_e)
            #org_op = file_content[left_start_chars_cnt - 1:right_end_chars_cnt]
            if mutator == 'AOD1Mutator': 
                new_op = file_content[left_start_chars_cnt - 1:left_end_chars_cnt]
                new_op = f"{new_op} = {new_op}" # a += b -> a = a
            else:
                left_op = file_content[left_start_chars_cnt - 1:left_end_chars_cnt]
                new_op = file_content[right_start_chars_cnt - 1:right_end_chars_cnt]
                new_op = f"{left_op} = {new_op}" # a += b -> a = b
        elif target_type == 'POSTFIX_EXPRESSION_OPERATOR': # a++ -> a = a + 1 -> a = a
            # here op should be -- or ++ 
            assert left_e is not None
            left_start_chars_cnt, left_end_chars_cnt = getStartAndEndCharCnt(left_e)
            left_op = file_content[left_start_chars_cnt - 1:left_end_chars_cnt]
            if mutator == 'AOD1Mutator': 
                new_op = f"{left_op} = {left_op}"
            else:
                new_op = f"{left_op} = 1"
        else: # PREFIX_EXPRESSION_OPERATOR # ++a (if prefix -> op should be ++ or --)
            assert right_e is not None
            right_start_chars_cnt, right_end_chars_cnt = getStartAndEndCharCnt(right_e)
            right_op = file_content[right_start_chars_cnt - 1:right_end_chars_cnt]
            if mutator == 'AOD1Mutator': 
                new_op = f"{right_op} = {right_op}"  # ++ a -> 
            else:
                new_op = f"{right_op} = 1"
    
        formatted['pos'] = (lineNumber, targeted_start_chars_cnt, targeted_end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target # the operator  
        mutOp = 'AOD_1' if mutator == 'AOD1Mutator' else 'AOD_2'
        formatted['mutOp'] = (mutOp, description) 
        return formatted
        #idx_to_target = None
        #for i, e in enumerate(all_es):
            #try:
                #_pos = int(e.attrib['pos'])
            #except Exception:
                #continue 
            #_length = int(e.attrib['length'])
            #if (_pos == pos) and (length == _length):
                #idx_to_target = i  
                #break 
        #assert idx_to_target is not None, f"{pos}, {length}" 
        #target_type = target.attrib['type']
        #first_member = all_es[idx_to_target - 1] 
        #first_start_chars_cnt, first_end_chars_cnt = getStartAndEndCharCnt(first_member)
        #org_op = None 
        #if target_type != 'POSTFIX_EXPRESSION_OPERATOR':
            #scnd_member = all_es[idx_to_target + 1] 
            #scnd_start_chars_cnt, scnd_end_chars_cnt = getStartAndEndCharCnt(scnd_member)
            #org_op = file_content[first_start_chars_cnt - 1:scnd_end_chars_cnt]
        #else: # POSTFIX_EXPRESSION_OPERATOR
            #org_op = file_content[first_start_chars_cnt - 1:end_chars_cnt] # up to target: a++
            #scnd_member, scnd_start_chars_cnt, scnd_end_chars_cnt = None, None, None
        #mutOp = None
        #new_op = None
        #if mutator == 'AOD1Mutator': # replace with first member
            #new_op = file_content[first_start_chars_cnt - 1:first_end_chars_cnt]
            #if target_type == 'INFIX_EXPRESSION_OPERATOR': # e.g., b + c -> c
                #new_op = new_op 
            #elif target_type == 'ASSIGNMENT_OPERATOR': # e.g., a += b -> a = a
                #new_op = f"{new_op} = {new_op}"
            #else: # POSTFIX_EXPRESSION_OPERATOR e.g., a++ ->  a = a + 1 -> a = a
                #new_op = f"{new_op} = {new_op}"
            #mutOp = "AOD_1"
        #elif mutator == 'AOD2Mutator': # replace with second memeber
            #new_op = file_content[scnd_start_chars_cnt - 1:scnd_end_chars_cnt]
            #if target_type == 'INFIX_EXPRESSION_OPERATOR': # e.g., b + c -> c
                #new_op = new_op 
            #elif target_type == 'ASSIGNMENT_OPERATOR': # e.g., a += b -> a = b
                #first_mem_str = file_content[first_start_chars_cnt - 1:first_end_chars_cnt]
                #new_op = f"{first_mem_str} = {new_op}"
            #else: # POSTFIX_EXPRESSION_OPERATOR e.g., a++ -> nothing as always .. -> ;1){...} 
                #first_mem_str = file_content[first_start_chars_cnt - 1:first_end_chars_cnt]
                #new_op = f"{first_mem_str} = 1" # e.g., a++ -> a = a + 1 -> a = 1
            #mutOp = "AOD_2"
        #else:
            #print (f"{mutator}: something wrong")
            #assert False
        ##formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        ## pos -> should cover the entire org_op to be replaced 
        #formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        #formatted['left'] = org_op
        #formatted['right'] = new_op
        #formatted['targeted'] = target 
        #formatted['mutOp'] = (mutOp, description) 
        #return formatted

    @staticmethod
    def bitwiseOperator(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        # OBBN
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        mutator = PitMutantProcessor.getMutator(mutant).split(".")[-1]
        ops = set(['|', '&'])

        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR"]
            )
            if found_es is None: return None
        targets = []
        for found_e, e_type in found_es:
            _op = found_e.attrib['label']
            if e_type == "ASSIGNMENT_OPERATOR":
                _op = _op[0] # e.g., += -> +
                if _op in ops:
                    targets.append([found_e, e_type])
            else: 
                if _op in ops:
                    targets.append([found_e, e_type])

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
                    print ("Mismatch after checking ordering: OBBN")
                    return None
        elif len(targets) == 1:
            target,_ = targets[0]
        else:
            print (f"failed due to no-match...(OBBN)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None  

        # get index of the found target 
        ## target -> currently, an operator 
        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        pos = start_chars_cnt - 1
        length = end_chars_cnt - pos
        parent_e, target_e, left_e, right_e = PitMutantProcessor.getParentAndLeftAndRight(
            codeTree, pos, length, 
            None, 0, False) 
        assert target_e == target, f"{target_e} vs {target}"
        assert left_e is not None and right_e is not None, f"{left_e}, {right_e}"

        parent_start_chars_cnt, parent_end_chars_cnt = getStartAndEndCharCnt(parent_e)
        targeted_start_chars_cnt = parent_start_chars_cnt
        targeted_end_chars_cnt = parent_end_chars_cnt
        target_type = target.attrib['type']
        target_op = target.attrib['label']
        if target_type == 'ASSIGNMENT_OPERATOR': target_op = target_op[0]
        
        org_op = file_content[targeted_start_chars_cnt - 1:targeted_end_chars_cnt]
        left_start_chars_cnt, left_end_chars_cnt = getStartAndEndCharCnt(left_e)
        left_op = file_content[left_start_chars_cnt - 1:left_end_chars_cnt]
        right_start_chars_cnt, right_end_chars_cnt = getStartAndEndCharCnt(right_e)
        right_op = file_content[right_start_chars_cnt - 1:right_end_chars_cnt]
        mutOp, new_op = None, None
        if target_type in 'INFIX_EXPRESSION_OPERATOR':
            if mutator == 'OBBN1Mutator': 
                new_op = "&" if target_op == "|" else "|"
                new_op = f"{left_op} {new_op} {right_op}"
                mutOp = 'OBBN1'
            elif mutator == 'OBBN2Mutator': # left 
                new_op = left_op
                mutOp = 'OBBN2'
            else: # mutator == 'OBBN3Mutator' # right
                new_op = right_op  
                mutOp = 'OBBN3'
        else: # ASSIGNMENT_OPERATOR
            if mutator == 'OBBN1Mutator': 
                new_op = "&" if target_op == "|" else "|"
                new_op = f"{left_op} {new_op}= {right_op}"
                mutOp = 'OBBN1'
            elif mutator == 'OBBN2Mutator': # left 
                new_op = f"{left_op} = {left_op}"
                mutOp = 'OBBN2'
            else: # mutator == 'OBBN3Mutator' # right
                new_op = f"{left_op} = {right_op}"  
                mutOp = 'OBBN3'

        formatted['pos'] = (lineNumber, targeted_start_chars_cnt, targeted_end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op
        formatted['targeted'] = target # the operator  
        formatted['mutOp'] = (mutOp, description) 
        return formatted

# not among the available operators
    ### cannot find in the github repo 
    #@staticmethod
    #def arithmeticOpReplace(
    # codeTree:ET.ElementTree, 
    # start_end_pos_pline:Dict[int, Tuple[int,int]], 
    # mutant:ET.Element
    #):
        ## AOR
        #pass 
#
    @staticmethod
    def constantReplace(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        file_content:str,
        groupedMutantsByLandM:Dict = None 
    ):
        ### this also target inlince constants -> so, eventhough QualifiedName may work, leave it
        # CRCR
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        mutator = PitMutantProcessor.getMutator(mutant).split(".")[-1]
        #pat = "Substituted .* with\s*(.*)\s*$"
        #pat = "Substituted (.*) with (.*)$"
        pat = "Substituted ([0-9\.-a-zA-Z]+) with ([0-9\.-a-zA-Z]+)\s*$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: {mutator}: {description}")
            return None 
        else:
            org_constant, new_constant = matched.groups()
        ##
        try:
            neg_org_constant = eval(org_constant) < 0
            if neg_org_constant: 
                targeTypes = ['NumberLiteral']
            else:
                if eval(org_constant) > 1:
                    targeTypes = ['NumberLiteral']
                else: # 0 and 1 (can be false and true)
                    if (org_constant in set(['0', '1'])) and (new_constant in set(['0', '1'])):
                        targeTypes = ['NumberLiteral', 'BooleanLiteral']
                    else:
                        targeTypes = ['NumberLiteral']
                
            isPreDefConstant = False
        except Exception as e:
            targeTypes = ['QualifiedName']
            neg_org_constant = False 
            isPreDefConstant = True

        if neg_org_constant:
            org_constant = org_constant[1:] 
            assert eval(org_constant) >= 0
        can_label_set = len(targeTypes) == 1 
        ##
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            targeTypes, 
            label = org_constant if can_label_set else None # due to true and false referred as 1 and 0
        )
        if found_es is None: return None
        ## new
        #_found_es = []
        #for e, e_type in found_es:
            #if not isPreDefConstant:
                #is_neg_constant = handleNegativeConstant(codeTree, e)
                ##if is_neg_constant:
                #if neg_org_constant == is_neg_constant:
                    #_found_es.append([e, e_type])
            #else:
                #_found_es.append([e, e_type])
        _found_es = []
        for e, e_type in found_es:
            if can_label_set: # true, meaning no possibitliy for boolean one. All the qualified name case will be here and the labels have been checked
                if not isPreDefConstant: # NumberLiteral
                    is_neg_constant = handleNegativeConstant(codeTree, e)
                    if neg_org_constant == is_neg_constant: # either both false or both true
                        _found_es.append([e, e_type])
                else: # QualifiedName
                    _found_es.append([e, e_type])
            else: # only the case of ['NumberLiteral', 'BooleanLiteral']
                assert not isPreDefConstant
                e_label = e.attrib['label'] 
                e_type = e.attrib['type']
                if e_type == 'BooleanLiteral': # for this case 
                    e_label = "1" if e_label == 'true' else "0"
                # compare the label value
                if e_label == org_constant: # can be numerical comparision or boolean one 
                    _found_es.append([e, e_type])
        found_es = _found_es
        ## 
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                targeTypes, 
                label = org_constant if can_label_set else None # due to true and false referred as 1 and 0
            )
            if found_es is None: return None
            #_found_es = []
            #for e, e_type in found_es:
                #if not isPreDefConstant:
                    #is_neg_constant = handleNegativeConstant(codeTree, e)
                    #if neg_org_constant == is_neg_constant:
                        #_found_es.append([e, e_type])
                #else:
                    #_found_es.append([e, e_type])
            #found_es = _found_es
            _found_es = []
            for e, e_type in found_es:
                if can_label_set: # true, meaning no possibitliy for boolean one. All the qualified name case will be here and the labels have been checked
                    if not isPreDefConstant: # NumberLiteral
                        is_neg_constant = handleNegativeConstant(codeTree, e)
                        if neg_org_constant == is_neg_constant: # either both false or both true
                            _found_es.append([e, e_type])
                    else: # QualifiedName
                        _found_es.append([e, e_type])
                else: # only the case of ['NumberLiteral', 'BooleanLiteral']
                    assert not isPreDefConstant
                    e_label = e.attrib['label'] 
                    e_type = e.attrib['type']
                    if e_type == 'BooleanLiteral': # for this case 
                        e_label = "1" if e_label == 'true' else "0"
                    # compare the label value
                    if e_label == org_constant: # can be numerical comparision or boolean one 
                        _found_es.append([e, e_type])
            found_es = _found_es
        ##
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
                    print (f"Mismatch after checking ordering: {mutator}")
                    return None
        elif len(found_es) == 1:
            target, _ = found_es[0]
        else:
            print (f"failed due to no-match...({mutator})\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
        ## convert 
        if target.attrib['type'] == 'BooleanLiteral': # 
            new_constant = 'true' if new_constant == '1' else 'false'
        #
        # for BooleanLiteral, already filtered out by limiting to the substituion between 0 and 1
        if mutator == 'CRCR1Mutator':
            mutOp = 'CRCR1'
        elif mutator == 'CRCR2Mutator':
            mutOp = 'CRCR2'
        elif mutator == 'CRCR3Mutator':
            mutOp = 'CRCR3'
        elif mutator == 'CRCR4Mutator':
            mutOp = 'CRCR4'
        elif mutator == 'CRCR5Mutator':
            mutOp = 'CRCR5'
        elif mutator == 'CRCR6Mutator':
            mutOp = 'CRCR6'

        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = file_content[start_chars_cnt - 1:end_chars_cnt] # if it is negative, it will not contain -1 
        formatted['right'] = new_constant 
        formatted['targeted'] = target # the constant -> for
        formatted['mutOp'] = (mutOp, description)
        formatted['isNegConstant'] = neg_org_constant
        return formatted 


    @staticmethod
    def relationalOpReplace(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        """
        """
        # ROR: ROR1, ROR2, ROR3, ROR4, ROR5
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        mutator = PitMutantProcessor.getMutator(mutant).split(".")[-1]
        #targeted_ops = set(['<', '<=', '>', '>=', '==', '!='])
        #pat = "Less than to less or equal"
        pat = "([a-zA-Z\s]+) to ([a-zA-Z\s]+)$"
        matched = re.match(pat, description)
        if not bool(matched):
            print (f"No match: {mutator}: {description}")
            return None 
        else:
            org_op, new_op = matched.groups()
            org_op = org_op.strip().lower()
            new_op = new_op.strip().lower()
        
        def convert(op_str:str) -> str:
            if op_str == 'less than':
                return "<"
            elif op_str == 'less or equal':
                return "<="
            if op_str == 'greater than':
                return ">"
            elif op_str == 'greater or equal':
                return ">="
            elif op_str == 'equal':
                return "=="
            elif op_str == 'not equal':
                return "!="
            else:
                print (f"Wrong opterator: {op_str} in ROR")
                assert False 
        
        org_op = convert(org_op)
        new_op = convert(new_op)
        # find the location 
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR"], 
            label = org_op
        )
        #print (description)
        #print ('org', lineNumber, len(found_es), found_es, org_op)
        #for e, _ in found_es:
        #    print ("++", e.attrib)
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR"], 
                label = org_op
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
                    print ("Mismatch after checking ordering: ROR")
                    return None 
        elif len(targets) == 1:
            target, _ = targets[0]
        else:
            print (f"failed due to no-match...(ROR)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 
            #target = None 
        
        if mutator == 'ROR1Mutator':
            mutOp = 'ROR1'
        elif mutator == 'ROR2Mutator':
            mutOp = 'ROR2'
        elif mutator == 'ROR3Mutator':
            mutOp = 'ROR3'
        elif mutator == 'ROR4Mutator':
            mutOp = 'ROR4'
        elif mutator == 'ROR5Mutator':
            mutOp = 'ROR5'

        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target) # target => targeted
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op
        formatted['right'] = new_op 
        formatted['targeted'] = target 
        formatted['mutOp'] = (mutOp, description)
        return formatted 
    

    @staticmethod
    def arithmeticOpReplace(
        codeTree:ET.ElementTree, 
        start_end_pos_pline:Dict[int, Tuple[int,int]], 
        mutant:ET.Element, 
        groupedMutantsByLandM:Dict = None 
    ):
        # AOR : AOR1, AOR2, AOR3, AOR4
        ## AOR = +, -, *, /, % -> five operators 
        formatted = {'left':[], 'right':[], 'pos':None, 'mutOp':None, 'targeted':None}
        description = PitMutantProcessor.getDescription(mutant)
        lineNumber = PitMutantProcessor.getLineNumber(mutant)
        mutator = PitMutantProcessor.getMutator(mutant).split(".")[-1]
        #print (description)
        #print (lineNumber)
        pat = "Replaced ([a-zA-Z0-9\s]+) with ([a-zA-Z0-9\s]+)\s*$"
        matched = re.match(pat, description)
        op_str_dict = {
            'addition':"+", 
            'subtraction':"-", 
            'multiplication':"*", 
            'division':"/", 
            'modulus':"%",
        }

        def convert(op_str) -> Tuple[str,str]:
            op_ts = op_str.split(" ")
            if len(op_ts) == 2: # double xxx : old op
                datatype, _op_str = op_ts
                op = op_str_dict[_op_str]
            else: # xxx : new op
                datatype = None
                op = op_str_dict[op_str] 
            return op, datatype

        if not bool(matched):
            print (f"No match: {mutator}: {description}")
            return None 
        else:
            org = matched.groups()[0].strip().lower()
            org_op, _ = convert(org)
            new = matched.groups()[1].strip().lower()
            new_op, _ = convert(new)
        
        found_es = findElement_revised(
            codeTree, 
            start_end_pos_pline, 
            lineNumber, 
            ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
        )
        if found_es is None: return None
        if len(found_es) == 0:
            found_es = findElement(
                codeTree, 
                start_end_pos_pline, 
                lineNumber, 
                ["INFIX_EXPRESSION_OPERATOR", "ASSIGNMENT_OPERATOR", "POSTFIX_EXPRESSION_OPERATOR"]
            )
            if found_es is None: return None
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

        if len(targets) > 1: # then, we cannot process this 
            if groupedMutantsByLandM is None:
                print (f"failed ...\n\tDescription: {description}\n\tLineNumber: {lineNumber}")
                return None
            else:
                #print ("In!")
                target, idx_to_target = getElementByEncounterOrder(
                    targets,
                    mutant,
                    groupedMutantsByLandM,
                )
                if target is None:
                    print ("Mismatch after checking ordering: AOR")
                    return None
                org_op_new_op_pair = org_op_new_op_pairs[idx_to_target]
        elif len(targets) == 1:
            target = targets[0]
            org_op_new_op_pair = org_op_new_op_pairs[0]
        else: # len(targets) == 0
            print (f"failed due to no-match...(AOR)\n\tDescription: {description}\n\tLineNumber: {lineNumber}") 
            return None 

        if mutator == 'AOR1Mutator':
            mutOp = 'AOR1'
        elif mutator == 'AOR2Mutator':
            mutOp = 'AOR2'
        elif mutator == 'AOR3Mutator':
            mutOp = 'AOR3'
        elif mutator == 'AOR4Mutator':
            mutOp = 'AOR4'

        start_chars_cnt, end_chars_cnt = getStartAndEndCharCnt(target)
        formatted['pos'] = (lineNumber, start_chars_cnt, end_chars_cnt)
        formatted['left'] = org_op_new_op_pair[0]
        formatted['right'] = org_op_new_op_pair[1] 
        formatted['targeted'] = target 
        formatted['mutOp'] = (mutOp, description)
        #print ("Normal")
        return formatted 
