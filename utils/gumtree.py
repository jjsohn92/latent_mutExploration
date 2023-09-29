from typing import List, Dict, Tuple, Set, Union
import os, sys 
import xml.etree.ElementTree as ET
import utils.java_utils as java_utils 

GUMTREE_HOME = os.getenv("GUMTREE_HOME")

def isDeleted(e:ET.Element) -> bool:
    try:
        _ = e.attrib['other_pos']
        return True
    except KeyError:
        return False 

def getFrirstUndeletedPrevOrCurr(es:List[ET.Element], start_idx:int, for_prev:bool = True) -> Tuple:
    """
    If all deleted (i.e., no other pos), also without other_pos as it is what it should be (this one also deleted)
    """
    # es = list(p), p = infixExpression 
    idx_to_end = 0 if for_prev else len(es) - 1
    def reachEnd(idx_to_end, idx, for_prev):
        if for_prev:
            return idx <= idx_to_end
        else:
            return idx >= idx_to_end

    incr_or_dcr = -1 if for_prev else 1 # for accessing previous -1, after +1
    new_prev_e = None
    new_idx_to_prev = start_idx
    while True:
        _prev_e = es[new_idx_to_prev] 
        # not deleted 
        if not isDeleted(_prev_e):
            new_prev_e = _prev_e
            break 
        if reachEnd(idx_to_end, new_idx_to_prev, for_prev): 
            # meaning _prev_e is deleted (i.e., need to look further) but the current one is already at the end
            break 
        new_idx_to_prev += incr_or_dcr 
    return new_prev_e 

def getOtherPosAndLength(e:ET.Element) -> Tuple[int, int]:
    if e is None:
        return (None, None)
    try:
        other_pos = int(e.attrib['other_pos'])
    except KeyError:
        return (None, None)
    other_length = int(e.attrib['other_length'])
    return (other_pos, other_length)

#def getStartAndEndCharCnt(e:ET.Element) -> Tuple[int, int]:
    ## start from 1 
    #start = int(e.attrib['pos']) + 1 # start fr
    #end = start + int(e.attrib['length']) - 1# can be same as start
    #return start, end 
def addMissingInfixOps(
    codeTree:ET.ElementTree, 
    prev_file_content:str, 
    curr_file_conent:str, 
    ret_added:bool = False
) -> Union[ET.ElementTree, Tuple[ET.ElementTree, List[ET.Element]]]:
    ## codeTree -> gumtree diff of prev_file_content and curr_file_content

    def op_convert(org:str) -> str:
        ## >, <, >=, <=, &, &&
        # >: &gt
        # <: &lt
        # >=: &gt;=
        # <=: &lt;=
        # &&: &amp;&amp
        # &: &amp
        org_c_conv_c = {">":"&gt", "<":"&lt", "&":"&amp"}
        #org_c_conv_c = {">":"&gt", "<":"&lt", ">=":"&gt;=", "<=":"&lt;=", "&":"&amp", "&&":"&amp;&amp"}
        #try:
        #    converted = org_c_conv_c[org]
        #except KeyError:
        #    converted = org
        converted = ""
        for c in org:
            try:
                conv_c = org_c_conv_c[c]
                converted += conv_c + ";"
            except KeyError:
                converted += c
        return converted

    infixExpressionNodes = codeTree.findall(".//*[@type = 'InfixExpression']")
    added_missed_infixOps = [] if ret_added else None
    for p in infixExpressionNodes:
        es = list(p)
        idx_to_infix_op = 1
        infix_op = es[idx_to_infix_op]
        assert infix_op.attrib['type'] == 'INFIX_EXPRESSION_OPERATOR', f"{infix_op.attrib}"
        ##
        num_es = len(es) # in most cases, will be three n_es = 3
        num_following_es = num_es - idx_to_infix_op - 1
        if num_following_es > 1: # in normal case, 3 - 1 - 1 = 1 
            at = []
            for i in range(num_following_es + 1, num_es, 2):
                prev_e, curr_e = es[i-1], es[i]
                ## for prev file content -> pos & length & label
                end_of_prev = int(prev_e.attrib['pos']) + int(prev_e.attrib['length']) 
                start_of_curr = int(curr_e.attrib['pos'])
                infix_op = prev_file_content[end_of_prev:start_of_curr]
                start_of_infix_op = end_of_prev
                #for j,c in enumerate(infix_op):
                for c in infix_op:
                    if bool(c.strip()): break 
                    #prefixes.append(end_of_prev + 1) # to refer the next (+1)
                    start_of_infix_op += 1
                end_of_infix_op = start_of_curr
                #for j,c in enumerate(infix_op[::-1]):
                for c in infix_op[::-1]:
                    if bool(c.strip()): break 
                    #postfixes.append(start_of_curr - j -1)
                    end_of_infix_op -= 1
                # set 
                length_of_infix_op = end_of_infix_op - start_of_infix_op
                infix_op_txt = infix_op.strip()
                n = len(infix_op_txt)
                msg = f"{infix_op_txt}"
                msg += f" (vs {prev_file_content[start_of_infix_op:end_of_infix_op]}): {n} vs {length_of_infix_op}"
                assert n == length_of_infix_op, msg 
                infix_op_txt = op_convert(infix_op_txt)
    
                ## for current file content
                #####
                ### -> need to consider that the element, either prev_e or curr_e might be deleted 
                prev_e_at_curr = getFrirstUndeletedPrevOrCurr(es, i - 1, for_prev = True)
                curr_e_at_curr = getFrirstUndeletedPrevOrCurr(es, i, for_prev = False)
                #if prev_e_at_curr is None and curr_e_at_curr is not None:
                #    print (p.attrib)
                #    print (curr_e_at_curr.attrib)
                #    with open("a.pkl", 'rb') as f:
                #        import pickle
                #        pickle.dump(codeTree, f)
                #    sys.exit()
                other_pos_prev, other_length_prev = getOtherPosAndLength(prev_e_at_curr)
                if other_pos_prev is None:
                    oth_start_of_infix_op = None #oth_end_of_prev
                    oth_length_of_infix_op = None
                    oth_infix_op_txt = None
                else:
                    #oth_end_of_prev = int(prev_e.attrib['other_pos']) + int(prev_e.attrib['other_length']) 
                    oth_end_of_prev = other_pos_prev + other_length_prev
                    oth_start_of_curr, _ = getOtherPosAndLength(curr_e_at_curr) # can be None
                    #if oth_start_of_curr is None:
                    #    print (prev_e_at_curr, curr_e_at_curr)
                    #    print (curr_e_at_curr.attrib)
                    #    with open("a.pkl", 'wb') as f:
                    #        import pickle
                    #        pickle.dump(codeTree, f)
                    #    sys.exit()
                    #oth_start_of_curr = int(curr_e.attrib['other_pos'])
                    oth_infix_op = curr_file_conent[oth_end_of_prev:oth_start_of_curr]
                    ##
                    oth_start_of_infix_op = oth_end_of_prev
                    for c in oth_infix_op:
                        if bool(c.strip()): break 
                        oth_start_of_infix_op += 1
                    oth_end_of_infix_op = oth_start_of_curr
                    for c in oth_infix_op[::-1]:
                        if bool(c.strip()): break 
                        oth_end_of_infix_op -= 1
                    # set 
                    oth_length_of_infix_op = oth_end_of_infix_op - oth_start_of_infix_op
                    oth_infix_op_txt = oth_infix_op.strip()
                    oth_n = len(oth_infix_op_txt)
                    msg = f"{oth_infix_op_txt}" 
                    msg += f" (vs {curr_file_conent[oth_start_of_infix_op:oth_end_of_infix_op]}): {oth_n} vs {oth_length_of_infix_op}"
                    assert oth_n == oth_length_of_infix_op, msg 
                #   ## generate and append
                    #e_str = f'<tree type="INFIX_EXPRESSION_OPERATOR" label="{infix_op_txt}" pos="{start_of_infix_op}" length="{length_of_infix_op}"></tree>'
                    #infix_op_txt = op_convert(infix_op_txt)
                    oth_infix_op_txt = op_convert(oth_infix_op_txt)

                # set string 
                e_str = f'<tree type="INFIX_EXPRESSION_OPERATOR" label="{infix_op_txt}"' 
                e_str += f' pos="{start_of_infix_op}" length="{length_of_infix_op}"'
                if oth_infix_op_txt is not None:
                    e_str += f' other_label="{oth_infix_op_txt}"'
                    e_str += f' other_pos="{oth_start_of_infix_op}" other_length="{oth_length_of_infix_op}"></tree>'
                else:
                    e_str += "></tree>"
                try: 
                    infix_op_e = ET.fromstring(e_str)
                except Exception as e:
                    print (e)
                    print (e_str)
                    sys.exit()
                at.append([i, infix_op_e]) # 
                if added_missed_infixOps is not None:
                    added_missed_infixOps.append(infix_op_e)
            #print ("At")
            #print ([(idx, a.attrib) for idx,a in at])
            #if len(at) > 0:
            #    sys.exit()
            # insert 
            for cnt, (idx, new_infix_op) in enumerate(at):
                p.insert(idx + cnt, new_infix_op) # + cnt because of the previous insert 
            #sys.exit()
    if ret_added:
        return codeTree, added_missed_infixOps
    else:
        return codeTree



# -> should be called once per file for efficiency
def processFile_gumtree_by_file(work_dir:str, fileA:str, fileB:str) -> ET.Element: #ET.ElementTree:
    import subprocess 
    from subprocess import CalledProcessError
    gumtree_bin = os.path.join(GUMTREE_HOME, "gumtree")
    cmd = f"{gumtree_bin} axmldiff {fileA} {fileB}" 
    java_home = os.getenv('JAVA_HOME')
    java_utils.changeJavaVer(17, use_sdk=False)
    print(cmd)
    try:
        output = subprocess.check_output(
            cmd, cwd = work_dir, shell = True).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e: # ... this actually will also catch a simple test failure
        print (cmd)
        print (e)
        assert False #
        #return None
    os.environ['JAVA_HOME'] = java_home
    tree = ET.fromstring(output) # if we want to add missing Infix ops -> here is the place
    return tree

# -> should be called once per file for efficiency
def processFile_gumtree(fileContent:str, use_sdk:bool = False, k:str = None) -> ET.ElementTree:
    import subprocess 
    from subprocess import CalledProcessError
    print ("Diff Gumtree Called")
    ## not sure why this happen 
    ##
    temp_file = "temp.java" if k is None else f"temp{k}.java"
    with open(temp_file, 'w') as f:
        f.write(fileContent)
    outputfile = "temp.xml" if k is None else f"temp{k}.xml"
    gumtree_bin = os.path.join(GUMTREE_HOME, "gumtree")
    cmd = f"{gumtree_bin} axmldiff {temp_file} {temp_file} > " + outputfile
    java_utils.changeJavaVer(17, use_sdk=use_sdk)
    try:
        _ = subprocess.run(cmd, shell = True)
    except CalledProcessError as e: # ... this actually will also catch a simple test failure
        print (cmd)
        print (e)
        assert False #
        #return None
    java_utils.changeJavaVer(8, use_sdk=use_sdk)
    tree = ET.parse(outputfile) # if we want to add missing Infix ops -> here is the place
    ###
    tree = addMissingInfixOps(tree, fileContent, fileContent)
    ###
    os.remove(temp_file)
    os.remove(outputfile)
    return tree


# -> should be called once per file for efficiency
# usually first arg (a_fileContent) is from prev commit
def processABFile_gumtree(a_fileContent:str, b_fileContent, ret_added:bool = False) -> ET.ElementTree:
    import subprocess 
    from subprocess import CalledProcessError
    import time 
    ts = time.time()
    ## not sure why this happen 
    ##
    with open(f"temp_a_{ts}.java", 'w') as f:
        f.write(a_fileContent)
    with open(f"temp_b_{ts}.java", 'w') as f:
        f.write(b_fileContent)
    #outputfile = f"temp_a_b_{ts}.xml"
    gumtree_bin = os.path.join(GUMTREE_HOME, "gumtree")
    #cmd = f"{gumtree_bin} axmldiff temp_a_{ts}.java temp_b_{ts}.java > " + outputfile
    cmd = f"{gumtree_bin} axmldiff temp_a_{ts}.java temp_b_{ts}.java"
    java_utils.changeJavaVer(17, use_sdk=False)
    try:
        #_ = subprocess.run(cmd, shell = True)
        output = subprocess.check_output(
            cmd, shell = True, cwd = "."
        ).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e: # ... this actually will also catch a simple test failure
        print (cmd)
        print (e)
        assert False #
        #return None
    java_utils.changeJavaVer(8, use_sdk=False)
    #tree = ET.parse(outputfile) # if we want to add missing Infix ops -> here is the place
    tree = ET.fromstring(output)
    ###
    tree = addMissingInfixOps(tree, a_fileContent, b_fileContent, ret_added = ret_added) # ... acutally for this, we need to think more about the pos and other pos here ... and
    ###
    # cleaning
    #os.remove(outputfile)
    os.remove(f"temp_a_{ts}.java")
    os.remove(f"temp_b_{ts}.java")
    if not ret_added: # only tree
        return tree
    else: # both tree and list of added infixOps
        tree, added_missed_infixOps = tree
        return tree, added_missed_infixOps
    
def mapPositions(
    tree:ET.ElementTree, wo_comment:bool = True
) -> Dict[Tuple[int,int],Tuple[int,int]]:
    """
    exclude
    """
    if not wo_comment:
        exclude = set([])
    else:
        exclude = ['Javadoc', 'TextElement', 'TagElement']
    loc_pairs = {}
    for e in tree.iter():
        try:
            _type = e.attrib['type']
        except KeyError:
            continue   
        if _type not in exclude:
            pos = int(e.attrib['pos'])
            length = int(e.attrib['length'])
            end = pos + length
            try:
                other_pos = int(e.attrib['other_pos'])
            except KeyError: # meaning, this element is deleted, thereby skipped
                continue
            other_length = int(e.attrib['other_length'])
            other_end = other_pos + other_length 
            loc_pairs[(pos, end)] = (other_pos, other_end)
    return loc_pairs

def mapPositions_fromList(
    missed_infixOps:List[ET.Element]
) -> Dict:
    """
    exclude
    """
    data = {
        'prev_type':[], 'curr_type':[],
        'prev_pos':[], 'curr_pos':[], 
        'prev_content':[], 'curr_content': []    
    }
    for e in missed_infixOps:
        e_type = e.attrib['type']
        pos = int(e.attrib['pos'])
        length = int(e.attrib['length'])
        end = pos + length
        label = e.attrib['label'] # since this is only for infixOp, always have label
        if 'other_pos' in e.attrib.keys():
            other_pos = int(e.attrib['other_pos'])
            other_length = int(e.attrib['other_length'])
            other_end = other_pos + other_length 
            other_label = e.attrib['other_label']
        else: # deleted -> for this... we will skip it as for this one is deleted (the same procedure was done in mapPositions and diffMaps (here, automatically excluded by RMiiner)
            continue 

        data['prev_type'].append(e_type)
        data['prev_pos'].append([pos, end])
        data['prev_content'].append(label)
        #
        data['curr_type'].append(e_type)
        data['curr_pos'].append([other_pos, other_end])
        data['curr_content'].append(other_label)
    return data
