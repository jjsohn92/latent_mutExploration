import javalang
from typing import List, Dict, Tuple 

def build_lno_pos_dict(fpath_or_file_content:str, is_path:bool) -> Dict[int, Tuple[int,int]]:
    accumulated_pos, lno_pos_dict = 0, {}
    if is_path:
        with open(fpath_or_file_content) as f:
            for i, line in enumerate(f.readlines()):
                lno_pos_dict[i + 1] = (accumulated_pos, accumulated_pos + len(line)) # [x:y]
                accumulated_pos += len(line)
    else:
        for i, line in enumerate(fpath_or_file_content.split("\n")):
            lno_pos_dict[i + 1] = (accumulated_pos, accumulated_pos + len(line) + 1)
            accumulated_pos += len(line) + 1
    return lno_pos_dict

def get_final_line(node: javalang.tree.Node) -> int:
    # traverse node and find the max line
    import contextlib
    max_line = 0
    for path, child in node.filter(javalang.tree.Node): # will output in order
        if isinstance(child, javalang.tree.BlockStatement): break 
        with contextlib.suppress(TypeError):
            max_line = max(max_line, child.position[0]) 
    return max_line

def get_start_and_end_lnos(node: javalang.tree.Node) -> Tuple[int, int]:
    start_lno = node.position.line 
    end_lno =get_final_line(node)
    return start_lno, end_lno

def parse(file_content:str):
    tree = javalang.parse.parse(file_content)
    return tree 

def build_positon_dict(tree: javalang.tree.CompilationUnit) -> Tuple[Dict, Dict]:
    build_pos_dict = {} # for 
    lno_to_node_dict = {}
    targeted_types = [
        javalang.tree.Import, javalang.tree.Declaration, javalang.tree.Statement
    ]
    is_target = lambda v: any([isinstance(v, t) for t in targeted_types])
    for _, node in tree:
        if not is_target(node): continue
        if node.position is not None:
            start_lno, end_lno = get_start_and_end_lnos(node)
            build_pos_dict[node] = (start_lno, end_lno)
            for lno in range(start_lno, end_lno + 1):
                try:
                    lno_to_node_dict[lno].append(node)
                except KeyError:
                    lno_to_node_dict[lno] = [node]
    return build_pos_dict, lno_to_node_dict

def get_lno_positions_of_chgd_nodes_v1(target_lno:int, build_pos_dict:Dict, lno_to_node_dict:Dict) -> List[int]:
    # will take the smallest 
    chgd_nodes = lno_to_node_dict[target_lno]
    min_end_lno = None
    for chgd_node in chgd_nodes:
        _, end_lno = build_pos_dict[chgd_node]
        if min_end_lno is None:
            min_end_lno = end_lno 
        elif min_end_lno > end_lno:
            min_end_lno = end_lno  
    return list(range(target_lno, min_end_lno + 1))

def get_lno_positions_of_chgd_nodes(
    target_lno:int, build_pos_dict:Dict, lno_to_node_dict:Dict
) -> List[int]:
    if target_lno in lno_to_node_dict.keys():
        chgd_nodes = lno_to_node_dict[target_lno]
        min_dist_to_target_lno = None
        closest_chgd_node = None
        for chgd_node in chgd_nodes:
            min_lno, end_lno = build_pos_dict[chgd_node]
            dist_to_target_lno = target_lno - min_lno
            #print (min_lno, dist_to_target_lno)
            if min_dist_to_target_lno is None: 
                min_dist_to_target_lno = dist_to_target_lno
                closest_chgd_node = chgd_node
            elif min_dist_to_target_lno > dist_to_target_lno:
                min_dist_to_target_lno = dist_to_target_lno
                closest_chgd_node = chgd_node 
        start_lno_of_chgd, end_lno_of_chgd = build_pos_dict[closest_chgd_node]
        rets = list(range(start_lno_of_chgd, end_lno_of_chgd + 1))
    else:
        min_dist_to_target_lno = None
        closest_chgd_node = None 
        #min_end_lno = None
        for lno, chgd_nodes in lno_to_node_dict.items():
            if lno < target_lno: # the element start after the targeted line
                for chgd_node in chgd_nodes:
                    min_lno, end_lno = build_pos_dict[chgd_node]
                    if end_lno < target_lno: continue # not our target
                    dist_to_target_lno = target_lno - min_lno
                    if min_dist_to_target_lno is None: 
                        min_dist_to_target_lno = dist_to_target_lno
                        closest_chgd_node = chgd_node
                    elif min_dist_to_target_lno > dist_to_target_lno:
                        min_dist_to_target_lno = dist_to_target_lno
                        closest_chgd_node = chgd_node 
        if closest_chgd_node is not None:
            start_lno_of_chgd, end_lno_of_chgd = build_pos_dict[closest_chgd_node]
            rets = list(range(start_lno_of_chgd, end_lno_of_chgd + 1))
        else:
            import numpy as np 
            processed_lines = np.array(sorted(list(lno_to_node_dict.keys()))) # in ascending order 
            idx_to_closed_bfr_chgd_node = np.where(processed_lines <= target_lno)[0].max() # 
            closest_lno = processed_lines[idx_to_closed_bfr_chgd_node]
            cand_bfr_chgd_nodes = lno_to_node_dict[closest_lno]
            cand_start_lnos = []
            for c in cand_bfr_chgd_nodes:
                cand_start_lnos.append(c.position.line)
            start_lno = min(cand_start_lnos)
            rets = list(range(start_lno, target_lno + 1))
    return rets 