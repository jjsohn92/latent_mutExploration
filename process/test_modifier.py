"""
"""
import os, sys 
sys.path.append("..")
from utils.parser_utils import get_final_line, get_start_and_end_lnos, parse
#from utils.file_utils import fileWrite, readFile
import javalang 
from typing import Tuple, List

def get_content_within_pos(file_content:str, start_lno:int, end_lno:int) -> str:
  lines = file_content.split("\n")
  return "\n".join(lines[start_lno-1:end_lno])

def get_method_body(file_content:str, start_lno:int, end_lno:int) -> Tuple[str, List[int]]:
  """
  this is something that needs to be implemented
  """
  lines = file_content.split("\n")
  init_body = "\n".join(lines[start_lno-1:end_lno]) 
  cnt_close_bracket = init_body.count("}")
  cnt_open_bracket = init_body.count("{")

  n_missing_bracket = cnt_open_bracket - cnt_close_bracket
  additional = []
  for line in lines[end_lno:]: 
    additional.append(line)
    if "}" in line:
      n_missing_bracket -= 1 # also need to update  
    if "{" in line: 
      n_missing_bracket += 1
    # check whether to end
    if n_missing_bracket == 0:
      break 
  final_lnos = list(range(start_lno, end_lno + len(additional) + 1))
  init_body += "\n" + "\n".join(additional)
  return init_body, final_lnos

def get_package_name(file_content:str) -> str:
  tree = parse(file_content)
  return tree.packagae.name if tree.package is not None else None

def get_body_of_a_test(file_content:str, ftest:str) -> Tuple[Tuple[int,int], str]:
  """
  :param file_content: a file content to parse
  :param ftest: a test id, e.g., {test_class}#{testMethod}

  return ((start_pos, end_pos), method_body)
  """ 
  full_test_cls, test_mth = ftest.split("#")
  test_cls = full_test_cls.split(".")[-1]
  tree = parse(file_content)
  
  package_name = None if tree.package is None else tree.package.name 
  parsed = {"package": package_name, "classes": []} 
  # loop and retreive 
  for path, node in tree.filter(javalang.tree.ClassDeclaration):
    class_decl_node: javalang.tree.ClassDeclaration = node 
    class_pos = (class_decl_node.position[0], get_final_line(class_decl_node))
    class_dict = {'name':class_decl_node.name, 'pos':class_pos}
    if class_decl_node.name == test_cls:
      full_cls_name = package_name
      method_pos = None
      for method_decl_node in class_decl_node.methods:
        if method_decl_node.name == test_mth:
          method_pos = (method_decl_node.position[0], get_final_line(method_decl_node))
          break

      if method_pos is None: # error 
        return None 
      else:
        for prev_class_dict in parsed['classes']: 
          if prev_class_dict['pos'][-1] > class_dict['pos'][-1]: # 
            if full_cls_name is not None: 
              full_cls_name += "." + prev_class_dict['name']
            else:
              full_cls_name == prev_class_dict['name']
        full_cls_name += "." + class_dict['name']
        if f"{full_cls_name}#{test_mth}" == ftest: 
          # .. 
          #method_body = get_content_within_pos(file_content, method_pos[0], method_pos[1])
          method_body, _ = get_method_body(file_content, method_pos[0], method_pos[1])
          return method_body, method_pos
    #
    parsed['classes'].append(class_dict)
  return None 


def inject_a_new_test(file_content:str, ftest:str, new_ftest_body:str) -> str:
  """
  inject ftest_body at the start of method list ...
  To be safe, append it at the end of methods 

  :param file_content:
  :param ftest: the target ftest id at file_content
  """
  tree = parse(file_content)
  ftestcls_name = ftest.split("#")[0].split(".")[-1]
  package_name = None if tree.package is None else tree.package.name 
  #target_start_end_pos = None
  # get the target class node
  ftest_class_node, parsed = None, {"classes": []} 
  for path, node in tree.filter(javalang.tree.ClassDeclaration):
    class_decl_node: javalang.tree.ClassDeclaration = node 
    class_pos = (class_decl_node.position[0], get_final_line(class_decl_node))
    class_dict = {'name':class_decl_node.name, 'pos':class_pos}
    if class_decl_node.name == ftestcls_name:
      full_cls_name = package_name # init 
      # get full cls namee
      for prev_class_dict in parsed['classes']: 
        if prev_class_dict['pos'][-1] > class_dict['pos'][-1]: # 
          full_cls_name += "." + prev_class_dict['name']
      full_cls_name += "." + class_dict['name']
      if full_cls_name == ftest.split("#")[0]: # check 
        #target_start_end_pos = class_dict['pos']
        ftest_class_node = class_decl_node
        break 
    parsed['classes'].append(class_dict)
  # prepare new test body
  new_ftest_body_w_anno = ["@Test"] + new_ftest_body.split("\n") 
  
  # get the location 
  first_mth_node = ftest_class_node.methods[0]
  cnt_annotations = len(first_mth_node.annotations)
  start_of_mth = first_mth_node.position[0] # 
  start_of_mth_w_anno = start_of_mth - cnt_annotations
  idx_to_start_of_mth_w_anno = start_of_mth_w_anno - 1 

  # inject
  lines = file_content.split("\n") 
  new_lines = lines[:idx_to_start_of_mth_w_anno] + new_ftest_body_w_anno + lines[idx_to_start_of_mth_w_anno:]
  
  return "\n".join(new_lines)

def replace_a_test(file_content:str, ftest:str, new_ftest_body:str) -> str:
  """
  
  :param file_content:
  :param ftest: the target ftest id at file_content
  :param ftest_pos: starting of the test method 
  """
  tree = parse(file_content)
  ftestcls_name = ftest.split("#")[0].split(".")[-1]
  package_name = None if tree.package is None else tree.package.name 

  #target_start_end_pos = None
  # get the target class node
  ftest_class_node, parsed = None, {"classes": []} 
  for path, node in tree.filter(javalang.tree.ClassDeclaration):
    class_decl_node: javalang.tree.ClassDeclaration = node 
    class_pos = (class_decl_node.position[0], get_final_line(class_decl_node))
    class_dict = {'name':class_decl_node.name, 'pos':class_pos}
    if class_decl_node.name == ftestcls_name:
      full_cls_name = package_name # init 
      # get full cls namee
      for prev_class_dict in parsed['classes']: 
        if prev_class_dict['pos'][-1] > class_dict['pos'][-1]: # 
          full_cls_name += "." + prev_class_dict['name']
      full_cls_name += "." + class_dict['name']
      if full_cls_name == ftest.split("#")[0]: # find the node 
        ftest_class_node = class_decl_node
        break 
    parsed['classes'].append(class_dict)

  # find the test method
  test_mth = ftest.split("#")[1]
  target_mth_decl_node = None
  for method_decl_node in ftest_class_node.methods:
    if method_decl_node.name == test_mth:
      target_mth_decl_node = method_decl_node
      break 
  if target_mth_decl_node is None: # error
    return None 
  
  # get the position to replace
  _, mth_lnos = get_method_body(file_content, target_mth_decl_node.position[0], get_final_line(target_mth_decl_node))
  
  # prepare new test body
  new_ftest_body_w_anno = ["@Test"] + new_ftest_body.split("\n") 

  # replace
  lines = file_content.split("\n") 
  new_lines = lines[:mth_lnos[0] - 1] + new_ftest_body_w_anno + lines[mth_lnos[-1]:]

  return "\n".join(new_lines)

if __name__ == "__main__":
  # some examples here 
  pass   