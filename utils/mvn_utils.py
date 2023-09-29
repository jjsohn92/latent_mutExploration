import os, sys
import xml.etree.ElementTree as ET
import numpy as np 
from typing import List, Dict, Tuple 

CLASSDIR = "target/classes"
TESTCLASSDIR = "target/test-classes"

def get_tree_root_and_root_tag(pom_file_or_content:str, is_content:bool = False):
    tree = ET.parse(pom_file_or_content) if not is_content else ET.ElementTree(ET.fromstring(pom_file_or_content))
    root = tree.getroot()
    tag_to_root = root.tag.split("}")[0] + "}"
    return tree, root, tag_to_root 

def get_all_pom_file(repo_dir:str) -> Dict[str,str]:
    pass 

def get_pom_file(workdir:str):
    return os.path.join(workdir, "pom.xml")

def get_junit_version(pom_file_or_content:str, is_content:bool = False) -> str:
    """
    """
    tree, root, tag_to_root = get_tree_root_and_root_tag(pom_file_or_content, is_content=is_content)
    tag_to_dependencies = "{}dependencies".format(tag_to_root)
    tag_to_dependency = "{}dependency".format(tag_to_root)
    tag_to_artifactId = "{}artifactId".format(tag_to_root)
    tag_to_version = "{}version".format(tag_to_root)

    junit_version = None
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
                    junit_version= version_nodes[0].text 
                    break
                else:
                    continue
        if junit_version is not None:
            break 
    return junit_version

def is_eqOrHigherThanJunit4(junit_ver:str):
    ver = int(junit_ver.split(".")[0])
    return ver >= 4

## temporary due to compilation error
def preprocess_lang(workdir:str):
    # TypeUtilsTest.java -> move as always occur
    import os 
    import shutil
    tempdir = os.path.join(workdir, "temp")
    os.makedirs(tempdir, exist_ok=True)
    for root , dirs, files in os.walk(workdir):
      for file in files:
        if file == 'TypeUtilsTest.java':
            shutil.move(os.path.join(root, file), os.path.join(tempdir, file))


def checkCobertura_or_Jacoco(
    pom_file_or_content:str, is_content:bool = False
) -> Tuple[bool, str]:
    """
    check which one is used to compute the coverage. 
    If nothing exists, add cobertura (but.. then checkout problem...)
    # -> for now, no adding 
    C -> has cobertura
    J -> has jacoco
    """
    _, root, tag_to_root = get_tree_root_and_root_tag(
        pom_file_or_content, is_content=is_content)

    build_node = root.find(f"{tag_to_root}build")
    plugins_node = build_node.find(f"{tag_to_root}plugins")
    has_cobertura_plugin = False
    for plugin_node in plugins_node.findall(f"{tag_to_root}plugin"):
        artifactId_node = plugin_node.find(f"{tag_to_root}artifactId")
        if artifactId_node is None:
            continue
        if artifactId_node.text == 'cobertura-maven-plugin':
            has_cobertura_plugin = True 
            break
    
    if has_cobertura_plugin: 
        return (True, 'C')
    else:
        properties_node = root.find(f"{tag_to_root}properties")
        jacoco_node = properties_node.find(f"{tag_to_root}commons.jacoco.version")
        if jacoco_node is not None:
            return (True, 'J')
        else:
            return (False, None) # has nothing ->

def addIncremental(pom_file_or_content:str, is_content:bool = False):
    _, root, tag_to_root = get_tree_root_and_root_tag(
        pom_file_or_content, is_content=is_content)
    
    build_node = root.find(f"{tag_to_root}build")
    plugins_node = build_node.find(f"{tag_to_root}plugins")
    pass 

def add_disable_dependency_report_plugin():
    """
    should be inserted under <reporting> <plugins>
    """
    disable_dep_report_plugin_text = "  plugin>\n\
        <groupId>org.apache.maven.plugins</groupId>\n\
        <artifactId>maven-project-info-reports-plugin</artifactId>\n\
        <version>2.7</version>\n\
        <configuration>\n\
            <dependencyLocationsEnabled>false</dependencyLocationsEnabled>\n\
        </configuration>\n\
      </plugin>"
    pass 


def add_cobertura_plugin():
    """
    should be inserted under <reporting> <plugins>
    """
    cobertura_plugin_text = "  <plugin>\n\
	    <groupId>org.codehaus.mojo</groupId>\n\
	    <artifactId>cobertura-maven-plugin</artifactId>\n\
	    <version>2.6</version>\n\
	    <configuration>\n\
		  <formats>\n\
		    <format>html</format>\n\
			<format>xml</format>\n\
		  </formats>\n\
	    </configuration>\n\
	  </plugin>"
    pass


def change_cobertura_report_to_xml():
    pass 

def add_jacoco_plugin():
    """
    on-the-fly => thus, can avoid ...
    """
    jacoco_plugin_text = "  <plugin>\n\
        <groupId>org.jacoco</groupId>\n\
          <artifactId>jacoco-maven-plugin</artifactId>\n\
          <version>0.8.2</version>\n\
          <executions>\n\
            <execution>\n\
              <goals>\n\
                <goal>prepare-agent</goal>\n\
              </goals>\n\
            </execution>\n\
            <!-- attached to Maven test phase -->\n\
            <execution>\n\
              <id>report</id>\n\
              <phase>test</phase>\n\
              <goals>\n\
                <goal>report</goal>\n\
              </goals>\n\
            </execution>\n\
          </executions>\n\
      </plugin>"
    pass 

def add_RMiner_plugin(pom_file:str, is_content:bool = False):
    """
    under dependencies node
    """
    rminer_plugin = "    <dependency>\n\
      <groupId>com.github.tsantalis</groupId>\n\
      <artifactId>refactoring-miner</artifactId>\n\
      <version>2.4.0</version>\n\
    </dependency>"

    tree, root, tag_to_root = get_tree_root_and_root_tag(pom_file, is_content=is_content)
    tag_to_dependencies = "{}dependencies".format(tag_to_root)
    tag_to_dependency = "{}dependency".format(tag_to_root)

    depds_nodes = root.findall(tag_to_dependencies)
    assert len(depds_nodes) == 0, tag_to_dependencies 
    depds_node = depds_nodes[0]
    depd_nodes = depds_node.findall(tag_to_dependency)
    # gen 
    rminer_plugin_node = ET.fromstring(rminer_plugin)
    depd_nodes.append(rminer_plugin_node)

    # add
    import shutil
    pom_dir = os.path.dirname(pom_file); pom_basename = os.path.basename(pom_file)
    backup_pom_file = os.path.join(pom_dir, "backup-{}".format(pom_basename))
    print (f"backup pom file ({pom_file}) as {backup_pom_file}")
    shutil.copy2(pom_file, backup_pom_file)
    # overwrite the current 
    rewrite_pom(tree, pom_file)


def rewrite_pom(tree:ET.ElementTree, pom_fpath:str):
    import shutil
    pom_dir = os.path.dirname(pom_fpath); pom_basename = os.path.basename(pom_fpath)
    backup_pom_fpath = os.path.join(pom_dir, "backup-{}".format(pom_basename))
    print (f"backup pom file ({pom_fpath}) as {backup_pom_fpath}")
    shutil.copyfile(pom_fpath, backup_pom_fpath)
    # overwrite the current 
    tree.write(pom_fpath, method='xml', xml_declaration=True)

def compile_tests(work_dir:str = None):
    import subprocess 
    from subprocess import CalledProcessError
    test_compile_cmd = "mvn test-compile -DskipTests=true"
    try:
        output = subprocess.check_output(
            test_compile_cmd, shell = True, cwd = work_dir if work_dir is not None else "."
            ).decode('utf-8', 'backslashreplace')
    except CalledProcessError as e: # ... this actually will also catch a simple test failure
        print (test_compile_cmd)
        print (e)
        return None, e
    return output, None

