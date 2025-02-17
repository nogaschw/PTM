import ast
import random
import numpy as np
import pandas as pd
from anytree import Node
from anytree.walker import Walker
from anytree.search import findall_by_attr

class Config:
    def __init__(self):
        self.code_path_length = 10
        self.code_path_width = 5
        self.path = "OriginalData/falcon/cleaned_code.csv"

def get_token(node):
    if isinstance(node, ast.AST):
        return type(node).__name__
    elif isinstance(node, str):
        return node
    elif isinstance(node, list):
        return 'List'
    else:
        return 'Unknown'

def get_children(node):
    if isinstance(node, ast.AST):
        return [getattr(node, field) for field in node._fields if getattr(node, field) is not None]
    elif isinstance(node, list):
        return node
    else:
        return []

def expand(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from expand(item)
        elif item:
            yield item

def get_trees(current_node, parent_node, order):
    token = get_token(current_node)
    children = list(expand(get_children(current_node)))
    node = Node([order, token], parent=parent_node, order=order)

    for child_order, child in enumerate(children):
        get_trees(child, node, order + str(child_order + 1))

def get_path_length(path):
    return len(path)

def get_path_width(raw_path):
    return abs(int(raw_path[0][-1].order) - int(raw_path[2][0].order))

def hashing_path(path, hash_table):
    if path not in hash_table:
        hash_value = random.getrandbits(128)
        hash_table[path] = str(hash_value)
        return str(hash_value)
    else:
        return hash_table[path]

def get_node_rank(node_name, max_depth):
    while len(node_name[0]) < max_depth:
        node_name[0] += "0"
    return [int(node_name[0]), node_name[1]]

def extracting_path(python_code, max_length, max_width, hash_path, hashing_table):
    tree = ast.parse(python_code)
    head = Node(["1", get_token(tree)])

    for child_order, child in enumerate(get_children(tree)):
        get_trees(child, head, "1" + str(child_order + 1))

    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
    max_depth = max(len(node.name[0]) for node in leaf_nodes)

    for leaf in leaf_nodes:
        leaf.name = get_node_rank(leaf.name, max_depth)

    walker = Walker()
    text_paths = []

    for leaf_index in range(len(leaf_nodes) - 1):
        for target_index in range(leaf_index + 1, len(leaf_nodes)):
            raw_path = walker.walk(leaf_nodes[leaf_index], leaf_nodes[target_index])
            walk_path = [n.name[1] for n in list(raw_path[0])] + [raw_path[1].name[1]] + [n.name[1] for n in list(raw_path[2])]
            text_path = "@$".join(walk_path)

            if get_path_length(walk_path) <= max_length and get_path_width(raw_path) <= max_width:
                if not hash_path:
                    text_paths.append(walk_path[0] + "," + text_path + "," + walk_path[-1])
                else:
                    text_paths.append(walk_path[0] + "," + hashing_path(text_path, hashing_table) + "," + walk_path[-1])

    return text_paths

def program_parser(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None

config = Config()
main_df = pd.read_csv(config.path)

parsed_code = []
j = 0
for i, code in enumerate(list(main_df['clean_code'])):
    print(f"{i} \ {len(list(main_df['clean_code']))}", end='\r')
    parsed = program_parser(code)
    if parsed == None:
        if j == 0:
            print(code)
        j += 1
    parsed_code.append(parsed if parsed is not None else "Uncompilable")
print(f"cant find {j} \ {len(list(main_df['clean_code']))}")
hashing_table = {}

AST_paths = [extracting_path(python_code, max_length=config.code_path_length, max_width=config.code_path_width, hash_path=True, hashing_table=hashing_table) for python_code in parsed_code]
main_df["RawASTPath"] = ["@$".join(paths) for paths in AST_paths]    
main_df.to_csv("labeled_paths_all_python.tsv", sep="\t", header=True)