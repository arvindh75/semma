
import argparse
import yaml
import jinja2
from jinja2 import meta
import easydict
import ast
import os

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_flags(file_path):
    """
    Reads a YAML file and returns a configuration object that allows access to its fields with dot notation.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Config: An object representing the YAML configuration.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return Config(**data)

def detect_variables(cfg_file, root=None):
    if(root != None):
        cfg_file = os.path.join(root, cfg_file)
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars

def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string
    
def load_config(cfg_file, context=None, root=None):
    if(root != None):
        cfg_file = os.path.join(root, cfg_file)
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg

def parse_args(root=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    if(root != None):
        vars = detect_variables(os.path.join(root, args.config))
    else:
        vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars