
from collections import OrderedDict
from os import path
import yaml

def overwrite_yaml_construction():

    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(yaml_file_path, is_train=True):
    
    #### transform yaml conf to ordered dict
    with open(yaml_file_path, mode='r') as f:
        Loader, _ = overwrite_yaml_construction()
        envir_conf = yaml.load(f, Loader=Loader)

    return envir_conf

