import numpy as np
import torch
import random
import logging
import argparse

def get_objects(modules):
    objects = {}
    for name in dir(modules):
        obj = getattr(modules, name)
        if isinstance(obj, type):
            objects[name] = obj
    return objects

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    # torch.cuda.manual_seed(i) # Removed to support CPU-only environments

def arg_parser():
    parser = argparse.ArgumentParser()
    return parser

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()