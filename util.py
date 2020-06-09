import os
from .definitions import ROOT_DIR


def make_dir_if_needed(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def abs_path(path):
    return os.path.join(ROOT_DIR, path)


def root_dir(file):
    return os.path.abspath(os.path.dirname(file))
