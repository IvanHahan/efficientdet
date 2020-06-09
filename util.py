import os


def make_dir_if_needed(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def root_dir(file):
    return os.path.abspath(os.path.dirname(file))
