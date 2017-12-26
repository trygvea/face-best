import numpy as np

def save_dict(file_name, dict):
    np.save(file_name, dict)

def load_dict(file_name):
    return np.load(file_name).item()


def dict_minus_immutable(dict, key):
    d = dict.copy()
    d.pop(key)
    return d


def remove_top_10_pct(numbers):
    return None