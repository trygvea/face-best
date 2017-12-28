import time
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

def timing(timings, name):
    def timing_decorator(f):
        def wrap(*args):
            start = time.time()
            ret = f(*args)
            took = time.time() - start
            if name not in timings:
                timings[name] = {"invocations": 0, "totaltime": 0}
            timings[name]["totaltime"] += took
            timings[name]["invocations"] += 1
            return ret
        return wrap
    return timing_decorator



