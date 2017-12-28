import time
import numpy as np
import pprint

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


# Timings global (no habit, but nice here)
timings = {}

def print_timings():
    if len(timings) > 0:
        print("Timings:")
        pprint.pprint(timings)

import atexit
atexit.register(print_timings)


def timing(f):
    name = f.__name__
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

