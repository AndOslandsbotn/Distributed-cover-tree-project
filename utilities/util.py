import numpy as np
from scipy.spatial.distance import cdist
import cpuinfo

def _dist( p1, p2):
    """
    :param p1: 1-dim np array
    :param p2: 1-dim np array
    :return: distance between p1 and p2
    """
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=0))

def calc_dist_matrix(items, neighbour_centers):
    "Calculate distance between each pair of items and neighbour_centers"
    assert type(items) == np.ndarray, "items need to be ndarray"
    assert items.ndim == 2, "items needs to be a 2 dimensional numpy array"
    assert neighbour_centers.ndim == 2, "neighbour_centers needs to be a 2 dimensional numpy array"
    assert len(neighbour_centers) > 0
    return cdist(items, neighbour_centers)

def get_catche_size():
    """Returns catche size in bytes"""
    cpuinfo_dict = cpuinfo.get_cpu_info()
    l2c = cpuinfo_dict['l2_cache_size']
    l3c = cpuinfo_dict['l3_cache_size']
    return l2c, l3c


