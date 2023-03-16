import os
from pathlib import Path

import numpy as np
from definitions import ROOT_DIR, GB, MB
from tqdm import tqdm
#from utilities.read_write import load_data_on_disk
from scipy.stats import uniform


def generate_dummy_data(t, n, d, filepath, filename):
    Path(filepath).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(filepath, filename)
    #f = np.memmap(filepath, dtype=np.float64, mode='w+', shape=(n*t, d))
    f = np.lib.format.open_memmap(filepath, dtype=np.float64, mode='w+', shape=(n*t, d))
    for i in tqdm(range(t)):
        #x = np.linspace(0, 1000, n)
        #y = np.linspace(0, 1000, 2)
        #xv, yv = np.meshgrid(y, x)
        t = uniform.rvs(loc=-1, scale=2, size=(n, d))
        #f[n*i:n*(i+1), :] = yv #np.random.rand(n, d)
        f[n * i:n * (i + 1), :] = t
    del f
    print(f'X array: {os.path.getsize(filepath):12} bytes ({os.path.getsize(filepath)/GB:3.2f} GB)')
