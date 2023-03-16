from utilities.data_generators import generate_dummy_data
import os
from definitions import ROOT_DIR

if __name__ == '__main__':
    n = 1000
    d = 2
    t = 10

    filepath = os.path.join(ROOT_DIR, 'Data')
    filename = f'DummyData_n{int(t*n)}_d{d}.npy'
    generate_dummy_data(t, n, d, filepath, filename)
