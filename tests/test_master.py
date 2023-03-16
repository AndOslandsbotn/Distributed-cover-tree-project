import ray
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
import numpy as np
import os
from definitions import ROOT_DIR

from distributed_covertree.master import Master
from time import perf_counter
from scipy.stats import uniform

from definitions import CONFIG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)

def test_master(data_filepath, exp_name = 'default', dim=2, num_batches=100, batch_size=1000, radius = 1.2, center = [0, 0]):
    def create_test_data_for_cover_test(dim, num_batches, batch_size):
        data = []
        for i in range(num_batches):
            data.append(uniform.rvs(loc=-1, scale=2, size=(batch_size, dim)))  # Uniform dist on [loc, loc+scale]
        return data
    #data = create_test_data_for_cover_test(dim=dim, num_batches=num_batches, batch_size=batch_size)

    lvl = 0
    start = perf_counter()
    tree = Master.remote(data_filepath, radius, center, lvl, exp_name=exp_name)
    ray.get(tree.main.remote(tree))
    stop = perf_counter()
    print("FINISHED RUNNING TREE")
    print("execution_time: ", stop - start)

if __name__ == '__main__':
    data_size = 10000
    data_filepath = os.path.join(ROOT_DIR, 'Data', f'DummyData_n{data_size}.npy')
    test_master(data_filepath=data_filepath, exp_name = 'defult', dim=2,
                num_batches=10000, batch_size=CONFIG['performance']['splitterq_batch_size'],
                radius=1.42, center=np.array([0, 0]))
