from tests.test_master import test_master
import numpy as np
import json
import os

from definitions import CONFIG_PATH, ROOT_DIR
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)
if __name__ == '__main__':
    print(json.dumps(CONFIG, indent=4, sort_keys=True))

if __name__ == '__main__':
    data_size = 10000
    data_filepath = os.path.join(ROOT_DIR, 'Data', f'DummyData_n{data_size}_d2.npy')
    test_master(data_filepath=data_filepath, exp_name='Experiment1', dim=2, num_batches=1000,
                batch_size=CONFIG['performance']['splitterq_batch_size'],
                radius=1.42, center=np.array([0, 0]))
