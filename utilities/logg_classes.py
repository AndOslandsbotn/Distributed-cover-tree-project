import numpy as np
from pathlib import Path
from os import path
import os
from time import time
import asyncio

from utilities.read_write import save_csv_async, save_pickle_async, save_csv, save_json, read_pickle_file

from definitions import ROOT_DIR, CONFIG_PATH, LOG_CFG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)
import logging.config
import sys

def setup_logger(logger_name, log_cfg_path, default_level=logging.INFO):
    """Setup logger from configuration file"""
    logger = logging.getLogger(logger_name)
    if os.path.exists(log_cfg_path):
        logging.config.dictConfig(yaml_loader(log_cfg_path))
    else:
        logger = logging.basicConfig(level=default_level)
    return logger

def init_logger(logger_name, logg_filepath, logg_filename, logger_level):
    """Setup logger"""
    Path(logg_filepath).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(logg_filepath, logg_filename), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


class NodePerformanceLogger():
    def __init__(self, filepath):
        self.logg_folder = os.path.join(ROOT_DIR, CONFIG['general']['logg_folder'], filepath)
        Path(self.logg_folder).mkdir(parents=True, exist_ok=True)

    def logg(self, time, filename):
        save_csv(path.join(self.logg_folder, filename), time, operation='a')


class ResultsLogger():
    def __init__(self, config, exp_name):
        self.config = config
        self.filepath = os.path.join(ROOT_DIR, config['general']['results_folder'], exp_name)
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

    def logg_summary(self, time):
        info = {}
        info['exec_time'] = time
        info['config'] = self.config
        save_json(info, self.filepath, 'summary', operation='w')

    async def logg_nodes(self, lvls, centers, radiuss):
        node_info = zip(lvls, centers, radiuss)
        await save_pickle_async(path.join(self.filepath, 'nodes'), node_info, operation='ab+')

    def load_nodes(self):
        return read_pickle_file(self.filepath, 'nodes')


class WorkerPerformanceLogger():
    def __init__(self, passthroughbytes_counter, worker_queue, worker_id, worker_type):
        self.worker_type = worker_type
        self.worker_id = worker_id
        self.worker_queue = worker_queue
        self.passthroughbytes_counter = passthroughbytes_counter
        self.qsize_counter = []

        self.logg_folder = os.path.join(ROOT_DIR, CONFIG['general']['logg_folder'],
                                        f"sq_bsz{CONFIG['performance']['splitterq_batch_size']}_" +
                                        f"lq_bsz{CONFIG['performance']['learnerq_batch_size']}")
        self.file_ptb = os.path.join(self.logg_folder, self.worker_type + f'{self.worker_id}' + 'ptbytes.csv')
        self.file_qsize = os.path.join(self.logg_folder, self.worker_type + f'{self.worker_id}' + 'qsize.csv')
        Path(self.logg_folder).mkdir(parents=True, exist_ok=True)

        self.logg_freq = CONFIG['performancelogging']['logg_freq']
        self.qsize_ping_freq = CONFIG['performancelogging']['qsize_ping_freq']

    async def write_column_names(self):
        await save_csv_async(self.file_ptb, ['time', 'ptbytes'], operation='w')
        await save_csv_async(self.file_qsize, ['time', 'qsize'], operation='w')

    async def logging(self):
        await self.write_column_names()
        while True:
            await asyncio.sleep(1 // self.logg_freq)
            await save_csv_async(self.file_ptb, [time(), self.passthroughbytes_counter[0]], operation='a')
            await save_csv_async(self.file_qsize, [time(), np.mean(self.qsize_counter)], operation='a')
            self.qsize_counter = []

    async def check_qsize(self):
        while True:
            await asyncio.sleep(1 // self.qsize_ping_freq)
            qsize = await self.worker_queue.actor.qsize.remote()
            self.qsize_counter.append(qsize)


