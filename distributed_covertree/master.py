import ray
from ray.util.queue import Queue
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
import itertools

from distributed_covertree.workers import LearningWorker, SplittingWorker
import asyncio
import numpy as np
from time import perf_counter

from utilities.read_write import StreamDataFromFile
from utilities.logg_classes import ResultsLogger, setup_logger
from scipy.stats import uniform
import json
from pathlib import Path

from definitions import CONFIG_PATH, ROOT_DIR, LOG_CFG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)

if __name__ == '__main__':
    print(json.dumps(CONFIG, indent=4, sort_keys=True))

import logging
import os

@ray.remote
class Master():
    """ The master class for constructing the epsilon cover tree.
    A ray actor which acts as the master. Sets up the program and
    is the owner of the input queues to the CoverTreeWorkers

    Objects:
    - data_reader: Reads data from file in batches
    - lrn_workers: A dictionary of instances of the LearningWorker class
    - splitterWorkers: A dictonary of instances of the SplittingWorker class

    Responsibilities
    - Checks available resources and instansiates the ray worker and splitter classes
    - Connect data reader to splitter
    - Keeps track of worker loads
    - Assign new nodes to the worker instance with least load

    Public methods:
    * main: Main function to be called to build the epsilon cover

    Private methods:
    * main_loop: Runs until the data_reader raises EOF and there are no remaining points in the worker and splitter queues.
    * monitor: Check number of points in worker queue every 1/q_ping_time seconds
    * check_queue_size: Checks the size of the worker queue of worker with id: worker_id
    * get_learning_worker_id: Returns the worker_id of the worker with the least load
    * get_splitting_worker_id: Returns the splt_worker_id of the splitter with the least load
    * set_ref_to_self: Sets a reference to the ray actor instance of the Master instance
    * check_available_resources
    * employ_workers: Instansiates the workers and their input queues
    * employ_splitters: Instansiates the splitters and input queue and outlier queues of the splitters

    Callbacks:
    * callback: Callback from the data_reader to inform that the end of the data file has been reached.
    * add_children: Each LearningNode instance makes a callback to add_children when their learning is complete, with a
    list of desired children. add_children assigns these children centers to the appropriate workers to start learning.
    """

    def __init__(self, filepath, init_radius, init_center, init_lvl, exp_name):
        self.start_time = 0
        self.exp_name = exp_name  # Experiment name

        self.data_reader = StreamDataFromFile.remote(filepath=filepath,
                                                     batch_size=CONFIG['performance']['splitterq_batch_size'],
                                                     loops=1)

        self.results_logger = ResultsLogger(CONFIG, exp_name)


        Path('Logg').mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(logger_name='Master', log_cfg_path=LOG_CFG_PATH)
        self.num_nodes = 0

        self.reduce_factor = CONFIG['general']['radius_reduce_factor']
        self.init_radius = init_radius  # Estimate of the span of the data
        self.init_center = init_center
        self.init_lvl = init_lvl

        self.num_lw = CONFIG['performance']['num_learning_workers']
        self.num_sw = CONFIG['performance']['num_splitting_workers']
        self.lrn_worker_loads = np.zeros(self.num_lw)
        self.splt_worker_loads = np.zeros(self.num_sw)

        self.splt_workers = {}
        self.track_outlier_qs = {}
        self.track_splt_worker_qs = {}

        self.lrn_workers = {}
        self.track_lrn_worker_qs = {}
        self.track_node_to_lrn_worker = {}

        self.remaining_points_input_qs = -1
        self.ref_master = None
        self.EOF = False
        self.running = True
        self.internal_start_time = perf_counter()

    # General functionality
    def callback(self, msg):
        """Callback function for the dataReader"""
        if msg == 'EOF':
            self.logger.info("End of file")
            self.EOF = True

    async def check_queue_size(self, worker_id):
        """Checks the size of the worker queue of worker with id: worker_id"""
        qsize = await self.track_lrn_worker_qs[worker_id].actor.qsize.remote()
        summary = (worker_id, self.get_internal_time(), qsize)
        return summary

    def get_internal_time(self):
        """Return time relative to the internal_start_time of the Master instance"""
        return perf_counter() - self.internal_start_time

    def get_learning_worker_id(self):
        """Returns the worker_id of the learning worker with the least load"""
        #print("Worker loads: ", self.lrn_worker_loads)
        return np.argmin(self.lrn_worker_loads)

    def get_splitting_worker_id(self):
        """Returns the splt_worker_id of the splitter worker with the least load"""
        return np.argmin(self.splt_worker_loads)

    def set_ref_to_self(self, ref_master):
        """Sets a reference to the ray actor instance of the Master instance"""
        self.ref_master = ref_master

    # Worker related
    #def check_available_resources(self):
    #    maxLearningWorkers = 10  # This should be found from checking resources on cluster
    #    maxSplitters = 1  # This should be found from checking resources on cluster
    #    return maxLearningWorkers, maxSplitters

    async def employ_learning_workers(self):
        """Instansiates the workers and their input queues"""
        for lrn_worker_id in range(0, self.num_lw):
            self.logger.info(f'Employ lw {lrn_worker_id}')
            self.track_lrn_worker_qs[lrn_worker_id] = Queue()
            self.lrn_workers[lrn_worker_id] = LearningWorker.remote(lrn_worker_id,
                                                                self.track_lrn_worker_qs[lrn_worker_id],
                                                                self.ref_master)
            await self.lrn_workers[lrn_worker_id].set_ref_self.remote(self.lrn_workers[lrn_worker_id])

    def employ_splitting_workers(self):
        """Instansiates the splitters and input queue and outlier queues of the splitters"""
        for splt_worker_id in range(0, self.num_sw):
            self.logger.info(f'Employ sw {splt_worker_id}')
            self.track_outlier_qs[splt_worker_id] = Queue()
            self.track_splt_worker_qs[splt_worker_id] = Queue()

            self.splt_workers[splt_worker_id] = SplittingWorker.remote(splt_worker_id,
                                                                self.track_splt_worker_qs[splt_worker_id],
                                                                self.track_outlier_qs[splt_worker_id],
                                                                self.ref_master)

    async def add_children(self, parent_node_id, parent_name, desired_children_centers, lvl):
        """Each LearningNode instance makes a callback to add_children when their learning is complete, with a
        list of desired children centers. add_children assigns these children centers to the appropriate workers. The
        workers then instansiate new LearningNodes on each of the centers and start the learning process

        Parameters
        ----------
        :params parent_node_id: Identifier of the parent node that has finished learning
        :params parent_name: Name of the parent node that has finished learning (Only for readability wrt debugging)
        :params desired_children_centers: The children centers that the parent node is requesting
        :params lvl: Level of the parent node

        Returns
        -------
        splt_worker_id: The id of the splitter that the new centers have been connected to
        """
        lrn_worker_id = self.get_learning_worker_id()
        splt_worker_id = self.get_splitting_worker_id()

        self.logger.info(f'Node {parent_name} requesting {len(desired_children_centers)} children')
        self.logger.info(f'Assign children of {parent_name} to lw {lrn_worker_id} and splitter {splt_worker_id}')

        num_children = len(desired_children_centers)
        self.lrn_worker_loads[lrn_worker_id] += num_children
        self.splt_worker_loads[splt_worker_id] += num_children

        node_ids = np.arange(self.num_nodes, self.num_nodes + num_children)  # Assign an id to each new node
        self.num_nodes += num_children  # Update node counter

        node_names = [parent_name + (i,) for i in range(num_children)]  # The node_names are just for readability in debuging purposes
        radius = self.init_radius / self.reduce_factor ** (lvl + 1)

        lrn_worker_qs = dict(zip(node_ids, itertools.repeat(self.track_lrn_worker_qs[lrn_worker_id]))) # Connect new nodes to worker queues
        temp_dict = dict(zip(node_ids, itertools.repeat(lrn_worker_id)))  # Connect new nodes to worker id
        self.track_node_to_lrn_worker.update(temp_dict)

        await self.lrn_workers[lrn_worker_id].add_nodes.remote(node_names, node_ids, desired_children_centers, radius, lvl + 1)
        await self.splt_workers[splt_worker_id].update_centers.remote(parent_node_id, node_names, node_ids,
                                                         desired_children_centers, radius*np.ones(num_children),
                                                         lrn_worker_qs)
        if not parent_node_id == None:
            self.lrn_worker_loads[self.track_node_to_lrn_worker[parent_node_id]] -= 1

        await self.results_logger.logg_nodes(np.ones(num_children) * (lvl+1), desired_children_centers, np.ones(num_children) * radius)
        return splt_worker_id

    # Master functions
    async def monitor(self):
        """Check number of points in worker queue every 1/q_ping_time seconds"""
        q_ping_freq = 1  # Number of times per second to check the queue size
        while self.running:
            await asyncio.sleep(1/q_ping_freq)
            qsizes = await asyncio.gather(*[self.check_queue_size(key) for key in list(self.track_lrn_worker_qs)])
            self.remaining_points_input_qs = np.sum([qs[2] for qs in qsizes])

    async def main_loop(self):
        """Runs until the data_reader raises EOF and there are no remaining points in the worker and splitter queues."""
        while self.running:
            await asyncio.sleep(1)
            if self.EOF and self.remaining_points_input_qs == 0:
                self.logger.info('Terminating master.main')
                self.results_logger.logg_summary(perf_counter()-self.start_time)

                self.running = False
                break

    async def main(self, tree):
        """Main function to be called to build the epsilon cover. Gather tasks to run concurrently in the event loop"""
        self.logger.info('Calling master.main')
        self.start_time = perf_counter()
        self.set_ref_to_self(tree)
        self.employ_splitting_workers()
        await self.employ_learning_workers()

        # Add first node
        splt_worker_id = await self.add_children(None, tuple(), [self.init_center], self.init_lvl-1)

        # Start data reader, queue monitor and main loop
        self.data_reader.stream_data.remote(self.track_splt_worker_qs[splt_worker_id], self.ref_master.callback)
        await asyncio.gather(self.monitor(), self.main_loop())


