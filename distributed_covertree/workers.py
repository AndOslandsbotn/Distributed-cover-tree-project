import ray
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

import asyncio
import numpy as np

from distributed_covertree.learning_node import LearningNode
from utilities.util import calc_dist_matrix
from utilities.logg_classes import WorkerPerformanceLogger, init_logger, setup_logger

from definitions import CONFIG_PATH, LOG_CFG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)

class BaseWorker():
    """Base class for the LearningWorker and SplittingWorker

    public methods:
    * set_ref_self: Sets a reference to the ray actor instance of the worker class
    * stop_processing: Stops processing
    * process: Main process, gathers all threads that will run concurrenctly in the event loop

    private methods:
    * process_loop: To be overridden by derived classes
    """
    def __init__(self, worker_id, worker_type, input_queue, outlier_queue, ref_covertree):

        self.worker_type = worker_type
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.outlier_queue = outlier_queue
        self.ref_covertree = ref_covertree

        self.passthroughbytes_counter = [0]
        self.performanceLogger = WorkerPerformanceLogger(self.passthroughbytes_counter,
                                                         self.input_queue,
                                                         self.worker_id,
                                                         self.worker_type)

        self.logger = setup_logger(logger_name=f'{worker_type}{worker_id}', log_cfg_path=LOG_CFG_PATH)

        self.ref_self = None
        self.running = True

    def set_ref_self(self, ref_self):
        """Sets a reference to the ray actor instance of the worker class"""
        self.ref_self = ref_self

    def stop_processing(self):
        """Stops processing"""
        self.logger.info('Terminate')
        self.running = False

    async def process_loop(self):
        """To be overridden in derived class"""
        return

    async def process(self):
        """Main process, gathers all threads that will run concurrenctly in the event loop"""
        await asyncio.gather(self.process_loop(),
                             self.performanceLogger.logging(),
                             self.performanceLogger.check_qsize())


@ray.remote
class LearningWorker(BaseWorker):
    """ A ray actor which is assigned nodes in the form of (centers, radius)

    Objects:
    - input_queue: A queue containing elements (node_id, items) assigned to worker
    - track_nodes: All LearningNodes that are assigned to a worker instance and are currently active (still learning)

    Responsibilities:
    - Instansiate a LearningNode on the assigned (centers, radius) and learning is started
    - Runs all assigned LearningNode instances asynchronously

    Public methods:
    * get_num_nodes: Returns the number of currently active nodes in the Worker instance
    * process_loop: Reads (node_id, items) from input_queue and assignes items to node with node_id.
    * add_nodes: Instansiates LearningNodes on the assigned (center, radius)

    Private methods:
    * remove_node
    """
    def __init__(self, worker_id, input_queue, ref_covertree):
        super().__init__(worker_id, 'LearningWorker', input_queue, None, ref_covertree)
        self.logger.info(f'Init lw{worker_id}')

        self.track_nodes = {}
        self.node_id_to_node_name = {}  # This is only for readability/debugging purposes

        self.num_nodes = 0
        self.running = True

    def get_num_nodes(self):
        "Returns the currently active number of nodes (nodes that are learning)"
        return self.num_nodes

    def remove_node(self, node_id):
        self.logger.info(f'Delete {self.node_id_to_node_name[node_id]} from dictionary')
        del self.track_nodes[node_id]

    async def process_loop(self):
        """Reads (node_id, items) from input_queue and assignes items to node with node_id."""
        self.logger.info(f'lw{self.worker_id} is processing')
        while self.running:
            node_id, items = await self.input_queue.get_async(block=True)
            self.passthroughbytes_counter[0] += items.nbytes

            if node_id in self.track_nodes:
                self.track_nodes[node_id].learn(items)
            else:
                # When items are assigned to a learning node that has completed learning,
                # we should perhaps implement a way to divert these items back to the splitter queue?
                self.logger.debug(f'Node id {node_id} is not in track_nodes, discard {len(items)} items')

    async def add_nodes(self, node_names, node_ids, centers, radius, lvl):
        """ Instansiates LearningNodes on the assigned (center, radius). Assumes that all centers in centers
        have the same radius/lvl.

        Parameters
        ----------
        :param node_names: Name of nodes
        :param node_ids: Identifiers of nodes
        :param radius: radius of nodes
        :param lvl: lvls of nodes
        """
        self.logger.info(f'Worker {self.worker_id} added nodes {node_names}')
        for i, node_id in enumerate(node_ids):
            self.num_nodes += 1
            self.node_id_to_node_name[node_id] = node_names[i]  # This is only for readability/debugging purposes
            self.track_nodes[node_id] = LearningNode(node_names[i], node_id, centers[i], radius, lvl,
                                                     self.ref_covertree, (self.worker_id, self.ref_self))

            if self.num_nodes == 1:
                asyncio.gather(self.process())


@ray.remote
class SplittingWorker(BaseWorker):
    """ A ray actor which is keeps track of a numpy array of centers with associated radiuses.

    Objects:
    - input_queue: A queue containing items to be assigned to the nearest covering node in track_nodes
    - track_lrn_worker_qs: Keeps track the worker queue to which each node in track_nodes is assigned

    Responsibilities:
    - Read items from input_queue
    - Find closes covering nodes for each item
    - Add items to the appropriate worker queue
    - Use buffer to ensure that items are assigned to nodes in batches of size self.batch_size

    Public methods:
    * process_loop: Main loop of the splitter, reads items from input_queue
    * get_buffered_items: This function is only for unit test of the splitter
    * update_centers: Update centers in splitter

    Private methods:
    * find_nearest_covering:  Find the nearest center for each item, for which ||center-items|| < radius.
    * assign_to_queues: Assigns item to the correct worker queues
    * buffer: Ensures that items are assigned to nodes in batches of size self.batch_size
    * remove_centers: remove node from splitter
    """

    def __init__(self, worker_id, input_queue, outlier_queue, ref_covertree):
        super().__init__(worker_id, 'SplittingWorker', input_queue, outlier_queue, ref_covertree)
        self.logger.info(f'Init sw{worker_id}')

        self.lock_on_center_manip = asyncio.Lock()
        self.center_ids = None  # center_id = 0 corresponds to everything not covered by any centers
        self.centers = None
        self.radiuss = None

        self.num_centers = 0
        self.track_lrn_worker_qs = {}

        self.batch_size = CONFIG['performance']['learnerq_batch_size']
        self.buffered_items = {}

        self.running = True

        self.node_name_to_center_id = {}  # This is just for debugging, so that we can more easily identify nodes.

    def get_buffered_items(self):
        """This function is only for unit test of the splitter"""
        return self.buffered_items

    def find_nearest_covering(self, items):
        """
        Find the nearest center for each item, for which ||center-items|| < radius. In other words,
        finds the nearest center among the centers that are covering item.

        Parameters
        ----------
        :param items: n x d numpy array of items for which we find nearest covering center
        Returns
        -------
        idx_nn_center: n-dim ndarray which for each of the n items contains idx to nearest covering neighbour center
        not_covered: indices of items that are not covered by any centers
        """

        dist_mat = calc_dist_matrix(items, self.centers)

        # Find indices of nearest neighbour center that is covering item
        idx = (dist_mat > self.radiuss)
        dist_mat[idx] = np.infty  # We set the distance to nodes that are not covering item, to infinity
        idx_nn_center = np.argmin(dist_mat, axis=1)  # We find indices to the closest centers among centers that cover item

        # Special treatment for items that are not covered by any centers
        not_covered = np.sum(idx, axis=1) == len(self.centers)  # We get indices of items that are not covered by any center
        idx_nn_center[not_covered] = -1  # For items that are not covered by any centers, we set the index to -1
        return idx_nn_center, not_covered

    async def assign_to_queues(self, items, idx_nn_center, not_covered):
        """
        Assigns item to the correct worker queues based on the item to center pairing in idx_nn_center
        Parameters
        ----------
        :param Items: n x d numpy array of items for which we will assign to the appropriate worker queue
        :param idx_nn_center: Elements in idx_nn_center list are center_ids. The center_id in position i
                            in the idx_nn_center list is the id of the worker we want to assign item i to.
        :param not_covered: Indices of items that are not covered by any centers
        """

        for center_id in self.center_ids:
            x = items[(idx_nn_center == center_id)]
            x = self.buffer(center_id, x)
            if len(x) > 0:
                queue = self.track_lrn_worker_qs[center_id]
                await queue.put_async((center_id, x), block=True)

        # Outliers
        outliers = items[not_covered]
        if len(outliers) > 0:
            await self.outlier_queue.put_async(outliers, block=True)

    def buffer(self, center_id, items):
        """Ensures that items are assigned to nodes in batches of size self.batch_size

        Parameters
        ----------
        :param items: items assigned to node with center_id
        :param center_id: The identifier of the node/center that items have been assigned to

        Return
        ----------
        if we have batch_size items assigned to node center_id
        return x: numpy array of length self.batch_size
        otherwise:
        return np.array([]): empty numy array
        """

        self.buffered_items[center_id] = np.concatenate((self.buffered_items[center_id], items), axis=0)
        if len(self.buffered_items[center_id]) > self.batch_size:
            x = self.buffered_items[center_id][0:self.batch_size]
            self.buffered_items[center_id] = self.buffered_items[center_id][self.batch_size::]
            return x
        else:
            return np.array([])

    # New centers
    def remove_centers(self, center_id):
        """Remove center corresponding to center_id"""
        self.logger.info(f"Remove center {center_id}")

        center_index = np.argwhere(self.center_ids == center_id)
        self.centers = np.delete(self.centers, center_index, axis=0)
        self.center_ids = np.delete(self.center_ids, center_index, axis=0)
        self.radiuss = np.delete(self.radiuss, center_index, axis=0)
        del self.track_lrn_worker_qs[center_id]
        del self.buffered_items[center_id]

    async def update_centers(self, parent_center_id, node_names, center_ids, centers, radiuss, lrn_worker_qs):
        """Update centers in splitter

        Parameters
        ----------

        :param parent_center_id: Identifier of parent
        :param node_names: Name of nodes
        :param center_ids: Identifier of new centers
        :param centers: Centers
        :param radiuss: Radius of new centers
        :param lrn_worker_qs: Dictionary of learning worker queues of each of the new centers
        """

        self.logger.info(f'sw add centers {node_names}, removing parent center {parent_center_id}')
        await self.lock_on_center_manip.acquire()  # To ensure we dont assign items while updating centers
        if parent_center_id is not None:
            self.remove_centers(parent_center_id)

        centers = np.array(centers)
        num_new_centers = len(centers)
        if self.num_centers == 0:
            self.center_ids = np.array(center_ids)
            self.centers = centers
            self.radiuss = np.array(radiuss)
        else:
            self.center_ids = np.concatenate((self.center_ids, np.array(center_ids)), axis=0)
            self.centers = np.concatenate((self.centers, centers), axis=0)
            self.radiuss = np.concatenate((self.radiuss, np.array(radiuss)), axis=0)

        for center_id in center_ids:
            self.track_lrn_worker_qs[center_id] = lrn_worker_qs[center_id]

        centers = centers.reshape(num_new_centers, 1, -1)
        self.buffered_items.update(dict(zip(center_ids, centers)))

        for node_name in node_names:
            self.node_name_to_center_id[node_name] = center_ids  # Just for debuging purposes
        self.lock_on_center_manip.release()  # Release lock

        if self.num_centers == 0:
            asyncio.gather(self.process())
        self.num_centers += num_new_centers

    # Processing items
    async def process_loop(self):
        """Reads items from input_queue, and organize them such that each item in items is assigned to the
        closes covering (center,radius) in self.track_nodes. The centers are put in the worker queue in
        track_worker_qs, for the worker to which the corresponding LearningNode with (center,radius) is assigned.
        """
        self.logger.info(f"Splitter {self.worker_id} is processing")
        while self.running:
            print("qsize before splitter reading", self.input_queue.qsize())
            items = await self.input_queue.get_async(block=True)
            print("length items from splitter reading: ", len(items))
            print("qsize after splitter reading", self.input_queue.qsize())
            self.passthroughbytes_counter[0] += items.nbytes

            await self.lock_on_center_manip.acquire()  # To ensure no add/delete of centers while assigning
            idx_nn_center, not_covered = self.find_nearest_covering(items)
            await self.assign_to_queues(items, idx_nn_center, not_covered)
            self.lock_on_center_manip.release()  # Release lock

