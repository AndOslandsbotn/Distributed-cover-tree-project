import ray
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

import asyncio
import numpy as np
from time import perf_counter
import logging

from utilities.util import calc_dist_matrix
from utilities.kmeans import kmeans
from utilities.logg_classes import NodePerformanceLogger, init_logger, setup_logger

from definitions import CONFIG_PATH, LOG_CFG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)

class LearningNode():
    """A node in the cover tree that reads the elements from input queue and uses nearest neighbour, kmeans and kmeans++
     to find a radius/2-cover.

     The LearningNode has two optional functionalities, that can be set in the configuration file CONFIG:
     1. ) kmeanspp method
     2. ) refine method

    Public methods:
        * process: Determines based on state of node, whether to call learn() or super.process()
        * learn: Process data from the input_queue:
            - Reads item and compute it's distances from current centers
            - Decides whether to add point as a center
            - updates cover_fraction
            - refines center positions using kmeans
            - makes a callback to cover tree to add desired children
        * set_state: Sets the state of the node (Only to be used by the cover-tree class)
        * get_children_centers: returns centers of children_centers
        * get_center: returns center of node

    Private methods:
        * update_cover_fraction: Update the cover fraction which estimates how much of the node is covered
            by the potential_children.
        * find_uncovered_items: Find items that are not covered by the current centers
        * kmeanspp: Use modified k-means++ to decide whether to add a new potential child or not
            based on the distance to the existing potential children
        * refine: Use the k-means method to refine the center positions of the potential_children
    """
    def __init__(self, node_name, node_id, center, radius, lvl, ref_master, ref_worker, logg=False):
        """
         Parameters
        ----------
        :param node_name: For readability only. The path to this node in the tree (mirroring value in tree)
        :param node_id: An integer that uniquely identifies the node
        :param center: The center corresponding to this node
        :param radius: points whose distance from the center is larger than radius are discarded.
        :param input_queue: A queue from where the node takes it's input.
        :param outlier_queue: A queue in which to put input points that are more than radius away from the closest center.
        """
        self.ref_master = ref_master
        self.worker_id, self.ref_worker = ref_worker
        self.timeLogger = NodePerformanceLogger(os.path.join('LearningNode'))
        self.logg = logg

        self.logger = setup_logger(logger_name=f'Node{node_name}', log_cfg_path=LOG_CFG_PATH)

        # Node specifications
        self.center = center
        self.radius = radius
        self.lvl = lvl
        self.node_name = node_name
        self.node_id = node_id
        self.child_centers = None

        # Set configuration parameters
        self.min_items_len = CONFIG['learningnode']['min_items_len']
        self.threshold = CONFIG['learningnode']['threshold']
        self.alpha = CONFIG['learningnode']['alpha']
        self.kmeans_refinement = CONFIG['learningnode']['kmeans_refinement']
        self.kmeans_sample_size = CONFIG['learningnode']['kmeans_sample_size']
        self.kmeanspp_conditioning = CONFIG['learningnode']['kmeanspp_conditioning']
        self.augmented_fcs = CONFIG['learningnode']['augmented_find_centers_search']
        self.reduce_factor = CONFIG['general']['radius_reduce_factor']

        # Node cover
        self.cover_fraction = 0
        self.potential_children = []  # Is it unfortunate that this is a list?

        self.running = True
        self.state = 'learning'

    def terminate(self):
        self.running = False

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def complete_learning(self):
        """Makes a callback to Master to initialize learning the children and move current node to data splitter.
        Makes another callback to worker, to inform that learning is completed."""
        if self.ref_master != None:
            self.logger.info(f'Callback to master')
            asyncio.gather(self.ref_master.add_children.remote(self.node_id, self.node_name,
                                                                  self.potential_children, self.lvl))
        if self.ref_worker != None:
            self.logger.info(f'Callback to worker')
            asyncio.gather(self.ref_worker.remove_node.remote(self.node_id))

    def learn(self, items):
        """The update method learns from the items it recieves:
            * Reads item and compute it's distances from current centers
            * Decides whether to add point as a center
            * updates cover_fraction
            * refines center positions using kmeans
            * makes a callback to cover tree to add desired children
        """
        self.logger.info(f"Recieved {len(items)} items to learn from")
        self.logger.debug(f"Current cover fraction {self.cover_fraction}")
        if self.state == 'learning':
            #print(f"Node{self.node_name} at Worker{self.worker_id} recieved items: ", len(items))
            self.find_centers(items)

        if self.state == 'learning' and self.cover_fraction > self.threshold:
            self.logger.info(f'Finished learning with cover fraction {self.cover_fraction}')

            if self.kmeans_refinement:
                self.logger.info(f'Make refinement')
                start = perf_counter()
                self.refine(items)
                if self.logg == True:
                    self.timeLogger.logg([perf_counter()-start, self.kmeans_sample_size], filename='refine')

            self.child_centers = self.potential_children
            self.set_state('learning_completed')
            self.complete_learning()

        elif self.state == 'learning_completed':
            #self.logger.debug(f'Node{self.node_name} New items assigned to learn, but learning is completed')
            pass

    def find_centers(self, items):
        if len(self.potential_children) == 0:
            # We just select an item, anything goes
            #self.logger.debug(f'Node{self.node_name} select first center')
            self.potential_children.append(items[0])

        current_centers = np.array(self.potential_children)

        start = perf_counter()
        items_uncovered, dist_mat_uncovered, idx_uncovered = self.find_uncovered_items(items, current_centers)
        if self.logg == True:
            self.timeLogger.logg([perf_counter()-start, len(items)], filename='find_uncovered_items')

        start = perf_counter()
        self.update_cover_fraction(len(items), len(items_uncovered))
        if self.logg == True:
            self.timeLogger.logg([perf_counter()-start, len(items)], filename='update_cover_fraction')

        if len(items_uncovered) > 0:
            start = perf_counter()
            new_center = self.select_center(items_uncovered, dist_mat_uncovered).reshape(1,-1)
            if self.logg == True:
                self.timeLogger.logg([perf_counter()-start, len(items)], filename='select_center')

            if self.augmented_fcs:
                start = perf_counter()
                self.augmented_find_centers(new_center, items_uncovered)
                if self.logg == True:
                    self.timeLogger.logg([perf_counter()-start, len(items)], filename='augmented_find_centers')

    def augmented_find_centers(self, new_center, items_uncovered):
        while True:
            start = perf_counter()
            items_uncovered, dist_mat_uncovered, idx_uncovered = self.find_uncovered_items(items_uncovered, new_center)
            if self.logg == True:
                self.timeLogger.logg([perf_counter() - start, len(items_uncovered)], filename='find_uncovered_items')

            if len(items_uncovered) < self.min_items_len:
                break
            start = perf_counter()
            new_center = self.select_center(items_uncovered, dist_mat_uncovered).reshape(1,-1)
            if self.logg == True:
                self.timeLogger.logg([perf_counter() - start, len(items_uncovered)], filename='select_center')

    def select_center(self, items, dist_mat):
        """Select center using kmeans++ if kmeanspp_conditioning is True. Otherwise, use the
        select the item furthest away from existing centers as the next center."""
        #print(f"Node{self.node_name} at Worker{self.worker_id} select new center")
        num_items = len(items)
        if self.kmeanspp_conditioning:
            idx_new_center = self.kmeanspp(np.arange(num_items), dist_mat)
        else:
            idx_new_center = np.argmax(np.sum(dist_mat, axis=1))

        new_center = items[idx_new_center]
        self.potential_children.append(new_center)
        return new_center

    def find_uncovered_items(self, items, centers):
        """Find the items that are not covered by neighbour_centers"""
        radius = self.radius/self.reduce_factor
        dist_mat = calc_dist_matrix(items, centers)
        num_neighbour_centers = len(centers)
        idx_uncovered = (np.sum((dist_mat > radius), axis=1) == num_neighbour_centers)
        print(f"{self.node_name} Number of centers {len(centers)}")
        print(f"{self.node_name} Num Uncovered items: {sum(idx_uncovered)}")
        print(f"{self.node_name} Items that are covered and therefore discarded: {len(items)-sum(idx_uncovered)}, with cf {self.cover_fraction}")
        return items[idx_uncovered, :], dist_mat[idx_uncovered, :], idx_uncovered

    def kmeanspp(self, indices, dist_mat):
        """Select a next center randomly among items with a probability weighted by square euclidean
         distance to existing centers"""
        square_distances = np.linalg.norm(dist_mat, axis=1)**2
        square_distances = square_distances/np.sum(square_distances)
        return np.random.choice(indices, p=square_distances)

    def update_cover_fraction(self, num_items, num_items_uncovered):
        """Update the cover fraction which estimates how much of the node is covered by the potential_children.
        The multiplication by alpha ensures that the ratio=num_items/num_items_uncovered
        is averaged over 1/alpha updates.
        """

        self.cover_fraction = (1-self.alpha)*self.cover_fraction + self.alpha*(1-num_items_uncovered/num_items)
        return

    def refine(self, items):
        """Use the k-means method to refine the center positions of the potential_children"""

        items = items[np.random.choice(np.arange(len(items)), size=self.kmeans_sample_size)]
        cost, new_centers = kmeans(items, self.potential_children, max_iter=10, stationary=[])
        self.potential_children = new_centers
        return

    def set_forward_queues(self, forward_queues):
        self.forward_queues = forward_queues

    def get_children_centers(self):
        return self.child_centers

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

    def get_lvl(self):
        return self.lvl


#### Test Code
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import uniform
from numpy.random import Generator, PCG64

seed = 42
numpy_randomGen = Generator(PCG64(seed))
uniform.random_state = numpy_randomGen


async def test_cover_properties_of_learning_node(dim=2, num_batches=3, batch_size=1000, radius=1.2):
    """Testing the cover properties of the learning node by plotting the node cover (see Figures folder).
     It also compares how the kmeans refinement and kmeans++ conditioning improves the cover properties."""
    print("Test cover properties of learning node")
    def create_test_data_for_cover_test(dim, num_batches, batch_size):
        data = []
        for i in range(num_batches):
            data.append(uniform.rvs(loc=-1, scale=2, size=(batch_size, dim)))  # Uniform dist on [loc, loc+scale]
        return data

    def plot_node_distribution(name, children_centers, radius):
        plt.figure()
        fig, ax = plt.subplots()
        plt.title(f'Children of {name}')
        ax.add_patch(plt.Rectangle(tuple([-1, -1]), 2, 2, fill=False))
        for center in children_centers:
            ax.add_patch(plt.Circle(tuple([center[0], center[1]]),
                                    radius=radius, color='r', fill=False))
            plt.scatter(center[0], center[1], s=20, c='k', marker='o', zorder=2)

        folder = 'Figures'
        Path(folder).mkdir(parents=True, exist_ok=True)
        kmeans_refinement = CONFIG['learningnode']['kmeans_refinement']
        kmeanspp_conditioning = CONFIG['learningnode']['kmeanspp_conditioning']
        filename = f"node_cover_kmeans{kmeans_refinement}_kmeanspp{kmeanspp_conditioning}"
        file = os.path.join(folder, filename)
        plt.savefig(file)
        plt.show()

    data = create_test_data_for_cover_test(dim, num_batches, batch_size)
    print("len data: ", len(data))

    center = np.array([0, 0])  # root center
    node_name = (0,)
    node_id = 0
    node = LearningNode(node_name, node_id, center, radius, None, None, logg=True)

    for d in data:
        node.learn(d)

    print('sleeping')
    await asyncio.sleep(10)

    centers = node.get_children_centers()
    plot_node_distribution(node_name, centers, radius/2)
    print('Test 2 finished, see plots to see cover properties')

if __name__ == '__main__':
    asyncio.run(test_cover_properties_of_learning_node(dim=2, num_batches=10, batch_size=10000, radius=1.2))
    print("Successfully tested learning node")
