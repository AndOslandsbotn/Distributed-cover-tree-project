import ray
from ray.util.queue import Queue
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

from utilities.util import calc_dist_matrix
from distributed_covertree.workers import LearningWorker, SplittingWorker

import asyncio
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

from definitions import CONFIG_PATH
from config.yaml_functions import yaml_loader
CONFIG = yaml_loader(CONFIG_PATH)

####################
### Test scripts ###
####################
@ray.remote
class DummyCoverTree():
    """Plots the centers"""
    def __init__(self, radius, plotting=False):
        self.radius = radius
        self.children_centers = None
        self.plotting = plotting

    def add_children(self, node_id, node_name, children_centers, lvl):
        print("DummyCoverTree callback to add_children")
        print(f"Node {node_name} is requesting children from master")
        if self.plotting == True:
            self.plot_node_distribution(f'{node_id}', children_centers, self.radius/2)
        self.children_centers = children_centers

    def get_children_centers(self):
        return self.children_centers

    def plot_node_distribution(self, name, children_centers, radius):
        print("Plotting children centers")
        plt.figure()
        fig, ax = plt.subplots()
        plt.title(f'Children of {name}')
        ax.add_patch(plt.Rectangle(tuple([-1, -1]), 2, 2, fill=False))
        for center in children_centers:
            ax.add_patch(plt.Circle(tuple([center[0], center[1]]),
                                    radius=radius, color='r', fill=False))
            plt.scatter(center[0], center[1], s=20, c='k', marker='o', zorder=2)
        plt.show()

async def test_worker(dim=2, num_batches=100, batch_size=1000, radius=1.2):
    def create_test_data_for_cover_test(dim, num_batches, batch_size):
        data = []
        for i in range(num_batches):
            data.append(uniform.rvs(loc=-1, scale=2, size=(batch_size, dim)))  # Uniform dist on [loc, loc+scale]
        return data

    data = create_test_data_for_cover_test(dim, num_batches, batch_size)

    dummyCoverTree = DummyCoverTree.remote(radius, plotting=True)

    # Employ workerf
    worker_id = 1
    worker_queue = Queue()
    worker = LearningWorker.remote(worker_id, worker_queue, dummyCoverTree)
    await worker.set_ref_self.remote(worker)

    # Node specs
    node_name = (0,)
    node_id = 0
    lvl = 0
    center = np.array([0, 0])  # root center
    radius = radius

    # Label data with node destination
    for d in data:
        worker_queue.put_nowait((node_id, d))

    # Assign node to worker
    worker.add_nodes.remote([node_name], [node_id], [center], radius, lvl)

    print('sleeping')
    await asyncio.sleep(10)
    print('Test cover properties finished successfully, see plots to see cover properties')

async def test_splitter(dim=2, num_batches=100, batch_size=1000):
    def create_test_data_for_cover_test(dim, num_batches, batch_size):
        data = []
        for i in range(num_batches):
            data.append(uniform.rvs(loc=-1, scale=2, size=(batch_size, dim)))  # Uniform dist on [loc, loc+scale]
        return data

    def make_reference(data, centers, radiuss):
        # Brute force sort the data points into their correct balls
        reference_data = np.array(data)
        reference_data = reference_data.reshape(num_batches * batch_size, -1)
        ref_dist_mat = calc_dist_matrix(reference_data, centers)

        items_for_center0 = []
        items_for_center1 = []
        items_for_center2 = []
        dictionary = {}
        dictionary[0] = items_for_center0
        dictionary[1] = items_for_center1
        dictionary[2] = items_for_center2
        outliers = []
        for i, distances in enumerate(ref_dist_mat):
            idx = np.argsort(distances)
            if distances[idx[0]] <= radiuss[idx[0]]:
                dictionary[idx[0]].append(reference_data[i])
            elif distances[idx[1]] <= radiuss[idx[1]]:
                dictionary[idx[1]].append(reference_data[i])
            elif distances[idx[2]] <= radiuss[idx[2]]:
                dictionary[idx[2]].append(reference_data[i])
            else:
                outliers.append(reference_data[i])

        items_for_center0 = np.concatenate((centers[0].reshape(1, -1), np.array(items_for_center0)))
        items_for_center1 = np.concatenate((centers[1].reshape(1, -1), np.array(items_for_center1)))
        items_for_center2 = np.concatenate((centers[2].reshape(1, -1), np.array(items_for_center2)))
        outliers = np.array(outliers)
        return items_for_center0, items_for_center1, items_for_center2, outliers

    def plot_data_assignments(items_for_center0, items_for_center1, items_for_center2, outliers, centers, radiuss, title):
        plt.figure()
        fig, ax = plt.subplots()
        plt.title(f'{title}')
        ax.add_patch(plt.Rectangle(tuple([-1, -1]), 2, 2, fill=False))

        plt.scatter(outliers[:, 0], outliers[:, 1], color='k', s=5)
        plt.scatter(items_for_center0[:, 0], items_for_center0[:, 1], color='b', s=5)
        plt.scatter(items_for_center1[:, 0], items_for_center1[:, 1], color='c', s=5)
        plt.scatter(items_for_center2[:, 0], items_for_center2[:, 1], color='g', s=5)

        for i, center in enumerate(centers):
            ax.add_patch(plt.Circle(tuple([center[0], center[1]]), radius=radiuss[i], color='r', fill=False))
        plt.show()

    centers = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    centers_ids = np.array([0, 1, 2])
    node_names = np.array([0, 1, 2])
    radiuss = np.array([1, 0.5, 0.5])
    data = create_test_data_for_cover_test(dim, num_batches, batch_size)
    ref_assigned_center0, ref_assigned_center1, ref_assigned_center2, ref_assigned_outliers = make_reference(data, centers, radiuss)
    plot_data_assignments(ref_assigned_center0, ref_assigned_center1, ref_assigned_center2, ref_assigned_outliers,
                          centers, radiuss, title='Reference assignments of data')


    dummyCoverTree = DummyCoverTree.remote(radius=1)

    # Employ splitter
    splitter_id = 1
    splitter_input_queue = Queue()
    outlier_queue = Queue()
    splitter = SplittingWorker.remote(splitter_id, splitter_input_queue, outlier_queue, dummyCoverTree)

    # Label data with node destination
    for d in data:
        splitter_input_queue.put_nowait(d)

    worker_queue_dict = {}
    worker_queue_dict[0] = Queue()
    worker_queue_dict[1] = Queue()
    worker_queue_dict[2] = Queue()

    splitter.update_centers.remote(None, node_names, centers_ids, centers, radiuss, worker_queue_dict)

    print('sleeping')
    await asyncio.sleep(50)

    assigned_queue0 = worker_queue_dict[0].get_nowait_batch(worker_queue_dict[0].qsize())
    assigned_queue1 = worker_queue_dict[1].get_nowait_batch(worker_queue_dict[1].qsize())
    assigned_queue2 = worker_queue_dict[2].get_nowait_batch(worker_queue_dict[2].qsize())
    assigned_queue_outliers = outlier_queue.get_nowait_batch(outlier_queue.qsize())

    # Get items that remain in the buffer list
    items_waiting_for_assignment = ray.get(splitter.get_buffered_items.remote())

    if len(assigned_queue0)>0:
        node_id, assigned_center0 = assigned_queue0[0]
        for assigned in assigned_queue0[1::]:
            node_id, items = assigned
            assigned_center0 = np.concatenate((assigned_center0, items))
        assigned_center0 = np.concatenate((assigned_center0, items_waiting_for_assignment[0]))
    else:
        assigned_center0 = items_waiting_for_assignment[0]

    if len(assigned_queue1) > 0:
        node_id, assigned_center1 = assigned_queue1[0]
        for assigned in assigned_queue1[1::]:
            node_id, items = assigned
            assigned_center1 = np.concatenate((assigned_center1, items))
        assigned_center1 = np.concatenate((assigned_center1, items_waiting_for_assignment[1]))
    else:
        assigned_center1 = items_waiting_for_assignment[1]

    if len(assigned_queue2) > 0:
        node_id, assigned_center2 = assigned_queue2[0]
        for assigned in assigned_queue2[1::]:
            node_id, items = assigned
            assigned_center2 = np.concatenate((assigned_center2, items))
        assigned_center2 = np.concatenate((assigned_center2, items_waiting_for_assignment[2]))
    else:
        assigned_center2 = items_waiting_for_assignment[2]

    assigned_outliers = assigned_queue_outliers[0]
    for assigned in assigned_queue_outliers[1::]:
        assigned_outliers = np.concatenate((assigned_outliers, assigned))

    plot_data_assignments(assigned_center0, assigned_center1, assigned_center2, assigned_outliers,
                          centers, radiuss, title='Splitter assignments of data')

    assert np.array_equal(assigned_center0, ref_assigned_center0)
    assert np.array_equal(assigned_center1, ref_assigned_center1)
    assert np.array_equal(assigned_center2, ref_assigned_center2)
    assert np.array_equal(assigned_outliers, ref_assigned_outliers)
    print("Tested splitter successfully")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_worker(dim=2, num_batches=10, batch_size=CONFIG['performance']['splitterq_batch_size'], radius=1.42))
    loop.run_until_complete(test_splitter(dim=2, num_batches=10, batch_size=CONFIG['performance']['splitterq_batch_size']))


