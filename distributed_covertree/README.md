# Organization

## Master.py file
#### Master 
The master class for constructing the epsilon cover tree. 
A ray actor which acts as the master. Sets up the program and 
is the owner of the input queues to the CoverTreeWorkers

   Objects:
   - data_reader: Reads data from file in batches
   - lrn_workers: A dictionary of instances of the Worker class
   - splt_workers: A dictonary of instances of the Splitter class

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
        * get_learning_worker_id: Returns the lrn_worker_id of the worker with the least load
        * get_splitting_worker_id: Returns the splt_worker_id of the splitter with the least load
        * set_ref_to_self: Sets a reference to the ray actor instance of the Master instance
        * check_available_resources
        * employ_learning_workers: Instansiates the learning workers and their input queues
        * employ_splitting_workers: Instansiates the splitting workers, input queue and outlier queues of the splitters

    Callbacks:
        * callback: Callback from the data_reader to inform that the end of the data file has been reached.
        * add_children: Each LearningNode instance makes a callback to add_children when their learning is complete, with a
        list of desired children. add_children assigns these children centers to the appropriate workers to start learning.


## Learning_node.py file
#### LearningNode
A node in the cover tree that reads the elements from input queue and 
uses nearest neighbour, kmeans and kmeans++ to find a radius/2-cover.

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

## workers.py file

#### BaseWorker
Base class for the LearningWorker and SplittingWorker

    public methods:
    * set_ref_self: Sets a reference to the ray actor instance of the worker class
    * stop_processing: Stops processing
    * process: Main process, gathers all threads that will run concurrenctly in the event loop
    
    private methods: 
    * process_loop: To be overridden by derived classes

#### LearningWorker
A ray actor which is assigned Nodes to process. Each Worker has an input queue. The elements
in the input queue contains a vector of elements and a tag
indicating which of the nodes assigned to the worker.

   Objects:
   - input_queue: A queue containing elements (node_id, items) assigned to worker
   - track_nodes: All LearningNodes that are assigned to a worker instance and are currently active (still learning)

   Responsibilities:
   - Instansiate a LearningNode on the assigned (centers, radius) and learning is started
   - Runs all assigned LearningNode instances asynchronously

    Public methods:
        * get_num_nodes: Returns the number of currently active nodes in the Worker instance
        * process_loop: Reads (node_id, items) from worker_input_queue and assignes items to node with node_id.
        * add_nodes: Instansiates LearningNodes on the assigned (center, radius)

    Private methods:
        * remove_node
        
#### SplittingWorker
A ray actor which is keeps track of a numpy array of centers with associated radiuses.

   Objects:
   - input_queue: A queue containing items to be assigned to the nearest covering node in track_nodes
   - track_lrn_worker_qs: Keeps track the worker queue to which each node in track_nodes is assigned

   Responsibilities:
   - Read items from splitter_input_queue
   - Find closes covering nodes for each item
   - Add items to the appropriate worker queue
   - Use buffer to ensure that items are assigned to nodes in batches of size self.batch_size

    Public methods:
        * process_loop: Main loop of the splitter, reads items from splitter_input_queue
        * get_buffered_items: This function is only for unit test of the splitter
        * update_centers: Update centers in splitter

    Private methods:
        * find_nearest_covering:  Find the nearest center for each item, for which ||center-items|| < radius.
        * assign_to_queues: Assigns item to the correct worker queues
        * buffer: Ensures that items are assigned to nodes in batches of size self.batch_size
        * remove_centers: remove node from splitter
