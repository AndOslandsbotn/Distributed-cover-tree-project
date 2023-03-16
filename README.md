# Distributed-cover-tree-project
In this project we develop a distributed cover-tree algorithm using the Ray framework
and asyncio.

The cover-tree is
an algorithm that generates an epsilon cover of a point cloud of data. But was
originally intended as a datastructure for nearest neighbor search in metric spaces. 


### Install
To install the code, clone the repository and install the packages in the 
requirements.txt file using pip install -r requirements.txt from the root of your repository

### Run
To run a test of the code on a dummy dataset. Run first

    - main_generate_dummydata.py
    
This generates a dataset of desired size. Then run 

    - main.py 
    
This constructs a cover-tree in a distributed manner.

### Note
To run tests it might be necessary to add a Logg folder to the tests folder
### Reference
The standard cover-tree algorithm can for example be found here:

<a id="1">[1]</a> A. Beygelzimer, S. Kakade and J. Langford (2006)
Cover trees for nearest neighbor
Proc. 23th Int. Conf. Mach. Learn. p. 97--104
https://doi.org/10.1145/1143844.1143857