#!/bin/bash
#SBATCH --job-name=rayio
#SBATCH --account=csd730
#SBATCH --mem=249325M
#SBATCH --output=rayio.%j.%N.out
#SBATCH --partition=compute
#SBATCH --time=00:30:00
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=2
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
## YF: We probably want to have more than one task per node, and less than 128 cpu's per task.
## YF: How do the parameters specified here interact with the ray configuration files?
## YF: How do we control the number of virtual-memory devices vs the number of nodes and the number of cores?

################ Start Conda environment
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ray

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://urldefense.com/v3/__https://github.com/ray-project/ray/issues/826*issuecomment-522116599__;Iw!!Mih3wA!GkJRww5IXIkWTEezlWK-M4STxDk2_LpqydaH9nV-vdfZuLXoAYen1VfPFEGdfEs6aD93WtfZOZo-ZjdULQ892A$ 
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

export TUNE_MAX_PENDING_TRIALS_PG=64

node_1=${nodes_array[0]} 
ipadr=`ssh $node_1 ip -f inet addr show ib0 | grep inet | awk '{print $2}' | awk -F '/' '{print$1}'` # making redis-address
port=6379
ip_head=$ipadr:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ipadr $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################

#### call your code below
#python tests/test_xsede_expanse_ray.py
python DistributedCoverTree/master.py
exit
