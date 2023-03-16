import ray
if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

import time
import pickle
from csv import writer, reader
import numpy as np
import sys
import asyncio

import codecs
import json
import os
from pathlib import Path

from definitions import GB, MB

# Pickle read write
def read_pickle_file(filepath, filename):
    obj = []
    with open(os.path.join(filepath, filename), 'rb') as f:
        try:
            while True:
                obj.append(pickle.load(f))
        except EOFError:
            return obj

async def save_pickle_async(file, info, operation='ab+'):
    with open(file, operation) as f:
        pickle.dump(info, f)
    f.close()

# CSV read write
def save_csv(file, info, operation='a'):
    with open(file, operation, newline='') as f:
        writer_obj = writer(f)
        writer_obj.writerow(info)
        f.close()

async def save_csv_async(file, info, operation='a'):
    with open(file, operation, newline='') as f:
        writer_obj = writer(f)
        writer_obj.writerow(info)
        f.close()

def load_csv(file, operation='r'):
    with open(file, operation, newline='') as f:
        reader_obj = reader(f)
        data = []
        for row in reader_obj:
            data.append(row)
        return data

async def load_csv_async(file, operation='r'):
    with open(file, operation, newline='') as f:
        reader_obj = reader(f)
        data = []
        for row in reader_obj:
            data.append(row)
        return data

def save_json(info, filepath, filename, operation='w'):
    Path(filepath).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(filepath, filename)
    json.dump(info, codecs.open(file_path, operation, encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)  ### this saves the array in .json format

def load_json( filepath, filename, operation='r'):
    file_path = os.path.join(filepath, filename)
    with open(file_path, operation) as f:
        return json.load(f)


@ray.remote
class ReadDataFromFile():
    """We should read from file using something like kafka ray functionality
    """
    def __init__(self, data):
        self.data = data

    def get_length_data(self):
        return len(self.data)

    def stream_data(self, queue, callback):
        counter = 0
        time.sleep(1)
        while True:
            try:
                p = self.data.pop(0)
                counter += 1
                if counter % 1000 == 0:
                    print(f"ReadFromFile, batch nr {counter}")
                queue.put(p)
            except IndexError as err:
                print(f"{err}")
                callback.remote('EOF')
                break

def load_data_on_disk(filepath):
    data_on_disk = np.load(filepath, mmap_mode='r')
    print(f'data on disk: {sys.getsizeof(data_on_disk):12} bytes ({sys.getsizeof(data_on_disk)/GB:3.3f} GB)')
    return data_on_disk

@ray.remote
def stream_data(filepath, batch_size=1000, loops=1):
    data_on_disk = load_data_on_disk(filepath)
    maxsize, dim = data_on_disk.shape

    for lopp in range(loops):
        i = 0
        while i < maxsize:
            ary = np.array(data_on_disk[i*batch_size:(i+1)*batch_size])
            print(f"array size {ary.nbytes/MB} MB")
            i += 1

@ray.remote
class StreamDataFromFile():
    def __init__(self, filepath, batch_size, loops):
        self.filepath = filepath
        self.batch_size = batch_size
        self.loops = loops
        self.max_iterations = 0

        self.data_on_disk = None
        self.maxsize = None
        self.dim = None
        self.data = asyncio.Queue()

        self.running = True

    def terminate(self):
        self.running = False

    def load_data_on_disk(self, filepath):
        data_on_disk = np.load(filepath, mmap_mode='r')
        print(f'data on disk: {sys.getsizeof(data_on_disk):12} bytes '
              f'({sys.getsizeof(data_on_disk) / GB:3.3f} GB)')
        return data_on_disk

    async def stream_data(self, queue, callback):
        print("Calling stream data")
        try:
            self.data_on_disk = load_data_on_disk(self.filepath)
            self.maxsize, self.dim = self.data_on_disk.shape
            self.max_iterations = int(self.maxsize/self.batch_size)
            print("Max iterations: ", self.max_iterations)
            print("max size: ", self.maxsize)
            print("Data on disk: ", self.data_on_disk)
        except FileNotFoundError:
            raise

        for loop in range(self.loops):
            print("Loop nr: ", loop)
            i = 0
            while i < self.max_iterations:
                time.sleep(0.1)
                print("Qeueue size: ", queue.qsize())
                if i % 10 == 0:
                    print("iteration: ", i)
                await queue.put_async(np.array(self.data_on_disk[i * self.batch_size:(i + 1) * self.batch_size]))
                i += 1
        self.terminate()
        callback.remote('EOF')

if __name__ == '__main__':
    filepath ='Data'
    filename = 'DummyData_n1000_d2.npy'
    filepath = os.path.join(filepath, filename)


