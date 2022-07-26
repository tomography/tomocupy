import h5py
import time
import numpy as np
import threading
import zarr
from numcodecs import Blosc,Zstd
from sys import stdout
from rechunker import rechunk


def write(data,a,k):
    data[k]=a
    
    
n=1200
a = np.random.random([n,1024,1024]).astype('float32')
f = h5py.File('/data/tmp.h5','w')
f.create_dataset('test1', (n,1024,1024),chunks=(1,1024,1024), dtype='f')
t = time.time()
threads = []
for k in range(n):
    write_thread = threading.Thread(target=write, args=(f['test1'],a[k],k,))
    threads.append(write_thread)
    write_thread.start()
for thread in threads:
    thread.join()
print(time.time()-t)    

f = zarr.open('/data/tmp2.zarr', mode='w')
d=f.create_dataset('test1',shape=(n,1024,1024),chunks=(1,1024,1024), dtype='f')
print(d.info)
t = time.time()
threads = []
for k in range(n):
    write_thread = threading.Thread(target=write, args=(f['test1'],a[k],k,))
    threads.append(write_thread)
    write_thread.start()
for thread in threads:
    thread.join()
print(time.time()-t)    
print(np.linalg.norm(f['test1']-a))

# t=time.time()
# c = zarr.array(d, chunks=(n,1,1024), store='c.zarr')
# print(time.time()-t)    

t=time.time()
target_chunks = {'test1': (n, 1, 1024)}
max_mem = "128GB"
plan = rechunk(f, target_chunks, max_mem,
               "/data/tmp3.zarr",
               temp_store="/data/temp_store.zarr")
plan.execute()
print(time.time()-t)    

# dest = zarr.open_group('/data/tmp2.zarr', mode='w')
# zarr.copy(f['test1'], dest, log=stdout)

#print(dest.info)
f= h5py.File('/data/tmp.h5','a')
t = time.time()
for k in range(1024):
    f['test1'][:,k]+=1





print(time.time()-t)        

t = time.time()
dest = zarr.open_group('/data/example.zarr', mode='w')
zarr.copy_all(f, dest)

print('copy',time.time()-t)        


f = zarr.open('/data/tmp3.zarr', mode='a')
t = time.time()
for k in range(1024):
    f['test1'][:,k]+=1
    # print(k)
print(time.time()-t)    
print(np.linalg.norm(f['test1']))

exit()

# #from mpi4py import MPI
# #print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())
"""A simple example of building a virtual dataset.
This makes four 'source' HDF5 files, each with a 1D dataset of 1000 numbers.
Then it makes a single 4x1000 virtual dataset in a separate file, exposing
the four sources as one dataset.
"""

import h5py
import numpy as np

# create some sample data
data = np.arange(0, 1000).reshape(1, 1000) + np.arange(1, 5).reshape(4, 1)

# Create source files (0.h5 to 3.h5)
for n in range(4):
    with h5py.File(f"/data/tmp/{n}.h5", "w") as f:
        d = f.create_dataset("data", (1000,), "i4", data[n])

# Assemble virtual dataset
layout = h5py.VirtualLayout(shape=(4, 1000), dtype="i4")
for n in range(4):
    filename = "/data/tmp/{}.h5".format(n)
    vsource = h5py.VirtualSource(filename, "data", shape=(1000,))
    layout[n] = vsource

# Add virtual dataset to output file
with h5py.File("/data/tmp/VDS.h5", "w") as f:
    f.create_virtual_dataset("vdata", layout, fillvalue=-5)
    f.create_dataset("data", data=data, dtype="i4")


# read data back
# virtual dataset is transparent for reader!
with h5py.File("/data/tmp/VDS.h5", "r") as f:
    print("Virtual dataset:")
    print(f["vdata"][:, :10])
    print("Normal dataset:")
    print(f["data"][:, :10])