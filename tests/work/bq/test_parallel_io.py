import h5py
import time
import numpy as np
import threading
import dxchange
from ast import literal_eval
import sys
import zarr

def read_write_hdf5(data,a,k):
    data[k*ntheta//nthreads:(k+1)*ntheta//nthreads]=a[k*ntheta//nthreads:(k+1)*ntheta//nthreads][:]

def write_tiff(a,k):
    dxchange.write_tiff_stack(a[k*ntheta//nthreads:(k+1)*ntheta//nthreads],f'{path}/t/t',start=k*ntheta//nthreads,overwrite=True)

def read_tiff(a,k):
    a[k*ntheta//nthreads:(k+1)*ntheta//nthreads] = dxchange.read_tiff_stack(f'{path}/t/t_00000.tiff',ind=np.arange(k*ntheta//nthreads,(k+1)*ntheta//nthreads))


nthreads = int(sys.argv[1])
[ntheta,nz,n] = literal_eval(sys.argv[2])
path = sys.argv[3]


# a = np.random.random([ntheta,nz,n]).astype('float32')
# f = h5py.File(f'{path}tmp.h5','w')
# f.create_dataset('test1', (ntheta,nz,n),chunks=(1,nz,n), dtype='float32')
# t = time.time()
# threads = []
# for k in range(nthreads):
#     write_thread = threading.Thread(target=read_write_hdf5, args=(f['test1'],a,k))
#     threads.append(write_thread)
#     write_thread.start()
# for thread in threads:
#     thread.join()
# f.close()

# print(f'time: {(time.time()-t)} write hdf5 speed {a.nbytes/1024/1024/1024/(time.time()-t)}GB/s')    



# a = np.random.random([ntheta,nz,n]).astype('float32')
# f = h5py.File(f'{path}tmp.h5','r')
# threads = []
# t = time.time()
# for k in range(n):
#     read_thread = threading.Thread(target=read_write_hdf5, args=(a,f['test1'],k))
#     threads.append(read_thread)
#     read_thread.start()
# for thread in threads:
#     thread.join()

# print(f'time: {(time.time()-t)} read hdf5 speed {a.nbytes/1024/1024/1024/(time.time()-t)}GB/s')    
# f.close()

# a = np.random.random([ntheta,nz,n]).astype('float32')
# threads = []
# t = time.time()
# for k in range(nthreads):
#     write_thread = threading.Thread(target=write_tiff,args=(a,k))
#     threads.append(write_thread)
#     write_thread.start()
# for thread in threads:
#     thread.join()
# print(f'time: {(time.time()-t)} write tiff speed {a.nbytes/1024/1024/1024/(time.time()-t)}GB/s')    




# a = np.random.random([ntheta,nz,n]).astype('float32')
# threads = []
# t = time.time()
# for k in range(nthreads):
#     write_thread = threading.Thread(target=read_tiff,args=(a,k))
#     threads.append(write_thread)
#     write_thread.start()
# for thread in threads:
#     thread.join()
# print(f'time: {(time.time()-t)} read tiff speed {a.nbytes/1024/1024/1024/(time.time()-t)}GB/s')    








a = np.random.random([ntheta,nz,n]).astype('float32')
f = zarr.open_array(f'{path}tmp.zarr', mode='w', shape=(ntheta,nz,n),chunks=(1,nz,n), dtype='float32', synchronizer=zarr.ThreadSynchronizer())

t = time.time()
threads = []
for k in range(nthreads):
    write_thread = threading.Thread(target=read_write_hdf5, args=(f,a,k))
    threads.append(write_thread)
    write_thread.start()
for thread in threads:
    thread.join()
f.close()
