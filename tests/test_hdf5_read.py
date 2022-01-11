import h5py
import time
import numpy as np
import threading
from sys import stdout


def read_write(data,a,k):
    data[k*n//nthreads:(k+1)*n//nthreads]=a[k*n//nthreads:(k+1)*n//nthreads][:]
    
nthreads = 1
n = 1500
a = np.ones([n,2048,2048]).astype('uint16')
f = h5py.File('/local/data/tmp.h5','w')
f.create_dataset('test1', (n,2048,2048),chunks=(1,2048,2048), dtype='uint16')
t = time.time()
threads = []
for k in range(nthreads):
    write_thread = threading.Thread(target=read_write, args=(f['test1'],a,k))
    threads.append(write_thread)
    write_thread.start()
for thread in threads:
    thread.join()
print(time.time()-t)    
f.close()


a = np.zeros([n,2048,2048],'uint16')
f = h5py.File('/local/data/tmp.h5','r')
threads = []
t = time.time()
for k in range(n):
    read_thread = threading.Thread(target=read_write, args=(a,f['test1'],k))
    threads.append(read_thread)
    read_thread.start()
for thread in threads:
    thread.join()

print(time.time()-t)    
print(f'{n*2048*2048*2/1024/1024/1024/(time.time()-t)}GB/s')    
