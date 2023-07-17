import h5py 
import numpy as np
import sys
# sizes = [512,1024,2048,4096,8192,16384]
# nz = [512,1024,2048,2048,1024,256]
sizes = [1024,2048,4096,8192]
# nz = [1024,2048,256,64]
# muls = [1,1,1,2,32,256]
for i,n in enumerate(sizes):
    print(n)
    shape = [n,n,n]
    dtype = 'uint8'
    with h5py.File(f'/local/data/tmp{n}.h5','w') as f:
        f.create_dataset("/exchange/data", chunks=(shape[0], 1,shape[2]),data=8+np.zeros(shape,dtype=dtype))
        f.create_dataset("/exchange/data_white", chunks=(1, 1,shape[2]),data=16+np.zeros([10,*shape[1:]],dtype=dtype))
        f.create_dataset("/exchange/data_dark", chunks=(1, 1,shape[2]),data=np.zeros([10,*shape[1:]],dtype=dtype))
        f.create_dataset("/exchange/theta", data=np.float32(np.arange(n)/n*180))
