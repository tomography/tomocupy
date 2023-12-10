import h5py 
import numpy as np
import sys
n=int(sys.argv[1])
ntheta = int(sys.argv[2])
shape = [ntheta,n,n]
dtype = 'uint8'
with h5py.File(f'/local/tmp/tmp{n}.h5','w') as f:
    f.create_dataset("/exchange/data", chunks=(shape[0], 1,shape[2]),data=8+np.zeros(shape,dtype=dtype))
    f.create_dataset("/exchange/data_white", chunks=(1, 1,shape[2]),data=16+np.zeros([10,*shape[1:]],dtype=dtype))
    f.create_dataset("/exchange/data_dark", chunks=(1, 1,shape[2]),data=np.zeros([10,*shape[1:]],dtype=dtype))
    f.create_dataset("/exchange/theta", data=np.float32(np.arange(n)/n*360))
