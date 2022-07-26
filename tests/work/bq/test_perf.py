import h5py 
import numpy as np

shape = [2048,2048,2048]
dtype = 'uint8'
with h5py.File('/local/data/tmp.h5','w') as f:
    f.create_dataset("/exchange/data", chunks=(1, shape[1],shape[2]),data=8+np.zeros(shape,dtype=dtype))
    f.create_dataset("/exchange/data_white", chunks=(1, shape[1],shape[2]),data=16+np.zeros([10,*shape[1:]],dtype=dtype))
    f.create_dataset("/exchange/data_dark", chunks=(1, shape[1],shape[2]),data=np.zeros([10,*shape[1:]],dtype=dtype))
    f.create_dataset("/exchange/theta", data=np.float32(np.arange(2048)/2048*180))
