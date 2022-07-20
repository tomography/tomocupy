import h5py 
import numpy as np
with h5py.File(f'data/test_data.h5','a') as f:
    theta = f['exchange/theta']
    theta[:] = np.float32(np.arange(0,len(theta))*180/len(theta))
    
        # f.create_dataset("/exchange/data_white", chunks=(1, 1,shape[2]),data=16+np.zeros([10,*shape[1:]],dtype=dtype))
        # f.create_dataset("/exchange/data_dark", chunks=(1, 1,shape[2]),data=np.zeros([10,*shape[1:]],dtype=dtype))
        # f.create_dataset("/exchange/theta", data=np.float32(np.arange(n)/n*180))
