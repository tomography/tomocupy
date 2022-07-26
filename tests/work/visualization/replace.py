import h5py 
import numpy as np
with h5py.File(f'/local/ssd/data/glass_beads_1x_119.h5','a') as f:
    f["/exchange/theta"][:] = np.float32(np.arange(1500)/1500*180)
    data = f['exchange/data'][:]
    dark = f['exchange/data_dark'][:]
    white = f['exchange/data_white'][:]

    del f['exchange/data']
    del f['exchange/data_dark']
    del f['exchange/data_white']

    f.create_dataset("/exchange/data", data=data[:,:,200:-200])
    f.create_dataset("/exchange/data_dark", data=dark[:,:,200:-200])
    f.create_dataset("/exchange/data_white", data=white[:,:,200:-200])

        # f.create_dataset("/exchange/data_white", chunks=(1, 1,shape[2]),data=16+np.zeros([10,*shape[1:]],dtype=dtype))
        # f.create_dataset("/exchange/data_dark", chunks=(1, 1,shape[2]),data=np.zeros([10,*shape[1:]],dtype=dtype))
        # f.create_dataset("/exchange/theta", data=np.float32(np.arange(n)/n*180))
