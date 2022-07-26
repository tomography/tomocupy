from gettext import npgettext


import numpy as np 
import h5py 
import cupyx.scipy.ndimage as ndimage
import cupy as cp

with h5py.File('/data/2022-07/Nikitin/glass_beads_2x_065.h5','a') as fid:    
    data = fid['/exchange/data_dark'][:]
    datanew = np.zeros([20,2048,2048],dtype='uint8')    
    datag = cp.array(data)
    datanew = ndimage.zoom(datag,(1,1,2048/2448.0)).get()
    del fid['/exchange/data_dark']
    fid.create_dataset('/exchange/data_dark',data=datanew)
    
    data = fid['/exchange/data_white'][:]
    datanew = np.zeros([20,2048,2048],dtype='uint8')    
    datag = cp.array(data)
    datanew = ndimage.zoom(datag,(1,1,2048/2448.0)).get()
    del fid['/exchange/data_white']
    fid.create_dataset('/exchange/data_white',data=datanew)
    
    data = fid['/exchange/data'][:]
    datanew = np.zeros([1800,2048,2048],dtype='uint8')
    del fid['/exchange/data']
    for k in range(10):
        print(k)
        datag = cp.array(data[k*180:(k+1)*180])
        print(datag.dtype)
        datanew[k*180:(k+1)*180] = ndimage.zoom(datag,(1,1,2048/2448.0)).get()
    fid.create_dataset('/exchange/data',data=datanew)
