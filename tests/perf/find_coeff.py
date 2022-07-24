import h5py 
import numpy as np
import sys
import dxchange
import os
shape=[4096,16,1024]
with h5py.File(f'/local/data/tmp_coeff.h5','w') as f:
        data = np.ones(shape,dtype='float32')
        data[:,:,shape[-1]//4:-shape[-1]//4] = 2
        f.create_dataset("/exchange/data", chunks=(shape[0], 1,shape[2]),data=data)
        f.create_dataset("/exchange/data_white", chunks=(1, 1,shape[2]),data=4+np.zeros([10,*shape[1:]],dtype=np.float32))
        f.create_dataset("/exchange/data_dark", chunks=(1, 1,shape[2]),data=np.zeros([10,*shape[1:]],dtype=np.float32))
        f.create_dataset("/exchange/theta", data=np.float32(np.arange(shape[0])/shape[0]*180))

os.system('tomocupy recon --file-name /local/data/tmp_coeff.h5 --reconstruction-type full ')
rec_fourier = dxchange.read_tiff_stack('/local/data_rec/tmp_coeff_rec/recon_00000.tiff',ind=range(0,shape[1]))[:]

os.system('tomocupy recon --file-name /local/data/tmp_coeff.h5 --reconstruction-type full --reconstruction-algorithm lprec')
rec_lprec = dxchange.read_tiff_stack('/local/data_rec/tmp_coeff_rec/recon_00000.tiff',ind=range(0,shape[1]))[:]
n1=np.linalg.norm(rec_fourier[:,shape[-1]//4:-shape[-1]//4,shape[-1]//4:-shape[-1]//4])
n2=np.linalg.norm(rec_lprec[:,shape[-1]//4:-shape[-1]//4,shape[-1]//4:-shape[-1]//4])
print(n1/n2)