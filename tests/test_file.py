import lprec
from lprec.timing import *
import h5py
import dxchange

[nslices,nproj,n] = [16,1500,2448] # 32 for RTX4000
[ndark,nflat] = [40,80]
data_type = 'uint8'

fid = h5py.File('/local/ssd/286_2_spfp_019.h5', 'r')
data = fid['exchange/data']# why <=4 doesnt work???
flat = fid['exchange/data_white']
dark = fid['exchange/data_dark']
theta = fid['exchange/theta']

tic()
clpthandle = lprec.LpRec(n, nproj, nslices, ndark, nflat, data_type)
print(toc())
tic()
clpthandle.recon_all(data,flat,dark,theta)
print(f'Total time for reconstruction:{toc():.3f}s')
# print(f'Saving tiffs...')
# tic()
# # dxchange.write_tiff_stack(obj.astype('float32'), '/local/ssd/lprec/r', start=0, overwrite=True)
# print(f'Save tiffs time: {toc():.3f}s')
