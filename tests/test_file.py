import lprec
from lprec.timing import *
import h5py
import dxchange

[Nslices,Nproj,N] = [16,1500,2448] # 32 for RTX4000

fid = h5py.File('/local/ssd/286_2_spfp_019.h5', 'r')
data = fid['exchange/data'][:,:]# why <=4 doesnt work???
flat = fid['exchange/data_white'][:,:]
dark = fid['exchange/data_dark'][:,:]
theta = fid['exchange/theta'][:]

tic()
clpthandle = lprec.LpRec(N, Nproj, Nslices)
print(toc())
tic()
obj = clpthandle.recon_all(data,flat,dark,theta)
print(f'Total time for reconstruction:{toc():.3f}s')
print(f'Saving tiffs...')
tic()
# dxchange.write_tiff_stack(obj.astype(
    # 'float32'), '/local/ssd/lprec/r', start=0, overwrite=True)
print(f'Save tiffs time: {toc():.3f}s')
