import lprec
from lprec.timing import *
import h5py
import dxchange

# sizes
[nz, nproj, n] = [16, 1500, 2048]  # 16 max for RTX4000 float32
[ndark, nflat] = [1, 1]
[ntheta, nrho] = [2048, 4096] 
data_type = 'uint16'

# take links to datasets in h5 file
# fid = h5py.File('/local/ssd/286_2_spfp_019.h5', 'r')
fname = '/local/ssd/data/tmp/tomo_00001.h5'
# fname = '/local/ssd/286_2_spfp_019.h5'
# fid = h5py.File(fname, 'r')

# data = fid['exchange/data']
# flat = fid['exchange/data_white']
# dark = fid['exchange/data_dark']
# theta = fid['exchange/theta']

# print(data.shape)
# # print(theta[:])
# exit()
# angles are iitialized as np.arange(nproj)*np.pi/nproj

tic()
clpthandle = lprec.LpRec(n, nproj, nz, ntheta, nrho, ndark, nflat, data_type)
print(f'Init time: {toc():.3f}s')
tic()
clpthandle.recon_all(fname)
print(f'Reconstruction time:{toc():.3f}s')
