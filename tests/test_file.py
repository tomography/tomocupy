import lprec
from lprec.timing import *
import h5py
import dxchange

# sizes
[nz, nproj, n] = [1, 1500, 2448]  # 16 max for RTX4000 float32
[ndark, nflat] = [40, 80]
[ntheta, nrho] = [2048, 4096] # to check other sizes
data_type = 'uint8'

# take links to datasets in h5 file
fid = h5py.File('/local/ssd/286_2_spfp_019.h5', 'r')
data = fid['exchange/data']
flat = fid['exchange/data_white']
dark = fid['exchange/data_dark']
# angles are iitialized as np.arange(nproj)*np.pi/nproj

tic()
clpthandle = lprec.LpRec(n, nproj, nz, ntheta, nrho, ndark, nflat, data_type)
print(f'Init time: {toc():.3f}s')
tic()
clpthandle.recon_all(data, flat, dark)
print(f'Reconstruction time:{toc():.3f}s')
