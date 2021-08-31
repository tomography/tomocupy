import lprec
import h5py
import dxchange

# sizes
[nz, nproj, n] = [8, 1500, 2448]  # 16 max for RTX4000 float32
[ndark, nflat] = [40, 80]
center = 1230.5
[ntheta, nrho] = [2048, 4096] 
data_type = 'uint8'

fname = '/local/ssd/tmp/286_2_spfp_019.h5'
clpthandle = lprec.LpRec(n, nproj, nz, ntheta, nrho, ndark, nflat, data_type)
clpthandle.recon_all(fname,center)
