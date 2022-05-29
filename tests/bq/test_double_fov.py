import h5lprec

# sizes
[nz, nproj, n] = [16, 3000, 2448]  # 16 max for RTX4000 float32
[ndark, nflat] = [40, 40]
center = 34
[ntheta, nrho] = [2048, 4096] 
data_type = 'uint8'

fname = '/local/data/pu_b01_02_166.h5'
clpthandle = h5lprec.H5LpRec(n, nproj, nz, ntheta, nrho, ndark, nflat, data_type,center,True)
clpthandle.recon_all(fname)
