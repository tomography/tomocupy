import lprec
import h5py
def create_h5():
    fid = h5py.File(fname,"w")
    data = fid.create_dataset("/exchange/data", proj_shape,
                                        chunks=(1, proj_shape[1], proj_shape[2]), dtype='u8')
    data_white = fid.create_dataset("/exchange/data_white", [nflat, proj_shape[1], proj_shape[2]],
                                        chunks=(1, proj_shape[1], proj_shape[2]), dtype='u8')

    data_dark = fid.create_dataset("/exchange/data_dark", [ndark, proj_shape[1], proj_shape[2]],
                                        chunks=(1, proj_shape[1], proj_shape[2]), dtype='u8')
    data[:] = 200
    data_white[:] = 1
    data_dark[:] = 0
    fid.close()

# sizes
[nz, nproj, n] = [16, 1500, 2048]  # 16 max for RTX4000 float32
[ndark, nflat] = [1, 1]
center = 1024
[ntheta, nrho] = [2048, 4096] 
data_type = 'uint16'
proj_shape = [nproj,2048,n]

fname = '/local/ssd/tmp/t.h5'
create_h5() # create if not exist
clpthandle = lprec.LpRec(n, nproj, nz, ntheta, nrho, ndark, nflat, data_type)
clpthandle.recon_all(fname,center)
