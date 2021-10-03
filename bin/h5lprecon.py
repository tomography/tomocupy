import sys
import h5lprec
import h5py
import numpy as np
import time

def main():
    fname = sys.argv[1]
    print(fname)
    rotation_axis = float(sys.argv[2])
    nzc = int(sys.argv[3])
    double_fov = sys.argv[4]=='True'
    with h5py.File(fname) as fid:
        [nproj,nz,n] = fid['/exchange/data'].shape
        ndark = fid['/exchange/data_dark'].shape[0]
        nflat = fid['/exchange/data_white'].shape[0]    
        data_type= fid['/exchange/data'].dtype
        
    ntheta = 2**int(np.round(np.log2(nproj)))
    nrho = 2*2**int(np.round(np.log2(n)))

    print(f'{fname=}')
    print(f'{rotation_axis=}')
    print(f'{nproj=}')
    print(f'{nz=}')
    print(f'{n=}')
    print(f'{ndark=}')
    print(f'{nflat=}')
    print(f'{data_type=}')
    print(f'{ntheta=}')
    print(f'{nrho=}')
    print(f'{nzc=}')
    print(f'{double_fov=}')

    print('Create reconstruction class')
    clpthandle = h5lprec.H5LpRec(n, nproj, nzc, ntheta, nrho, ndark, nflat, data_type,rotation_axis,double_fov)
    print('Reconstruction')
    t = time.time()
    clpthandle.recon_all(fname)
    print('')
    print(f'Time {time.time()-t}s')




if __name__ == '__main__':
    main()
