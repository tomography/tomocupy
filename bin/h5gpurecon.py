import sys
import h5gpurec
import h5py
import numpy as np
import time

def main():
    fname = sys.argv[1]
    print(fname)
    rotation_axis = float(sys.argv[2])
    nzc = int(sys.argv[3])
    double_fov = sys.argv[4]=='True'
    method = sys.argv[5]
    with h5py.File(fname) as fid:
        [nproj,nz,n] = fid['/exchange/data'].shape
        ndark = fid['/exchange/data_dark'].shape[0]
        nflat = fid['/exchange/data_white'].shape[0]    
        data_type= fid['/exchange/data'].dtype
        theta = fid['/exchange/theta'][:]
        
    print(f'{fname=}')
    print(f'{rotation_axis=}')
    print(f'{nproj=}')
    print(f'{nz=}')
    print(f'{n=}')
    print(f'{ndark=}')
    print(f'{nflat=}')
    print(f'{data_type=}')
    print(f'{nzc=}')
    print(f'{double_fov=}')

    print('Create reconstruction class')
    print(theta)
    clpthandle = h5gpurec.H5GPURec(n, nproj, nzc, ndark, nflat, data_type, rotation_axis, double_fov, method, theta)
    print('Reconstruction')
    t = time.time()
    clpthandle.recon_all(fname)
    print('')
    print(f'Time {time.time()-t}s')




if __name__ == '__main__':
    main()
