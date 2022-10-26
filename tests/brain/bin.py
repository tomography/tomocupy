import h5py
import sys
import numexpr as ne

def downsampling(data,bin):
    for j in range(bin):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
    for k in range(bin):
        x = data[:, ::2]
        y = data[:, 1::2]
        data = ne.evaluate('x + y')
        
infile = sys.argv[1]
bin = sys.argv[2]

outfile = f'{sys.argv[1][:-3]}_bin{bin}.h5'
with h5py.File(infile,'r') as f:
    with h5py.File(outfile,'w') as fout:        
        
        data = f['/exchange/data']
        data_dark = f['/exchange/data_dark']
        data_white = f['/exchange/data_white']
        theta = f['/exchange/theta'][::2**bin]
        
        print('init new dataset')
        fout.create_dataset('exchange/data',chunks=(1,data.shape[1]//2**bin,data.shape[2]//2**bin),shape=(data.shape[0]//2**bin,data.shape[1]//2**bin,data.shape[2]//2**bin),dtype='float32')        
        data_dark0 = downsampling(data_dark[:].astype('float32'),bin)             
        data_white0 = downsampling(data_white[:].astype('float32'),bin)                     
        fout.create_dataset('exchange/data_dark',chunks=(1,*data_dark0.shape[1:]),data=data_dark0)
        fout.create_dataset('exchange/data_white',chunks=(1,*data_white0.shape[1:]),data=data_white0)
        fout.create_dataset('exchange/theta',data=theta)
        
        print('set projections')
        for k in range(0,data.shape[0],2**bin):
            print(k)
            data0 = downsampling(data[k:k+1],bin)             
            fout['exchange/data'][k] = data0
        
        
        