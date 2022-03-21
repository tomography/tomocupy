from tomocupyfp16_cli.cfunc_fourierrec import cfunc_fourierrec
import cupy as cp

class FourierRec(cfunc_fourierrec):
    def __init__(self, n, nproj, nz, theta):         
        super().__init__(nproj, nz//2, n, theta.data.ptr)

    def backprojection(self, obj, data, stream): 
        [nz,n] = obj.shape[:2]
        # reorganize data as a complex array, reuse data
        data[:] = cp.ascontiguousarray(cp.concatenate((data[:data.shape[0]//2,:,:,cp.newaxis],data[data.shape[0]//2:,:,:,cp.newaxis]),axis=3).reshape(data.shape))
        # reuse obj array
        objc = cp.ascontiguousarray(obj.reshape(nz//2,n,2*n)) 
        super().backprojection(obj.data.ptr, data.data.ptr,stream.ptr)        
        obj[:] = cp.concatenate((objc[:,:,::2],objc[:,:,1::2]))
    
    def filter(self, data, w, stream): 
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(data)                
        w = cp.ascontiguousarray((w/data.shape[2]).view('float32').astype('float16'))        
        
        super().filter(data.data.ptr, w.data.ptr,stream.ptr)        
    