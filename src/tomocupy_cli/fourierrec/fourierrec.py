from tomocupy_cli.cfunc_fourierrec import cfunc_fourierrec
import cupy as cp

class FourierRec(cfunc_fourierrec):
    def __init__(self, n, nproj, nz, theta):         
        super().__init__(nproj, nz//2, n, theta.data.ptr)

    def backprojection(self, obj, data, stream): 
        [nz,n] = obj.shape[:2]
        datac = cp.ascontiguousarray(data[:data.shape[0]//2]+1j*data[data.shape[0]//2:]) #cant reuse memory 
        objc = obj.reshape(nz//2,n,2*n) # reuse memory
        super().backprojection(objc.data.ptr, datac.data.ptr,stream.ptr)        
        obj[:] = cp.concatenate((objc[:,:,::2],objc[:,:,1::2]))
        