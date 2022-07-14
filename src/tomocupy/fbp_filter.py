from tomocupy import cfunc_filter
from tomocupy import cfunc_filterfp16
import cupy as cp


class FBPFilter():
    def __init__(self, n, ntheta, nz, dtype):
        if dtype == 'float16':
            self.fslv = cfunc_filterfp16.cfunc_filter(ntheta, nz, n)
        else:
            self.fslv = cfunc_filter.cfunc_filter(ntheta, nz, n)


    def filter(self, data, w, stream):
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(data)
        w = cp.ascontiguousarray(w.view('float32').astype(data.dtype))
        self.fslv.filter(data.data.ptr, w.data.ptr, stream.ptr)
        