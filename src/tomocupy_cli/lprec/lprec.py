from h5gpurec.lprec import initsgl
from h5gpurec.lprec import initsadj
from h5gpurec.lprec import initsadj
from h5gpurec.cfunc_lprec import cfunc_lprec
import cupy as cp

class LpRec(cfunc_lprec):
    def __init__(self, n, nproj, nz):        
        ntheta = 2**int(cp.round(cp.log2(nproj)))
        nrho = 2*2**int(cp.round(cp.log2(n)))
        # precompute parameters for the lp method
        self.Pgl = initsgl.create_gl(n, nproj, ntheta, nrho)
        self.Padj = initsadj.create_adj(self.Pgl)
        lp2p1 = self.Padj.lp2p1.data.ptr
        lp2p2 = self.Padj.lp2p2.data.ptr
        lp2p1w = self.Padj.lp2p1w.data.ptr
        lp2p2w = self.Padj.lp2p2w.data.ptr
        C2lp1 = self.Padj.C2lp1.data.ptr
        C2lp2 = self.Padj.C2lp2.data.ptr
        # conevrt fZ from complex to float.. coudl be improved..
        fZn = cp.zeros(
            [self.Padj.fZ.shape[0], self.Padj.fZ.shape[1]*2], dtype='float32')
        fZn[:, ::2] = self.Padj.fZ.real
        fZn[:, 1::2] = self.Padj.fZ.imag
        self.fZn = fZn  # keep in class, otherwise collector will remove it
        fZnptr = cp.ascontiguousarray(self.fZn).data.ptr
        lpids = self.Padj.lpids.data.ptr
        wids = self.Padj.wids.data.ptr
        cids = self.Padj.cids.data.ptr
        nlpids = len(self.Padj.lpids)
        nwids = len(self.Padj.wids)
        ncids = len(self.Padj.cids)
        super().__init__(nproj, nz, n, ntheta, nrho)
        super().setgrids(fZnptr, lp2p1, lp2p2, lp2p1w, lp2p2w,
                         C2lp1, C2lp2, lpids, wids, cids,
                         nlpids, nwids, ncids)
    def backprojection(self,obj, data, stream):
        super().backprojection(obj.data.ptr, data.data.ptr,stream.ptr)