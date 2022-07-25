from tomocupy import cfunc_lprec
from tomocupy import cfunc_lprecfp16
from tomocupy import cfunc_lprec
from tomocupy import logging
import cupy as cp
import numpy as np

log = logging.getLogger(__name__)
#cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

class Pgl:
    def __init__(self, Nspan, N, Nproj, Ntheta, Nrho, proj, s, thsp, rhosp, aR, beta, am, g, B3com):
        self.Nspan = Nspan
        self.N = N
        self.Nproj = Nproj
        self.Ntheta = Ntheta
        self.Nrho = Nrho
        self.proj = proj
        self.s = s
        self.thsp = thsp
        self.rhosp = rhosp
        self.aR = aR
        self.beta = beta
        self.am = am
        self.g = g
        self.B3com = B3com

class Padj:
    def __init__(self, fZ, lp2p1, lp2p2, lp2p1w, lp2p2w, C2lp1, C2lp2, cids, lpids, wids):
        self.fZ = fZ
        self.lp2p1 = lp2p1
        self.lp2p2 = lp2p2
        self.lp2p1w = lp2p1w
        self.lp2p2w = lp2p2w
        self.C2lp1 = C2lp1
        self.C2lp2 = C2lp2
        self.cids = cids
        self.lpids = lpids
        self.wids = wids
           
def create_gl(N, Nproj, Ntheta, Nrho):
    Nspan = 3
    beta = cp.pi/Nspan
    # size after zero padding in the angle direction (for nondense sampling rate)
    proj = cp.arange(0, Nproj)*cp.pi/Nproj-beta/2
    s = cp.linspace(-1, 1, N)
    # log-polar parameters
    (dtheta, drho, aR, am, g) = getparameters(
        beta, proj[1]-proj[0], 2.0/(N-1), N, Nproj, Ntheta, Nrho)
    # log-polar space
    thsp = (cp.arange(-Ntheta/2, Ntheta/2) *
            cp.float32(dtheta)).astype('float32')
    rhosp = (cp.arange(-Nrho, 0)*drho).astype('float32')
    erho = cp.tile(cp.exp(rhosp)[..., cp.newaxis], [1, Ntheta])
    # compensation for cubic interpolation
    B3th = splineB3(thsp, 1)
    B3th = cp.fft.fft(cp.fft.ifftshift(B3th))
    B3rho = splineB3(rhosp, 1)
    B3rho = (cp.fft.fft(cp.fft.ifftshift(B3rho)))
    B3com = cp.outer(B3rho,B3th)
    
    # struct with global parameters
    P = Pgl(Nspan, N, Nproj, Ntheta, Nrho, proj,
            s, thsp, rhosp, aR, beta, am, g, B3com)
    return P

def getparameters(beta, dtheta, ds, N, Nproj, Ntheta, Nrho):
    aR = cp.sin(beta/2)/(1+cp.sin(beta/2))
    am = (cp.cos(beta/2)-cp.sin(beta/2))/(1+cp.sin(beta/2))

    # wrapping
    g = osg(aR, beta/2)
    dtheta = (2*beta)/Ntheta
    drho = (g-cp.log(am))/Nrho
    return (dtheta, drho, aR, am, g)

def osg(aR, theta):
    t = cp.linspace(-cp.pi/2, cp.pi/2, 1000)
    w = aR*cp.cos(t)+(1-aR)+1j*aR*cp.sin(t)
    g = max(cp.log(abs(w))+cp.log(cp.cos(theta-cp.arctan2(w.imag, w.real))))
    return g

def splineB3(x2, r):
    sizex = len(x2)
    x2 = x2-(x2[-1]+x2[0])/2
    stepx = x2[1]-x2[0]
    ri = int(cp.ceil(2*r))
    r = r*stepx
    x2c = x2[int(cp.ceil((sizex+1)/2.0))-1]
    x = x2[int(cp.ceil((sizex+1)/2.0)-ri-1):int(cp.ceil((sizex+1)/2.0)+ri)]
    d = cp.abs(x-x2c)/r
    B3 = x*0
    for ix in range(-ri, ri+1):
        id = ix+ri
        if d[id] < 1:  # use the first polynomial
            B3[id] = (3*d[id]**3-6*d[id]**2+4)/6
        else:
            if(d[id] < 2):
                B3[id] = (-d[id]**3+6*d[id]**2-12*d[id]+8)/6
    B3f = x2*0
    B3f[int(cp.ceil((sizex+1)/2.0)-ri-1):int(cp.ceil((sizex+1)/2.0)+ri)] = B3
    return B3f
 
def create_adj(P):
    # convolution function
    fZ = cp.fft.fftshift(fzeta_loop_weights_adj(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-np.log(P.am), 0, 4))

    # (C2lp1,C2lp2), transformed Cartesian to log-polar coordinates
    [x1, x2] = cp.meshgrid(cp.linspace(-1, 1, P.N), cp.linspace(-1, 1, P.N))
    x1 = x1.flatten()
    x2 = x2.flatten()
    x2 = x2*(-1)  # adjust for tomocupy
    x1-=1/P.N
    x2-=1/P.N
    cids = cp.where(x1**2+x2**2 <= 1)[0].astype('int32')
    C2lp1 = cp.zeros([P.Nspan,len(cids)],dtype='float32')
    C2lp2 = cp.zeros([P.Nspan,len(cids)],dtype='float32')
    for k in range(0, P.Nspan):
        z1 = P.aR*(x1[cids]*cp.cos(k*P.beta+P.beta/2)+x2[cids]
                   * cp.sin(k*P.beta+P.beta/2))+(1-P.aR)
        z2 = P.aR*(-x1[cids]*cp.sin(k*P.beta+P.beta/2) +
                   x2[cids]*cp.cos(k*P.beta+P.beta/2))
        C2lp1[k] = cp.arctan2(z2, z1)
        C2lp2[k] = cp.log(cp.sqrt(z1**2+z2**2))
    
    # (lp2p1,lp2p2), transformed log-polar to polar coordinates
    [z1, z2] = cp.meshgrid(P.thsp, cp.exp(P.rhosp))
    z1 = z1.flatten()
    z2 = z2.flatten()
    z2n = z2-(1-P.aR)*cp.cos(z1)
    z2n = z2n/P.aR
    lpids = cp.where((z1 >= -P.beta/2) & (z1 < P.beta/2) & (abs(z2n) <= 1))[0].astype('int32')
    lp2p1 = cp.zeros([P.Nspan,len(lpids)],dtype='float32')
    lp2p2 = cp.zeros([P.Nspan,len(lpids)],dtype='float32')    
    for k in range(P.Nspan):
        lp2p1[k] = (z1[lpids]+k*P.beta)
        lp2p2[k] = z2n[lpids]
    # (lp2p1w,lp2p2w), transformed log-polar to polar coordinates (wrapping)
    # right side
    wids = cp.where(cp.log(z2) > +P.g)[0].astype('int32')
    
    z2n = cp.exp(cp.log(z2[wids])+cp.log(P.am)-P.g)-(1-P.aR)*cp.cos(z1[wids])
    z2n = z2n/P.aR
    lpidsw = cp.where((z1[wids] >= -P.beta/2) &
                      (z1[wids] < P.beta/2) & (abs(z2n) <= 1))[0]
    # left side
    wids2 = cp.where(cp.log(z2) < cp.log(P.am)-P.g+(P.rhosp[1]-P.rhosp[0]))[0].astype('int32')
    z2n2 = cp.exp(cp.log(z2[wids2])-cp.log(P.am)+P.g) - \
        (1-P.aR)*cp.cos(z1[wids2])
    z2n2 = z2n2/P.aR
    lpidsw2 = cp.where((z1[wids2] >= -P.beta/2) &
                       (z1[wids2] < P.beta/2) & (abs(z2n2) <= 1))[0]
    lp2p1w = cp.zeros([P.Nspan,len(lpidsw)+len(lpidsw2)],dtype='float32')
    lp2p2w = cp.zeros([P.Nspan,len(lpidsw)+len(lpidsw2)],dtype='float32')    
    for k in range(P.Nspan):
        lp2p1w[k] = (z1[cp.concatenate((lpidsw, lpidsw2))]+k*P.beta)
        lp2p2w[k] = cp.concatenate((z2n[lpidsw], z2n2[lpidsw2]))
    # join for saving
    wids = cp.concatenate((wids[lpidsw], wids2[lpidsw2])).astype('int32')
    
    # pids, index in polar grids after splitting by spans
    pids = [None]*P.Nspan
    for k in range(P.Nspan):
        pids[k] = cp.where((P.proj >= k*P.beta-P.beta/2) &
                           (P.proj < k*P.beta+P.beta/2))[0]

    # first angle and length of spans
    proj0 = [None]*P.Nspan
    projl = [None]*P.Nspan
    for k in range(P.Nspan):
        proj0[k] = P.proj[pids[k][0]]
        projl[k] = P.proj[pids[k][-1]]-P.proj[pids[k][0]]

    #shift in angles
    projp = (P.Nproj-1)/(proj0[P.Nspan-1]+projl[P.Nspan-1]-proj0[0])

    # adapt for interpolation
    for k in range(P.Nspan):
        lp2p1[k] = (lp2p1[k]-proj0[k])/projl[k] * \
            (len(pids[k])-1)+(proj0[k]-proj0[0])*projp
        lp2p1w[k] = (lp2p1w[k]-proj0[k])/projl[k] * \
            (len(pids[k])-1)+(proj0[k]-proj0[0])*projp
        lp2p2[k] = (lp2p2[k]+1)/2*(P.N-1)
        lp2p2w[k] = (lp2p2w[k]+1)/2*(P.N-1)
        C2lp1[k] = (C2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
        C2lp2[k] = (C2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)

    const = (P.N+1)*(P.N-1)/P.N**2/2/np.sqrt(2)*np.pi/6*0.86*4 # to understand where this is coming from
    fZ = cp.ascontiguousarray(fZ[:, :P.Ntheta//2+1]/(P.B3com[:, :P.Ntheta//2+1]))*const  
    
                    
    Padj0 = Padj(fZ, lp2p1, lp2p2, lp2p1w, lp2p2w,
                 C2lp1, C2lp2, cids, lpids, wids)
    return Padj0


def fzeta_loop_weights_adj(Ntheta, Nrho, betas, rhos, a, osthlarge):
    krho = cp.arange(-Nrho/2, Nrho/2, dtype='float32')
    Nthetalarge = osthlarge*Ntheta
    thsplarge = cp.arange(-Nthetalarge/2, Nthetalarge/2,
                          dtype='float32') / Nthetalarge*betas
    fZ = cp.zeros([Nrho, Nthetalarge], dtype='complex64')
    h = cp.ones(Nthetalarge, dtype='float32')
    # correcting = 1+[-3 4 -1]/24correcting(1) = 2*(correcting(1)-0.5)
    # correcting = 1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0correcting[0] = 2*(correcting[0]-0.5)
    correcting = 1+cp.array([-216254335, 679543284, -1412947389, 2415881496, -3103579086,
                             2939942400, -2023224114, 984515304, -321455811, 63253516, -5675265])/958003200.0
    correcting[0] = 2*(correcting[0]-0.5)
    h[0] = h[0]*(correcting[0])
    for j in range(1, len(correcting)):
        h[j] = h[j]*correcting[j]
        h[-1-j+1] = h[-1-j+1]*(correcting[j])
    for j in range(len(krho)):
        fcosa = pow(cp.cos(thsplarge), (2*cp.pi*1j*krho[j]/rhos-a))
        fZ[j, :] = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(h*fcosa)))
    fZ = fZ[:, Nthetalarge//2-Ntheta//2:Nthetalarge//2+Ntheta//2]
    fZ = fZ*(thsplarge[1]-thsplarge[0])
    # put imag to 0 for the border
    fZ[0] = 0
    fZ[:, 0] = 0
    return fZ

class LpRec():
    def __init__(self, n, nproj, nz, theta, dtype):        

        # check angles 
        nproj_test = int(np.round(np.pi/(theta[1]-theta[0])))
        if nproj != nproj_test:
            log.error('lprec method works only with equally spaced angles in the interval [0,180) deg.')
            exit(1)
        ntheta = 2**int(cp.round(cp.log2(nproj)))
        nrho = 2*2**int(cp.round(cp.log2(n)))        
        log.info(f'Log-polar grid sizes: {ntheta=},{nrho=}')
        # precompute parameters for the lp method
        self.Pgl = create_gl(n, nproj, ntheta, nrho)
        self.Padj = create_adj(self.Pgl)
        lp2p1 = self.Padj.lp2p1.data.ptr
        lp2p2 = self.Padj.lp2p2.data.ptr
        lp2p1w = self.Padj.lp2p1w.data.ptr
        lp2p2w = self.Padj.lp2p2w.data.ptr
        C2lp1 = self.Padj.C2lp1.data.ptr
        C2lp2 = self.Padj.C2lp2.data.ptr
        # conevrt fZ from complex to float.. coudl be improved..
        fZn = cp.zeros(
            [self.Padj.fZ.shape[0], self.Padj.fZ.shape[1]*2], dtype=dtype)
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
        if dtype == 'float16':
            self.fslv = cfunc_lprecfp16.cfunc_lprec(nproj, nz, n, ntheta, nrho)
        else:
            self.fslv = cfunc_lprec.cfunc_lprec(nproj, nz, n, ntheta, nrho)

        self.fslv.setgrids(fZnptr, lp2p1, lp2p2, lp2p1w, lp2p2w,
                         C2lp1, C2lp2, lpids, wids, cids,
                         nlpids, nwids, ncids)

    def backprojection(self,obj, data, stream):
        data = cp.ascontiguousarray(data)###????
        self.fslv.backprojection(obj.data.ptr, data.data.ptr,stream.ptr)


