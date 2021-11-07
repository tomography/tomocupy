import cupy as cp
import numpy as np
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2

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
        
        

def create_adj(P):
    # convolution function
    fZ = cp.fft.fftshift(fzeta_loop_weights_adj(
        P.Ntheta, P.Nrho, 2*P.beta, P.g-np.log(P.am), 0, 4))

    # (C2lp1,C2lp2), transformed Cartesian to log-polar coordinates
    [x1, x2] = cp.meshgrid(cp.linspace(-1, 1, P.N), cp.linspace(-1, 1, P.N))
    x1 = x1.flatten()
    x2 = x2.flatten()
    #x2 = x2*(-1)  # adjust for tomopy
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

    const = (P.N+1)*(P.N-1)/P.N**2/2*np.sqrt(P.Nproj/P.N/2)
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
