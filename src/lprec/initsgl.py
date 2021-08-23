import cupy as cp


class Pgl:
    def __init__(self, Nspan, N, Nproj, Ntheta, Nrho, proj, s, thsp, rhosp, aR, beta, am, g):
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


def create_gl(N, Nproj):
    Nspan = 3
    beta = cp.pi/Nspan
    # size after zero padding in the angle direction (for nondense sampling rate)
    proj = cp.arange(0, Nproj)*cp.pi/Nproj-beta/2
    s = cp.linspace(-1, 1, N)
    # log-polar parameters
    (Nrho, Ntheta, dtheta, drho, aR, am, g) = getparameters(
        beta, proj[1]-proj[0], 2.0/(N-1), N, Nproj)
    # log-polar space
    thsp = (cp.arange(-Ntheta/2, Ntheta/2) *
            cp.float32(dtheta)).astype('float32')
    rhosp = (cp.arange(-Nrho, 0)*drho).astype('float32')
    erho = cp.tile(cp.exp(rhosp)[..., cp.newaxis], [1, Ntheta])
    # struct with global parameters
    P = Pgl(Nspan, N, Nproj, Ntheta, Nrho, proj,
            s, thsp, rhosp, aR, beta, am, g)
    return P


def getparameters(beta, dtheta, ds, N, Nproj):
    aR = cp.sin(beta/2)/(1+cp.sin(beta/2))
    am = (cp.cos(beta/2)-cp.sin(beta/2))/(1+cp.sin(beta/2))

    # wrapping
    g = osg(aR, beta/2)
    Ntheta = 2048#N
    Nrho = 4096#2*N
    dtheta = (2*beta)/Ntheta
    drho = (g-cp.log(am))/Nrho
    return (Nrho, Ntheta, dtheta, drho, aR, am, g)


def osg(aR, theta):
    t = cp.linspace(-cp.pi/2, cp.pi/2, 1000)
    w = aR*cp.cos(t)+(1-aR)+1j*aR*cp.sin(t)
    g = max(cp.log(abs(w))+cp.log(cp.cos(theta-cp.arctan2(w.imag, w.real))))
    return g
