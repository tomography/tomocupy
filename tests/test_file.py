import numpy as np
import matplotlib.pyplot as plt
import lprec


[Nslices,Nproj,N] = [128,1500,1920]
clpthandle = lprec.LpRec(N, Nproj, Nslices)
clpthandle.recon_all('/local/data/lprec/coal_water_sd600_387.h5')
