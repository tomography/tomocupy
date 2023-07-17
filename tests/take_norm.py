import numpy as np
import dxchange 

data = dxchange.read_tiff_stack('data_rec/chip_rec/recon_00000.tiff',ind=range(0,256))

print(np.linalg.norm(data))