import numpy as np
import dxchange
import os
os.system('rm -rf /home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec*')
cmd = 'tomocupy recon_steps --reconstruction-type full --lamino-angle -10 --file-name /home/beams/TOMO/conda/tomocupy-dev/tests/data/chip.h5  --minus-log False'
os.system(cmd)
a = dxchange.read_tiff_stack('/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec/recon_00000.tiff',ind=range(0,192))
dxchange.write_tiff_stack(a,'/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec0/recon.tiff',overwrite=True)
print(np.linalg.norm(a))

cmd = 'tomocupy recon_steps --reconstruction-type full --lamino-angle -10 --file-name /home/beams/TOMO/conda/tomocupy-dev/tests/data/chip.h5 --start-row 0 --end-row 96 --minus-log False'
os.system(cmd)
b = dxchange.read_tiff_stack('/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec/recon_00000.tiff',ind=range(0,192))
cmd = 'tomocupy recon_steps --reconstruction-type full --lamino-angle -10 --file-name /home/beams/TOMO/conda/tomocupy-dev/tests/data/chip.h5 --start-row 96 --end-row 192 --minus-log False'
os.system(cmd)
c = dxchange.read_tiff_stack('/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec/recon_00000.tiff',ind=range(0,192))
dxchange.write_tiff_stack(b+c,'/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec2/recon.tiff',overwrite=True)

print(np.linalg.norm(b+c))

# cmd = 'tomocupy recon_steps --reconstruction-type full --lamino-angle -10 --file-name /home/beams/TOMO/conda/tomocupy-dev/tests/data/chip.h5 --lamino-start-row 0 --lamino-end-row 96 --minus-log False'
# os.system(cmd)
# # b = dxchange.read_tiff_stack('/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec/recon_00000.tiff',ind=range(0,100))
# cmd = 'tomocupy recon_steps --reconstruction-type full --lamino-angle -10 --file-name /home/beams/TOMO/conda/tomocupy-dev/tests/data/chip.h5 --lamino-start-row 96 --lamino-end-row 192 --minus-log False'
# os.system(cmd)
# c = dxchange.read_tiff_stack('/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec/recon_00000.tiff',ind=range(0,192))
# # dxchange.write_tiff_stack(b+c,'/home/beams/TOMO/conda/tomocupy-dev/tests/data_rec/chip_rec2/recon.tiff',overwrite=True)

# print(np.linalg.norm(c))


