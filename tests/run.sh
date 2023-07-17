tomocupy recon_steps --file-name /local/data/tmp2048.h5 --reconstruction-type full --lamino-angle 20
tomocupy recon_steps --file-name /local/data/tmp4096.h5 --reconstruction-type full --lamino-angle 20
tomocupy recon_steps --file-name /local/data/tmp8192.h5 --reconstruction-type full --lamino-angle 20

tomocupy recon_steps --file-name /local/data/tmp2048.h5 --reconstruction-type full --lamino-angle 20 --reconstruction-algorithm linerec
tomocupy recon_steps --file-name /local/data/tmp4096.h5 --reconstruction-type full --lamino-angle 20 --reconstruction-algorithm linerec
tomocupy recon_steps --file-name /local/data/tmp8192.h5 --reconstruction-type full --lamino-angle 20 --reconstruction-algorithm linerec
