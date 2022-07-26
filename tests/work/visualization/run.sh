#apply crop for both fp16,32fp
tomocupy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --dtype float32 --remove-stripe-method fw --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/fp32.tiff
# tomocupy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --dtype float16 --remove-stripe-method fw --nsino-per-chunk 4 --start-row 1020 --end-row 1028
# cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/fp16.tiff
# tomocupy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --dtype float32 --remove-stripe-method fw --reconstruction-algorithm lprec --nsino-per-chunk 4 --start-row 1020 --end-row 1028 --nsino-per-chunk 4
# cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/lfp32.tiff
# tomocupy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --dtype float16 --remove-stripe-method fw --reconstruction-algorithm lprec --nsino-per-chunk 4 --start-row 1020 --end-row 1028
# cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/lfp16.tiff
# tomocupy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --dtype float32 --remove-stripe-method fw --reconstruction-algorithm linerec --nsino-per-chunk 4 --start-row 1020 --end-row 1028
# cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/linefp32.tiff
# source ~/.bashrc;conda activate tomopy;
# tomopy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --rotation-axis-auto manual --remove-stripe-method fw --start-row 1020 --end-row 1028
# cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/tfp32.tiff

source ~/.bashrc;conda activate tomopy;
tomopy recon --file-name /local/ssd/data//glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1035.5 --rotation-axis-auto manual --remove-stripe-method fw --start-row 1020 --end-row 1028 --gridrec-padding True;
cp /local/ssd/data_rec/glass_beads_1x_119_rec/recon*1024.tiff res/tfp32.tiff
