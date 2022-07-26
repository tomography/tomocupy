#apply crop for both fp16,32fp
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float32 --remove-stripe-method fw --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/fp32.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float16 --remove-stripe-method fw --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/fp16.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float32 --remove-stripe-method fw --reconstruction-algorithm lprec --nsino-per-chunk 4 --start-row 1020 --end-row 1028 --nsino-per-chunk 4
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/lfp32.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float16 --remove-stripe-method fw --reconstruction-algorithm lprec --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/lfp16.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float32 --remove-stripe-method fw --reconstruction-algorithm linerec --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/linefp32.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --dtype float16 --remove-stripe-method fw --reconstruction-algorithm linerec --nsino-per-chunk 4 --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/linefp16.tiff
source ~/.bashrc;conda activate tomopy;
tomopy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --rotation-axis-auto manual --remove-stripe-method fw --start-row 1020 --end-row 1028
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/tfp32.tiff
source ~/.bashrc;conda activate tomopy;
tomopy recon --file-name /data/2022-07/Nikitin/glass_beads_2x_065.h5 --reconstruction-type full --rotation-axis 1025 --rotation-axis-auto manual --remove-stripe-method fw --start-row 1020 --end-row 1028 --gridrec-padding True;
cp /data/2022-07/Nikitin_rec/glass_beads_2x_065_rec/recon*1024.tiff res/tfppad32.tiff
linerec