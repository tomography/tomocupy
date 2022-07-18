#apply crop for both fp16,32fp
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1235.5 --dtype float32 --remove-stripe-method fw
cp /data/2022-07/Nikitin_rec/glass_beads_1x_119_rec/recon*1024.tiff res/fp32.tiff
tomocupy recon --file-name /data/2022-07/Nikitin/glass_beads_1x_119.h5 --reconstruction-type full --rotation-axis 1235.5 --dtype float16 --remove-stripe-method fw
cp /data/2022-07/Nikitin_rec/glass_beads_1x_119_rec/recon*1024.tiff res/fp16.tiff

