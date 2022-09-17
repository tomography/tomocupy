# tomocupy recon --file-name /local/ssd/data/2019-12/exp1_vertical_2_206.h5 --reconstruction-type full --rotation-axis-auto auto --dtype float32 --nsino-per-chunk 4 --start-row 512 --end-row 1536
# tomocupy recon --file-name /local/ssd/data/2019-12/exp1_vertical_2_213.h5 --reconstruction-type full --rotation-axis-auto auto --dtype float32 --nsino-per-chunk 4 --start-row 512 --end-row 1536
# tomocupy recon --file-name /local/ssd/data/2019-12/exp1_vertical_2_220.h5 --reconstruction-type full --rotation-axis-auto auto --dtype float32 --nsino-per-chunk 4 --start-row 512 --end-row 1536
tomocupy recon_steps --file-name /local/ssd/data/2019-12/exp1_roi_3_249.h5 --reconstruction-type full \
 --rotation-axis 1169 --dtype float32 --nsino-per-chunk 4  --retrieve-phase-method paganin \
 --energy 20 --propagation-distance 150 --retrieve-phase-alpha 0.00012 --fbp-filter shepp --pixel-size 0.69 --start-row 200 --end-row 472

# tomocupy recon_steps --file-name /local/ssd/data/2019-12/exp1_roi_3_498.h5 --reconstruction-type full \
#  --rotation-axis 1169 --dtype float32 --nsino-per-chunk 4  --retrieve-phase-method paganin \
#  --energy 20 --propagation-distance 150 --retrieve-phase-alpha 0.00008 --fbp-filter shepp --pixel-size 0.69 --start-row 200 --end-row 472

 tomocupy recon_steps --file-name /local/ssd/data/2019-12/exp1_roi_3_482.h5 --reconstruction-type full \
 --rotation-axis 1156 --dtype float32 --nsino-per-chunk 4  --retrieve-phase-method paganin \
 --energy 20 --propagation-distance 150 --retrieve-phase-alpha 0.00012 --fbp-filter shepp --pixel-size 0.69 --start-row 200 --end-row 472




cp /local/ssd/data/2019-12_rec/exp1_vertical_2_206_rec/recon_01296.tiff res/206.tiff
cp /local/ssd/data/2019-12_rec/exp1_vertical_2_213_rec/recon_01296.tiff res/213.tiff
cp /local/ssd/data/2019-12_rec/exp1_roi_3_249_rec/recon_00336.tiff res/249.tiff
cp /local/ssd/data/2019-12_rec/exp1_roi_3_482_rec/recon_00336.tiff res/482.tiff