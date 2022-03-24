tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --center-search-width 4
tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --center-search-width 4 --save-format h5
tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --center-search-width 4 --save-format h5 --file-type double_fov --rotation-axis 100
tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --reconstruction-type full --start-row 900 --end-row 1000
tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --reconstruction-type full --save-format h5 --start-row 900 --end-row 1000
tomocupy recon --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --reconstruction-type full --save-format h5 --file-type double_fov --nsino-per-chunk 4 --rotation-axis 100 --start-row 900 --end-row 1000
tomocupy reconstep --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --retrieve-phase-method paganin --reconstruction-type full --start-row 900 --end-row 1000
tomocupy reconstep --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --retrieve-phase-method paganin --reconstruction-type full --save-format h5 --start-row 900 --end-row 1000
tomocupy reconstep --file-name /local/ssd/data/tomo_00002.h5 --remove-stripe-method fw --retrieve-phase-method paganin --reconstruction-type full --file-type double_fov --nsino-per-chunk 4 --start-row 900 --end-row 1000