=====
Usage
=====

Example
=======

::
 
    (tomocupy)$ tomocupy recon --file-name /data/2021-11/Banerjee/ROM_R_3474_072.h5 --rotation-axis 339 --reconstruction-type full --file-type double_fov --remove-stripe-method fw --binning 0 --nsino-per-chunk 8


Extra functionality for reconstruction with phase retrieval. Data splitting is done by steps involving splitting by slices and projections + automatic center search with using SIFT for projection pairs. Example

::
 
    (tomocupy)$ tomocupy recon_steps --file-name /data/2021-12/Duchkov/exp4_ho_130_vertical_0_2018.h5 --remove-stripe-method fw --nproj-per-chunk 32 --nsino-per-chunk 32 --retrieve-phase-alpha 0.001 --retrieve-phase-method none  --binning 0 --reconstruction-type full --rotation-axis 1198 --rotation-axis-pairs [0,1200,599,1799,300,1500] --rotation-axis-auto auto --start-row 400 --end-row 1800


More options
============
::

    (tomocupy)$ tomocupy -h
    (tomocupy)$ tomocupy recon -h
    (tomocupy)$ tomocupy recon_steps -h