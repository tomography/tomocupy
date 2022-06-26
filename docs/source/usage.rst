=====
Usage
=====

Examples
=======
Try center reconstruction
::
   (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --reconstruction-type try --center-search-width 100

Full volume reconstruction
::
   (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full

Double FOV reconstruction
::
    (tomocupy)$ (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full --file-type double_fov

Full volume reconstruction with phase retrieval
::
    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full --energy 20 --pixel-size 1.75 --propagation-distance 100 --retrieve-phase-alpha 0.001 --retrieve-phase-method paganin --reconstruction-type full 

Laminographic try reconstruction
::
    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8 --reconstruction-type try --center-search-width 100 --lamino-angle 20

Laminographic try angle reconstruction
::
    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8 --rotation-axis 700 --reconstruction-type try-lamino --lamino-search-width 2 --lamino-angle 20

Laminographic full reconstruction
::
    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8--reconstruction-type full --rotation-axis 700 --lamino-angle 20

More options
============
::

    (tomocupy)$ tomocupy -h
    (tomocupy)$ tomocupy recon -h
    (tomocupy)$ tomocupy recon_steps -h
