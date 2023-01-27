=======
Install
=======

1. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c conda-forge cupy scikit-build swig pywavelets numexpr opencv tifffile h5py python=3.10

2. Activate tomocupy environment

::

    (base)$ conda activate tomocupy

3. Install pytorch

::

    (tomocupy)$ pip install torch torchvision torchaudio 


4. Install the pytorch pywavelets package for ring removal

::

    (tomocupy)$ git clone https://github.com/fbcotter/pytorch_wavelets
    (tomocupy)$ cd pytorch_wavelets
    (tomocupy)$ pip install .
    (tomocupy)$ cd -

3. Make sure that the path to nvcc compiler is set (or set it by e.g. 'export CUDACXX=/local/cuda-11.7/bin/nvcc') and install tomocupy

::
    
    (tomocupy)$ git clone https://github.com/tomography/tomocupy
    (tomocupy)$ cd tomocupy
    (tomocupy)$ pip install .

==========
Unit tests
==========
Check the library path to cuda or set it by 'export LD_LIBRARY_PATH=/local/cuda-11.7/lib64'

Run the following to check all functionality
::

    (tomocupy)$ cd tests; bash test_all.sh


Update
======

**tomocupy** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ git pull
    (tomocupy)$ pip install .
