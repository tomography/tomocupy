=======
Install
=======


1. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c pytorch -c conda-forge scikit-build swig pywavelets numexpr astropy olefile opencv tifffile h5py pytorch torchvision torchaudio cudatoolkit=11.3

2. Install the pytorch pywavelets package for ring removal

::

    (tomocupy)$ git clone https://github.com/fbcotter/pytorch_wavelets
    (tomocupy)$ cd pytorch_wavelets
    (tomocupy)$ pip install .
    (tomocupy)$ cd -

3. Make sure that the path to nvcc compiler is set (or set it by e.g. 'export CUDACXX=/local/cuda-11.4/bin/nvcc') and install tomocupy

::
    
    (tomocupy)$ git clone https://github.com/nikitinvv/tomocupy
    (tomocupy)$ cd tomocupy
    (tomocupy)$ python setup.py install 

==========
Unit tests
==========
Run the following to check all functionality
::

    (tomocupy)$ cd tests; bash test_all.sh


Update
======

**tomocupy** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ git pull
    (tomocupy)$ python setup.py install
