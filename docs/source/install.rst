=======
Install
=======


1. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c conda-forge python=3.9 cupy scikit-build swig pywavelets numexpr astropy olefile opencv
    (base)$ conda activate tomocupy
    (tomocupy)$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

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


Update
======

**tomocupy** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ git pull
    (tomocupy)$ python setup.py install
