=======
Install
=======


1. Create environment with necessary dependencies

::

  conda create -n tomocupy -c conda-forge python=3.9 dxchange cupy scikit-build swig pywavelets numexpr astropy olefile opencv
  conda activate tomocupy
  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

2. Install the pytorch pywavelets package for ring removal

::

  git clone https://github.com/fbcotter/pytorch_wavelets
  cd pytorch_wavelets
  pip install .
  cd -

3. Set path to the nvcc profiler (e.g. /local/cuda-11.4/bin/nvcc ) and install tomocupy

::

  export CUDACXX=/local/cuda-11.4/bin/nvcc 
  git clone https://github.com/nikitinvv/tomocupy-cli
  cd tomocupy-cli
  python setup.py install 


Update
======

**tomocupy-cli** is constantly updated to include new features. To update your locally installed version::

    $ cd tomocupy-cli
    $ git pull
    $ python setup.py install
