=====================
Installation and test
=====================

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict

2. Create environment with installed tomocupy

::

    (base)$ conda create -n tomocupy tomocupy

3. Activate tomocupy environment

::

    (base)$ conda activate tomocupy
    

3. Test installation

::

    (tomocupy)$ tomocupy recon -h

============================
Installation for development
============================

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict

2. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c conda-forge cupy scikit-build swig pywavelets numexpr opencv tifffile h5py python=3.9


.. warning:: Conda has a built-in mechanism to determine and install the latest version of cudatoolkit supported by your driver. However, if for any reason you need to force-install a particular CUDA version (say 11.0), you can do:
  
  conda install -c conda-forge cupy cudatoolkit=11.0
  

3. Activate tomocupy environment

::

    (base)$ conda activate tomocupy

4*. (If needed) Install meta for supporting hdf meta data writer used by option: --save-format h5

::

    (tomocupy)$ git clone https://github.com/xray-imaging/meta.git
    (tomocupy)$ cd meta
    (tomocupy)$ pip install .
    (tomocupy)$ cd -


5. Make sure that the path to nvcc compiler is set (or set it by e.g. 'export CUDACXX=/local/cuda-11.7/bin/nvcc') and install tomocupy

::
    
    (tomocupy)$ git clone https://github.com/tomography/tomocupy
    (tomocupy)$ cd tomocupy
    (tomocupy)$ pip install .

===================================
Additional instructions for Windows
===================================

Install Build VS 2019 utils:

https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019

Install CUDA toolkit, e.g. 

https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64

Note: it is better to have only 1 version of VS and 1 version of CUDA toolkit on your system to avoid problems with environmental variables

Install Anaconda for windows https://docs.anaconda.com/free/anaconda/install/windows/ and use Powershell in which tomocupy environment can be created
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
