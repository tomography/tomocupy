================
H5GPURec
================

**h5gpurec** is a command-line interface for reconstruction tomographic hdf5 files on GPU.


Example of 'try' reconstruction:

.. image:: try.png
    :width: 90%
    :align: center
    
Example of 'full' reconstruction for double field of view:

.. image:: full.png
    :width: 90%
    :align: center


================
Installation
================
1. Create environment with necessary dependencies
================
::

  conda create -n h5gpurec -c conda-forge python=3.9 dxchange cupy scikit-build swig pywavelets
  conda activate h5gpurec
  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

2. Install of the pytorch pywavelets package for ring removal
================
::

  git clone https://github.com/fbcotter/pytorch_wavelets
  cd pytorch_wavelets
  pip install .
  cd -

3. Install h5gpurec
================
::

  git clone https://github.com/nikitinvv/h5gpurec
  cd h5gpurec
  python setup.py install

4. Usage with real data, see
================
::

  h5gpurec -h

5. Example
================
::
 
  h5gpurecon recon --file-name /data/2021-11/Banerjee/ROM_R_3474_072.h5 --rotation-axis 339 --reconstruction-type full --file-type double_fov --remove-stripe-method fw --binning 0 --nsino-per-chunk 8
