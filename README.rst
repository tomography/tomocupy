================
Tomocupy-cli
================

**Tomocupy-cli** is a command-line interface for GPU reconstruction of tomographic data. All preprocessing operations are implemented on GPU with using cupy library. Two backprojection methods: Fourier-based (fourierrec) and Log-polar-based (lprec) are implemented with CUDA C++ and python wrappers. lprec works only with equally-spaced angles in the interval [0,180), fourierrec supports arbitrary angles. Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). It is also possible to reconstruct data from 360 degrees scans where the rotation axis is located at the border of the fields of view (option '--file-type double_fov').



================
Installation
================
1. Create environment with necessary dependencies
================
::

  conda create -n tomocupy -c conda-forge python=3.9 dxchange cupy scikit-build swig pywavelets numexpr astropy olefile opencv
  conda activate tomocupy
  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

2. Install the pytorch pywavelets package for ring removal
================
::

  git clone https://github.com/fbcotter/pytorch_wavelets
  cd pytorch_wavelets
  pip install .
  cd -

3. Set path to the nvcc profiler (e.g. /local/cuda-11.4/bin/nvcc ) and install tomocupy
================
::

  export CUDACXX=/local/cuda-11.4/bin/nvcc 
  git clone https://github.com/nikitinvv/tomocupy-cli
  cd tomocupy-cli
  python setup.py install 

4. Usage with real data, see
================
::

  tomocupy -h

5. Example
================
::
 
  tomocupy recon --file-name /data/2021-11/Banerjee/ROM_R_3474_072.h5 --rotation-axis 339 --reconstruction-type full --file-type double_fov --remove-stripe-method fw --binning 0 --nsino-per-chunk 8


6. Extra functionality for reconstruction with phase retrieval. Data splitting is done by steps involving splitting by slices and projections + automatic center search with using SIFT for projection pairs. Example
================
::
 
  tomocupy reconstep --file-name /data/2021-12/Duchkov/exp4_ho_130_vertical_0_2018.h5 --remove-stripe-method fw --nproj-per-chunk 32 --nsino-per-chunk 32 --retrieve-phase-alpha 0.001 --retrieve-phase-method none  --binning 0 --reconstruction-type full --rotation-axis 1198 --rotation-axis-pairs [0,1200,599,1799,300,1500] --rotation-axis-auto auto --start-row 400 --end-row 1800