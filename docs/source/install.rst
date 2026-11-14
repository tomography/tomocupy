=====================
Installation and test
=====================

Tomocupy works in NVidia GPUs of compute capability 6.0 and higher. To run tomocupy the system should have an `nvidia driver installed <https://developer.nvidia.com/cuda-downloads>`_. Cuda Toolkit is not necessary.
Run ``nvidia-smi`` to check whether the driver is installed. For check compute capability of the GPU, see `this document <http://mylifeismymessage.net/find-the-compute-capability-of-your-nvidia-graphics-card-gpu/>`_. 

0. For faster installation of packages, it is better to use `Miniforge <https://github.com/conda-forge/miniforge>`_. Then go to Step 3.

1. For regular Anaconda, add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict


2. Environmental solver mamba works much faster than the regular one, use

::

    (base)$ conda install -n base conda-libmamba-solver
    (base)$ conda config --set solver libmamba

3. Create environment with installed tomocupy

::

    (base)$ conda create -n tomocupy tomocupy

4. Activate tomocupy environment

::

    (base)$ conda activate tomocupy
    

5. Test installation

::

    (tomocupy)$ tomocupy recon -h

============================
Installation for development
============================

0. For faster installation of packages, it is better to use `Miniforge <https://github.com/conda-forge/miniforge>`_. Then go to Step 3.

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict

2. Environmental solver mamba works much faster than the regular one, use

::

    (base)$ conda install -n base conda-libmamba-solver
    (base)$ conda config --set solver libmamba

3. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c conda-forge cupy scikit-build numexpr opencv tifffile h5py cmake ninja swig scipy pywavelets python=3.10


.. warning:: Conda has a built-in mechanism to determine and install the latest version of cudatoolkit supported by your driver. However, if for any reason you need to force-install a particular CUDA version (say 11.0), you can do:

  conda install -c conda-forge cupy cudatoolkit=11.0


4. Activate tomocupy environment

::

    (base)$ conda activate tomocupy

5*. (If needed) Install meta for supporting hdf meta data writer used by option: --save-format h5

::

    (tomocupy)$ git clone https://github.com/xray-imaging/meta.git
    (tomocupy)$ cd meta
    (tomocupy)$ pip install .
    (tomocupy)$ cd -


6. Make sure ``nvcc`` is on ``PATH``:

::

    (tomocupy)$ which nvcc
    (tomocupy)$ nvcc --version

If ``nvcc`` is not found, point at a CUDA toolkit install. Two common setups:

- System CUDA toolkit at e.g. ``/usr/local/cuda-12.1/``:

::

    (tomocupy)$ export CUDAHOME=/usr/local/cuda-12.1
    (tomocupy)$ export CUDA_PATH=$CUDAHOME
    (tomocupy)$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDAHOME/lib64
    (tomocupy)$ export PATH=$PATH:$CUDAHOME/bin

- NVIDIA HPC SDK loaded via environment modules:

::

    (tomocupy)$ module use /local/nvidia/hpc_sdk/modulefiles
    (tomocupy)$ module add nvhpc-hpcx-cuda13/26.1

Add the chosen lines to ``~/.bashrc`` to make the setting persistent across logins.

7. Install tomocupy

Build on a local (non-NFS) filesystem such as ``/tmp`` or ``/local``. This avoids an NFS silly-rename race in scikit-build's test-compile cleanup that breaks the build when the source tree lives on a shared / NFS-mounted home directory (typical on cluster machines such as APS beamline workstations). The installed package lands in your conda env regardless of where the build happened.

::

    (tomocupy)$ git clone https://github.com/tomography/tomocupy /tmp/tomocupy-build
    (tomocupy)$ cd /tmp/tomocupy-build
    (tomocupy)$ pip install .
    (tomocupy)$ cd ~ && rm -rf /tmp/tomocupy-build

.. note::
    ``cupy`` must be installed separately via conda (see Step 3) as it is CUDA-version specific and is not declared as a pip dependency.

.. note::
    The CUDA toolkit (``nvcc``) major version does not need to match ``cupy``'s runtime CUDA version exactly — tomocupy's CUDA kernels are simple enough that minor mismatches (e.g. ``nvcc`` 12.x compiling against a 13.x runtime) typically work. If the build succeeds but the kernels fail at runtime, align the toolkit and runtime to the same major version.

===================================
Additional instructions for Windows
===================================

#. Install `Build VS 2019 utils <https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019>`_

#. Install CUDA toolkit, e.g. `cuda 11_2_2 <https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64>`_

#. Install `Anaconda for windows <https://docs.anaconda.com/free/anaconda/install/windows/>`_ and use Powershell in which tomocupy environment can be created

.. note::
    It is better to have only one version of VS and one version of CUDA toolkit on your system to avoid problems with environmental variables

================================
AI-based Center of Rotation
================================

To use the AI-based center of rotation detection (``--rotation-axis-method ai``), install the following additional dependencies::

    (tomocupy)$ conda install -c conda-forge pytorch pillow einops

Then run reconstruction with::

    (tomocupy)$ tomocupy recon --file-name <file.h5> --rotation-axis-method ai --rotation-axis-auto auto --infer-model-path <path/to/model.pth>

``--infer-model-path`` must be set when using ``--rotation-axis-method ai``. For users within the Argonne National Laboratory, download the model from https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq.

To find the center of rotation across a larger range, the hierarchical search feature is supported. Run reconstruction with::

    (tomocupy)$ tomocupy recon --file-name <file.h5> --rotation-axis-method ai --rotation-axis-auto auto --infer-model-path <path/to/model.pth> --ai-search-method full --bin-infer-model-path <path/to/range_model.pth>

``--bin-infer-model-path`` must be set when using ``--rotation-axis-method ai`` and ``--ai-search-method full``. For users within the Argonne National Laboratory, download the range model from https://anl.box.com/s/7kcmv9yav2j9y1c0b8lwhc1bp43tyrop.

==========
Unit tests
==========
Check the library path to cuda or set it by ``export LD_LIBRARY_PATH=/local/cuda-11.7/lib64``

Run the following to check all functionality
::

    (tomocupy)$ cd tests; bash test_all.sh


===============
Troubleshooting
===============

**Build fails with** ``OSError: [Errno 39] Directory not empty: 'CMakeFiles'`` **or** ``OSError: [Errno 16] Device or resource busy: '.nfsXXXXXXXX'``

This happens when the source tree is on an **NFS-mounted filesystem** (typical for shared cluster home directories such as ``/home/beams/...`` at APS). scikit-build's ``cleanup_test()`` calls ``shutil.rmtree`` on the test-compile directory while the just-finished CMake test process still has open file handles. NFS does not delete files that are still held open; instead it renames them to ``.nfsXXXXXXXX`` ("silly rename"), which makes ``rmtree`` fail because the directory is "not empty" or the silly-rename file is "busy".

The first ``pip install .`` of a fresh install almost always hits this. The retry succeeds because the file handles from the failed first attempt have closed by the time you re-run, so ``rm -rf`` can fully clear the directory and the second build's cleanup finds it empty.

Workaround (always works):

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ rm -rf _cmake_test_compile _skbuild build *.egg-info
    (tomocupy)$ pip install .

Permanent alternative — build on a local (non-NFS) filesystem such as ``/tmp`` or ``/local``:

::

    (tomocupy)$ cp -r ~/path/to/tomocupy /tmp/tomocupy-build
    (tomocupy)$ cd /tmp/tomocupy-build
    (tomocupy)$ pip install .
    (tomocupy)$ rm -rf /tmp/tomocupy-build

The installed package lands in your conda env (which is itself on NFS, but install-time writes don't trigger silly-rename because nothing else has those files open).

**Build fails with** ``CMake Error ... No CMAKE_CUDA_COMPILER could be found``

``nvcc`` is not on ``PATH``. Re-do Step 6 of "Installation for development" (verify with ``which nvcc``) and retry.

**Build fails because pip cannot reach PyPI (private/air-gapped network)**

On beamline workstations, HPC login nodes, or other machines without direct internet access, ``pip install .`` fails during the build-dependency fetch step with a timeout or connection error::

    ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.

Install the build dependencies into the conda env first (only needed once), then build with ``--no-build-isolation`` so pip uses what is already installed instead of fetching from PyPI:

::

    (tomocupy)$ conda install -n tomocupy -c conda-forge scikit-build cmake ninja swig
    (tomocupy)$ pip install --no-build-isolation .

**Runtime crash: "incomplete type __nv_fp8_e8m0" errors from cupy**

``cupy 14`` bundles CCCL headers that reference ``__nv_fp8_e8m0``, a type introduced in CUDA 12.8. If cupy discovers a system CUDA toolkit older than 12.8 at runtime, every kernel compilation (e.g. inside ``find_center_vo``) fails with a cascade of errors like::

    error: incomplete type "__nv_fp8_e8m0" is not allowed

This happens when ``CUDA_PATH``, ``CUDAHOME``, or ``nvcc`` on ``PATH`` points cupy to the system toolkit instead of its own bundled headers. The fix is to downgrade cupy to a version compatible with your system CUDA:

::

    (base)$ conda install -n tomocupy -c conda-forge cupy=12 cuda-version=<your_system_cuda>

For example, for a system with CUDA 12.1::

    (base)$ conda install -n tomocupy -c conda-forge cupy=12 cuda-version=12.1

To find your system CUDA version run ``nvcc --version`` or ``nvidia-smi``.

**Conda solver picks an incompatible CUDA version**

If your driver supports e.g. CUDA 12.x but conda installs ``cupy`` against a newer cudart, pin the version explicitly:

::

    (base)$ conda create -n tomocupy -c conda-forge cupy scikit-build numexpr opencv tifffile h5py cmake ninja pywavelets cuda-version=12.9 python=3.10

Match the ``cuda-version`` value to what ``nvidia-smi`` reports as the maximum supported.

**Runtime crash: HDF5 BlockingIOError when writing output on NFS**

When the reconstruction output (``--save-format h5``, ``h5nolinks``, or ``h5sino``, including the default ``h5nolinks``) is written to an NFS-mounted directory such as ``/data2/2BM/...``, the writer can fail with::

    BlockingIOError: [Errno 11] Unable to synchronously create file
      (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')

HDF5 ≥ 1.10 acquires a POSIX file lock before truncating or creating a file, and NFS does not reliably support that locking primitive. The failure rate is highest when an output file from a previous run already exists at the same path — HDF5 must lock it to truncate before re-creating.

Fix: tell HDF5 to skip file locking by setting an environment variable in the ``tomocupy`` conda env:

::

    (tomocupy)$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    (tomocupy)$ echo 'export HDF5_USE_FILE_LOCKING=FALSE' \
                > $CONDA_PREFIX/etc/conda/activate.d/hdf5_no_lock.sh
    (tomocupy)$ conda deactivate && conda activate tomocupy
    (tomocupy)$ echo $HDF5_USE_FILE_LOCKING    # should print: FALSE

Every ``conda activate tomocupy`` shell — including the one tomocupy runs in — will now have the variable set. The setting is the documented HDF5 remedy for NFS; it is safe as long as you are not doing concurrent writes to the same file from multiple processes.

For a one-off shell without changing the env, just export it directly before running::

    (tomocupy)$ export HDF5_USE_FILE_LOCKING=FALSE
    (tomocupy)$ tomocupy recon_steps ...

Update
======

**tomocupy** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ git pull
    (tomocupy)$ pip install .



Installation on Polaris supercomputer
=====================================
1. connect to Polaris main node (computing nodes don't have access to the internet)  and install anaconda

2. add modules:

::

    module add gcc/11.2.0
    module add cudatoolkit-standalone/11.4.4

.. note::
    We work with cuda-11.4 not with cuda-12.1 because the current driver version on polaris is 11.4

3. create tomocupy environment, specifying cudatoolkit=11.4

::

    conda create -n tomocupy -c conda-forge cupy scikit-build numexpr opencv tifffile h5py cmake ninja cudatoolkit=11.4

4. clone tomocupy:

::

    git clone https://github.com/tomography/tomocupy

5. install tomocupy

::

    cd tomocupy; pip install .

6. test tomocupy:

:: 

    tomocupy recon -h

7. connect to a node with GPUs in interactive mode and a debug allocation for now, smth like

::

    qsub -I -A hp-ptycho -l select=4:system=polaris -l filesystems=home:eagle -l walltime=30:00 -q debug-scaling

.. note::
    Replace hp-ptycho by your project

8. test tomocupy:

::

    cd tests; bash test_all.sh
