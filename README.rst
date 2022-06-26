============
Tomocupy-cli
============

**Tomocupy-cli** is a command-line interface for GPU reconstruction of tomographic/laminographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using cupy library. The Fourier-based back-projection method is implemented with CUDA C++ and python wrappers.  Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). It is also possible to reconstruct data from 360 degrees scans where the rotation axis is located at the border of the fields of view (option '--file-type double_fov').

The package also support laminographic reconstruction where the backprojection operator is implemented with regular discretization of line intergrals with linear interpolation.

=============
Tests
=============
To check the package works properly, run  

cd tests; python tests_all.py

=============
Documentation
=============

**Tomocupy-cli**  documentation is available here available `here <https://tomocupy.readthedocs.io/en/latest/>`_
