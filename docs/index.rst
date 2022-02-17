============
Tomocupy-cli
============

**Tomocupy-cli** is a command-line interface for GPU reconstruction of tomographic data. All preprocessing operations are implemented on GPU with using cupy library. Two backprojection methods: Fourier-based (fourierrec) and Log-polar-based (lprec) are implemented with CUDA C++ and python wrappers. lprec works only with equally-spaced angles in the interval [0,180), fourierrec supports arbitrary angles. Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). It is also possible to reconstruct data from 360 degrees scans where the rotation axis is located at the border of the fields of view (option '--file-type double_fov').


Features
--------

* List here 
* the module features


Contribute
----------

* Documentation: https://github.com/tomography/tomocupy-cli/tree/master/doc
* Issue Tracker: https://github.com/tomography/tomocupy-cli/docs/issues
* Source Code: https://github.com/tomography/tomocupy-cli/

Content
-------

.. toctree::
   :maxdepth: 2

   source/install
   source/usage
   source/performance
   source/api
   source/credits
