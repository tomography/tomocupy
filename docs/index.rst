========
Tomocupy
========

**Tomocupy** is a package and a command-line interface for GPU reconstruction of tomographic/laminographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using cupy library, backprojection is implemented with CUDA C.
Tomocupy implements an efficient data processing conveyor. First, independent Python threads are started for reading data chunks from the hard disk into a Python data queue object (CPU RAM memory) and for writing reconstructed chunks from another Python queue object (CPU RAM memory) to the hard disk. 
Second, CPU-GPU data transfers are overlapped with GPU computations by using CUDA streams. 

Features
--------

* Fast tomographic reconstruction on GPU
* Fast laminographic reconstruction on GPU
* Efficeint conveyor data processing
* 16-bit or 32-bit arithmetics
* Manual rotation center search and automatic rotation center search with using SIFT algorithm
* Manual search of the laminographic tilting angle
* Additional preprocessing steps: ring removal with wavelets, phase retrieval procedure with Paganin filter, dezinger filtering
* Fourier-based method or direct discretization of the back-proejection operator for reconstruction
* Saving data in tiff and h5 formats



Contribute
----------

* Documentation: https://github.com/tomography/tomocupy/tree/master/doc
* Issue Tracker: https://github.com/tomography/tomocupy/docs/issues
* Source Code: https://github.com/tomography/tomocupy/

Content
-------

.. toctree::
   :maxdepth: 2

   source/install
   source/usage
   source/performance
   source/api
   source/credits
