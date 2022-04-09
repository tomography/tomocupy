============
Tomocupy-cli
============

**Tomocupy-cli** is a command-line interface for GPU reconstruction of tomographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using cupy library. The Fourier-based back-projection method is implemented with CUDA C++ and python wrappers. Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). It is also possible to reconstruct data from 360 degrees scans where the rotation axis is located at the border of the fields of view (option '--file-type double_fov').

Compared to **Tomocupy-cli** package, this package works faster and uses twice lower amount of GPU memory. Reconstructions are save as tiff16 files and can be opened with BIOP plugin (ImageJ->Import->BioFormats).

The 16-bit precision artihmetic works **ONLY** with reconstrction sizes n x n x nz where **n is a power of 2**.

=============
Documentation
=============

**Tomocupy-cli**  documentation is available here available `here <https://tomocupy.readthedocs.io/en/latest/>`_
