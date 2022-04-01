============
Tomocupy-cli-fp16
============

**Tomocupy-cli-fp16** is a command-line interface for GPU reconstruction of tomographic data in 16-bit precision. All preprocessing operations are implemented on GPU with using cupy library. Two backprojection methods: Fourier-based (fourierrec) and Log-polar-based (lprec) are implemented with CUDA C++ and python wrappers. lprec works only with equally-spaced angles in the interval [0,180), fourierrec supports arbitrary angles. Ring removal is implemented with using the pytorch wavelet library on GPU.

The package supports two types of reconstructions: manual center search (option '--reconstruction-type try') and whole volume reconstruction (option '--reconstruction-type full'). It is also possible to reconstruct data from 360 degrees scans where the rotation axis is located at the border of the fields of view (option '--file-type double_fov').

Compared to **Tomocupy-cli** package, this package works faster and uses twice lower amount of GPU memory. Reconstructions are save as tiff16 files and can be opened with BIOP plugin (ImageJ->Import->BioFormats).

The package works **ONLY** with reconstrction sizes n x n x nz where **n is a power of 2**.

=============
Documentation
=============

**Tomocupy-cli-fp16**  documentation is the same as for **Tomocupy-cli** and available here available `here <https://tomocupy.readthedocs.io/en/latest/>`_
