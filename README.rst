========
Tomocupy
========

**Tomocupy** is a Python package and a command-line interface for GPU reconstruction of tomographic/laminographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using CuPy library, the backprojection operation is implemented with CUDA C.
The current implementation works with h5 data files having the following structure::

/exchange/data
/exchange/data_white
/exchange/data_dark
/exchange/theta

For other files structures, please adjust src/reader.py. For reconstruction working with numpy arrays see https://github.com/nikitinvv/tomocupy-stream with a jupyter notebook example in tests/test_for_compression.ipynb.

**Tomocupy**  documentation is available `here <https://tomocupy.readthedocs.io/en/latest/>`_.

