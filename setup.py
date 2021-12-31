"""This is the main entry point for building tomocupy.
The setup process for ptychocg is very much like any python module except
that the compilation of the the extension module(s) is driven by CMake through
scikit-build. Scikit-build defines a custom Extension class which calls CMake
and provides some common (for Python) CMake package finders.
You can pass build options to CMake using the normal -DBUILD_OPTION=something
syntax, but these options must separated from the setuptools options with two
dashes (--). For example, we can pass the EXTENSION_WRAPPER option as follows:
$ python setup.py build -- -DEXTENSION_WRAPPER=swig
For skbuild >= 0.10.0, the two dashes will not be required. See the top-level
CMakeLists.txt for the curent list of build options.
"""
from skbuild import setup
from setuptools import find_packages

setup(
    name='tomocupy-cli',
    author='Viktor Nikitin',
    version='0.3.0',
    package_dir={"": "src"},
    scripts=['bin/tomocupycli.py'],  
    entry_points={'console_scripts':['tomocupy = tomocupycli:main'],},
    packages=find_packages('src'),
    zip_safe=False,
)