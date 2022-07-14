from skbuild import setup
from setuptools import find_packages

setup(
    name='tomocupy',
    version=open('VERSION').read().strip(),
    author='Viktor Nikitin',
    package_dir={"": "src"},
    scripts=['bin/tomocupy_cli.py'],  
    entry_points={'console_scripts':['tomocupy = tomocupy_cli:main'],},
    packages=find_packages('src'),
    zip_safe=False,
)
