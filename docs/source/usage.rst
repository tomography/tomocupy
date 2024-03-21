=====
Usage
=====

Examples
========


Reconstruct APS data 
--------------------

This section contains the readrec_aps script.

Download file: :download:`readrec_aps.py
<../../docs/demo/readrec_aps.py>`

.. literalinclude:: ../../docs/demo/readrec_aps.py
    :tab-width: 4
    :linenos:
    :language: guess

Command Line Interface
----------------------

**tomocupy** includes a commad-line-interface (CLI). The simplest way to set a reconstruction parameter is to directly
pass it as an option to the ``tomocupy`` command. Some options also accept an argument, while others simple enable certain
behavior. Parameters given directly via the command line will override those given via a parameter file or global configuration file.

To list all the options supported by the tomocupy CLI, after installing tomocupy, type::

    (tomocupy)$ tomocupy -h
    (tomocupy)$ tomocupy recon -h
    (tomocupy)$ tomocupy recon_steps -h

Below are different **reconstruction** examples

Try center
~~~~~~~~~~
::

   (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --reconstruction-type try --center-search-width 100

Full volume
~~~~~~~~~~~
::

   (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full

Double FOV
~~~~~~~~~~
::

    (tomocupy)$ (tomocupy)$ tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full --file-type double_fov

Full volume rec with phase retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 4 --rotation-axis 700 --reconstruction-type full --energy 20 --pixel-size 1.75 --propagation-distance 100 --retrieve-phase-alpha 0.001 --retrieve-phase-method paganin --reconstruction-type full 

Laminographic try
~~~~~~~~~~~~~~~~~
::

    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8 --reconstruction-type try --center-search-width 100 --lamino-angle 20

Laminographic try angle
~~~~~~~~~~~~~~~~~~~~~~~
::

    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8 --rotation-axis 700 --reconstruction-type try_lamino --lamino-search-width 2 --lamino-angle 20

Laminographic full rec
~~~~~~~~~~~~~~~~~~~~~~
::
    
    (tomocupy)$ tomocupy recon_steps --file-name data/test_data.h5 --nsino-per-chunk 8 --nproj-per-chunk 8--reconstruction-type full --rotation-axis 700 --lamino-angle 20
