Installation
============

Implicit can be installed from pypi with:

.. code-block:: bash

    pip install implicit

Installing with pip will use prebuilt binary wheels on x86_64 Linux, Windows
and OSX. The wheels includes GPU support on Linux.

There are also binary packages on conda-forge for Linux, Windows and OSX. These can be installed with:

.. code-block:: bash

    conda install -c conda-forge implicit

There are also GPU enabled packages on conda-forge for x86_64 Linux systems using either CUDA
11.0, 11.1 or 11.2. The GPU packages can be installed with:

.. code-block:: bash

    conda install -c conda-forge implicit implicit-proc=*=gpu


Requirements
------------

This library requires SciPy version 0.16 or later.
