Installation
============

There are binary packages on conda-forge for Linux, Windows and OSX. These can be installed with:

.. code-block:: bash

    conda install -c conda-forge implicit

There are also GPU enabled packages on conda-forge for x86_64 Linux systems using either CUDA
11.0, 11.1 or 11.2. The GPU packages can be installed with:

.. code-block:: bash

    conda install -c conda-forge implicit implicit-proc=*=gpu


There is also an sdist package on PyPi. This package can be installed with:

.. code-block:: bash

    pip install implicit

Note that installing with pip requires a C++ compiler to be installed on your system, since this
method will build implicit from source.


Requirements
------------

This library requires SciPy version 0.16 or later. Running on OSX requires an OpenMP compiler,
which can be installed with homebrew: ``brew install gcc``.
