.. _installation:
.. index:: Installation

Installation
************


Installing via pip
====================

In the most simple case, ACAT can be simply installed
via pip::

    $ pip3 install acat


Cloning the repository
======================

If you want to get the absolutely latest version you can clone the
repo::

    $ git clone https://gitlab.com/asm-dtu/acat.git

and then install ACAT via::

    $ cd acat/
    $ python3 setup.py install

in the root directory. This will set up ACAT as a Python module
for the current user.


Requirements
============

ACAT requires Python3.6+ and depends on the following libraries

* `ASE <https://wiki.fysik.dtu.dk/ase>`_ 
* `ASAP <https://wiki.fysik.dtu.dk/asap>`_ 
* `NetworkX <https://networkx.org>`_

You can simply install these libraries via pip::

    $ pip3 install ase asap3 networkx
