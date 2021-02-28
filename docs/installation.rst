.. _installation:
.. index:: Installation

Installation
************


Installing via `pip`
====================

In the most simple case, :program:`ACAT` can be simply installed
via `pip`::

    pip install acat --user


Cloning the repository
======================

If you want to get the absolutely latest version you can clone the
repo::

    git clone https://gitlab.com/shuanghan/acat.git

and then install :program:`ACAT` via::

    cd acat
    python3 setup.py install --user

in the root directory. This will set up :program:`ACAT` as a Python module
for the current user.


Requirements
============

:program:`ACAT` requires Python3.6+ and depends on the following libraries

* `ASE <https://wiki.fysik.dtu.dk/ase>`_ 
* `ASAP <https://wiki.fysik.dtu.dk/asap>`_ 
* `NetworkX <https://networkx.org>`_

You can simply install these libraries via::

    pip install ase asap3 networkx --user
