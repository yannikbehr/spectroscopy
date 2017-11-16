**************************
Datamodel for spectroscopy
**************************

The datamodel is intended to store raw spectroscopy data, analysis results, and
the corresponding processing flows. The general idea is that a file contains
all the information required to reproduce the final flux measurements. This
implementation of the datamodel is based on HDF5 but other implementations may,
for example, use the datamodel as a database or XML scheme.

Simplified datamodel
====================
.. image:: _images/datamodel_simple.svg
   
   
Classes & Functions
===================

.. automodule:: spectroscopy.dataset
   :members:
.. automodule:: spectroscopy.datamodel
   :members:

   
