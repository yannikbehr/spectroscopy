**************************
Datamodel for spectroscopy
**************************

The datamodel is intended to store raw spectroscopy data, analysis results, and
the corresponding processing flows. The general idea is that a file contains
all the information required to reproduce the final flux measurements. This
implementation of the datamodel is based on HDF5 but other implementations may,
for example, use the datamodel as a database or XML scheme.

Example
=======

The following shows an example on how the ``Dataset`` type, which implements
the datamodel, can be used to get a quick overview of the data contained in a
FlySpec file::

	from spectroscopy.dataset import Dataset
        import tempfile
        d = Dataset(tempfile.mktemp(), 'w')
	d.read('testfile', ftype='FLYSPEC')
	d.plot()

.. image:: _images/chile_retrievals_overview.png
   :align: center
   :scale: 40 %

Simplified datamodel
====================
.. image:: _images/datamodel_simple.svg
   
   
Classes & Functions
===================

.. automodule:: spectroscopy.dataset
   :members:
.. automodule:: spectroscopy.datamodel
   :members:

   
