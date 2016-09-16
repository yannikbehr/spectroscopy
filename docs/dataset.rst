**************************
Datamodel for spectroscopy
**************************

The datamodel is intended to store raw spectroscopy data, analysis results, and
the corresponding processing flows. The general idea is that a file contains
all the information required to reproduce the final flux measurements. It was
designed with the HDF5 format in mind for I/O but plugins can also be written
for storing datasets in a database or XML files.

Example
=======

The following shows an example on how the ``Dataset`` type, which implements
the datamodel, can be used to get a quick overview of the data contained in a
FlySpec file::

	from spectroscopy.dataset import Dataset
	d = Dataset.open('testfile', format='FLYSPEC')
	d.plot()

.. image:: _images/chile_retrievals_overview.png
   :align: center
   :scale: 40 %

UML
===
.. image:: _images/datamodel.svg
   
   
Classes & Functions
===================

.. automodule:: spectroscopy.dataset
   :members:
   