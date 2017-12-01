# SpectroscoPy
A Python package for storing and processing spectroscopy data from volcanic gas
monitoring. It is based on HDF5 using [pytables](http://www.pytables.org) and a
newly developed datamodel for volcanic gas chemistry data and analysis results.
For more details please visit the [website](http://yannikbehr.github.io/spectroscopy).

## Installation
### From source

Currently, this is the only available option. The following shows, as an
example, how to install the package and its dependencies under Ubuntu.

First install the dependencies::

    apt-get install python-pip git libgeos-dev libproj-dev proj-bin \
    python-scipy python-numpy python-dateutil python-pyproj python-matplotlib \
    python-tables python-nose python-pandas

Then clone the source code and install it::

    git clone --depth=1 https://github.com/yannikbehr/spectroscopy.git
    cd spectroscopy
    python setup.py install

And finally run the tests::

    python setup.py test
