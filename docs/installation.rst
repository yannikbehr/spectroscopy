Installation
============

Currently the only available option is to install from source. The following
shows, as an example, how to install it under Ubuntu.

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
