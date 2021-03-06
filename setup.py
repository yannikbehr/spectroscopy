
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension
from setuptools.command.install import install

####################################################################
#                    CONFIGURATION
####################################################################

# do the build/install
setup(
    name="spectroscopy",
    version="0.1",
    description="Python package for processing field spectroscopy data.",
    long_description="Python package for processing field spectroscopy data.",
    author="Yannik Behr and Nial Peters",
    author_email="y.behr@gns.cri.nz",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
    install_requires=['tables', 'numpy', 'matplotlib', 'scipy',
                      'python-dateutil', 'cartopy', 'pyproj'],
    packages=['spectroscopy', 'spectroscopy.flux', 'spectroscopy.doas',
                'spectroscopy.plugins'],
    test_suite='nose.collector',
    test_require=['nose']
)
