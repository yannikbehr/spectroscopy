
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
    author="Nial Peters",
    author_email="nonbiostudent@hotmail.com",
    url="",
    license="GPL v3",
    package_dir={'': 'src'},
    install_requires=['pyinotify'],
    packages=['spectroscopy', 'spectroscopy.flux', 'spectroscopy.doas',
                'spectroscopy.plugins'],
    package_data={"spectroscopy.flux": ["flyspec_config.cfg"]},
)
