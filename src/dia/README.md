# Dia plugin

This directory contains the datamodel as a [dia](http://dia-installer.de/) file together with
a plugin that produces the corresponding classes in Python.

To use the plugin, copy or link `dia_renderer.py` to the directory `${HOME}/.dia/python` before starting `dia`.
In `dia` open `datamodel.dia`. You can then export the datamodel from dia using the 
`Gas chemistry code generation (Python)` format.
