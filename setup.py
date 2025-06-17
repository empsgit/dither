from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="dither_core",
    ext_modules=cythonize("dither_core.pyx"),
    include_dirs=[numpy.get_include()]
)
