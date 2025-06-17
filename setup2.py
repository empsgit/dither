from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("error_dither_core", ["error_dither_core.pyx"],
              extra_compile_args=["-O3"])
]

setup(
    name="error_dither_core",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)