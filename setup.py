import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Determine compiler flags for OpenMP (multi-threading support)
ext_compiler_args = []
ext_linker_args = []

if sys.platform == "win32":
    ext_compiler_args = ["/openmp"]
elif sys.platform == "darwin":
    # On macOS, standard clang doesn't support -fopenmp without extra config.
    # We will try compiling without OpenMP, or user can set brew paths.
    ext_compiler_args = []
else:
    # Linux (GCC)
    ext_compiler_args = ["-fopenmp"]
    ext_linker_args = ["-fopenmp"]

extensions = [
    Extension(
        "nufi.kernels.cy_kernels",
        sources=["nufi/kernels/cy_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=ext_compiler_args,
        extra_link_args=ext_linker_args,
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
