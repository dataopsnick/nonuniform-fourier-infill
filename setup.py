import os
import sys

# Task 8: Check for build-time dependencies
try:
    from setuptools import setup, Extension
except ImportError:
    raise ImportError(
        "Setuptools is required to build this package. "
        "Install it via: pip install setuptools"
    )

try:
    from Cython.Build import cythonize
    from Cython import __version__ as cython_version
except ImportError:
    raise ImportError(
        "Cython is required to build this package. "
        "Install it via: pip install cython>=3.0.0"
    )
else:
    try:
        from packaging.version import Version
        if Version(cython_version) < Version("3.0.0"):
            raise ImportError(
                f"Cython >= 3.0.0 is required, but {cython_version} is installed. "
                "Upgrade via: pip install cython>=3.0.0"
            )
    except ImportError:
        cy_parts = [int(x) for x in cython_version.split(".") if x.isdigit()]
        if len(cy_parts) > 0 and cy_parts[0] < 3:
            raise ImportError(
                f"Cython >= 3.0.0 is required, but {cython_version} is installed. "
                "Upgrade via: pip install cython>=3.0.0"
            )

try:
    import numpy as np
    from numpy import __version__ as numpy_version
except ImportError:
    raise ImportError(
        "NumPy is required to build this package. "
        "Install it via: pip install numpy>=1.20.0"
    )
else:
    try:
        from packaging.version import Version
        if Version(numpy_version) < Version("1.20.0"):
            raise ImportError(
                f"NumPy >= 1.20.0 is required, but {numpy_version} is installed. "
                "Upgrade via: pip install numpy>=1.20.0"
            )
    except ImportError:
        np_parts = [int(x) for x in numpy_version.split(".") if x.isdigit()]
        if len(np_parts) >= 2 and (np_parts[0] < 1 or (np_parts[0] == 1 and np_parts[1] < 20)):
            raise ImportError(
                f"NumPy >= 1.20.0 is required, but {numpy_version} is installed. "
                "Upgrade via: pip install numpy>=1.20.0"
            )

# Determine compiler flags for OpenMP (multi-threading support)
ext_compiler_args = []
ext_linker_args = []
ext_include_dirs = [np.get_include()]

if "NUIFI_NO_OPENMP" in os.environ:
    ext_compiler_args = []
    ext_linker_args = []
else:
    if sys.platform == "win32":
        ext_compiler_args = ["/openmp"]
    elif sys.platform == "darwin":
        # On macOS, standard clang doesn't support -fopenmp without extra config.
        # Try to detect Homebrew libomp; fall back to no OpenMP with a warning.
        libomp_candidates = [
            "/opt/homebrew/opt/libomp",   # Apple Silicon Homebrew
            "/usr/local/opt/libomp",       # Intel Homebrew
            "/opt/local/lib/libomp",       # MacPorts
            sys.prefix,  # conda / virtualenv (include/ and lib/ live directly under prefix)
            "/usr/local",
            "/usr",
        ]
        # Also allow explicit override via environment variable
        env_libomp = os.environ.get("LIBOMP_ROOT") or os.environ.get("LIBOMP_PATH")
        if env_libomp:
            libomp_candidates.insert(0, env_libomp)
        libomp_path = None
        for candidate in libomp_candidates:
            if candidate and os.path.isdir(os.path.join(candidate, "include")):
                # Ensure we have omp.h to avoid matching sys.prefix false positives
                if os.path.exists(os.path.join(candidate, "include", "omp.h")):
                    libomp_path = candidate
                    break
        if libomp_path:
            ext_compiler_args = ["-Xpreprocessor", "-fopenmp"]
            ext_linker_args = ["-L" + os.path.join(libomp_path, "lib"), "-lomp"]
            ext_include_dirs.append(os.path.join(libomp_path, "include"))
        else:
            import warnings
            warnings.warn(
                "OpenMP not found. Install libomp via 'brew install libomp' "
                "or set the LIBOMP_ROOT environment variable to the libomp prefix. "
                "Conda users can install via 'conda install llvm-openmp'."
            )
            ext_compiler_args = []
            ext_linker_args = []
    else:
        # Linux, BSD, or other systems. Let's check compiler support for -fopenmp.
        import subprocess
        import tempfile
        import shutil

        has_openmp = False
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp()
            test_file = os.path.join(tmpdir, "test.c")
            with open(test_file, "w") as f:
                f.write("#include <omp.h>\nint main(void) { return omp_get_num_threads(); }\n")
            import sysconfig
            cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
            cmd = [cc, "-fopenmp", test_file, "-o", os.path.join(tmpdir, "test")]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                has_openmp = True
        except Exception:
            pass
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

        if has_openmp:
            ext_compiler_args = ["-fopenmp"]
            ext_linker_args = ["-fopenmp"]
        else:
            import warnings
            warnings.warn("OpenMP not supported by compiler, disabling.")
            ext_compiler_args = []
            ext_linker_args = []

extensions = [
    Extension(
        "nufi.kernels.cy_kernels",
        sources=["nufi/kernels/cy_kernels.pyx"],
        include_dirs=ext_include_dirs,
        extra_compile_args=ext_compiler_args,
        extra_link_args=ext_linker_args,
    )
]

setup(
    name="nufi",
    version="0.1.0",
    packages=["nufi", "nufi.kernels"],
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
)
