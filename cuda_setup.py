# Adapted from https://github.com/rmcgibbo/npcuda-example and
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
import logging
import os
import sys

from distutils import ccompiler, errors, msvccompiler, unixccompiler
from setuptools.command.build_ext import build_ext as setuptools_build_ext


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    If a valid cuda installation is found this returns a dict with keys 'home', 'nvcc', 'include',
    and 'lib64' and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything is based on finding
    'nvcc' in the PATH.

    If nvcc can't be found, this returns None
    """
    nvcc_bin = 'nvcc'
    if sys.platform.startswith("win"):
        nvcc_bin = 'nvcc.exe'

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', nvcc_bin)
    elif 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        nvcc = os.path.join(home, 'bin', nvcc_bin)
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path(nvcc_bin, os.environ['PATH'])
        if nvcc is None:
            logging.warning('The nvcc binary could not be located in your $PATH. Either add it to '
                            'your path, or set $CUDAHOME to enable CUDA extensions')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home,
                  'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64':   os.path.join(home, 'lib64')}
    post_args = ['-gencode=arch=compute_30,code=sm_30',
                 '-gencode=arch=compute_50,code=sm_50',
                 '-gencode=arch=compute_60,code=sm_60',
                 '-gencode=arch=compute_60,code=compute_60',
                 '--ptxas-options=-v', '-O2']

    if sys.platform == "win32":
        cudaconfig['lib64'] = os.path.join(home, 'lib', 'x64')
        post_args += ['-Xcompiler', '/MD']
    else:
        post_args += ['-c', '--compiler-options', "'-fPIC'"]

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            logging.warning('The CUDA %s path could not be located in %s', k, v)
            return None

    cudaconfig['post_args'] = post_args
    return cudaconfig


# This code to build .cu extensions with nvcc is taken from cupy:
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
class _UnixCCompiler(unixccompiler.UnixCCompiler):
    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.append('.cu')

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        # For sources other than CUDA C ones, just call the super class method.
        if os.path.splitext(src)[1] != '.cu':
            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        # For CUDA C source files, compile them with NVCC.
        _compiler_so = self.compiler_so
        try:
            nvcc_path = CUDA['nvcc']
            post_args = CUDA['post_args']
            # TODO? base_opts = build.get_compiler_base_options()
            self.set_executable('compiler_so', nvcc_path)

            return unixccompiler.UnixCCompiler._compile(
                self, obj, src, ext, cc_args, post_args, pp_opts)
        finally:
            self.compiler_so = _compiler_so


class _MSVCCompiler(msvccompiler.MSVCCompiler):
    _cu_extensions = ['.cu']

    src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
    src_extensions.extend(_cu_extensions)

    def _compile_cu(self, sources, output_dir=None, macros=None,
                    include_dirs=None, debug=0, extra_preargs=None,
                    extra_postargs=None, depends=None):
        # Compile CUDA C files, mainly derived from UnixCCompiler._compile().
        macros, objects, extra_postargs, pp_opts, _build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)

        compiler_so = CUDA['nvcc']
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        post_args = CUDA['post_args']

        for obj in objects:
            try:
                src, _ = _build[obj]
            except KeyError:
                continue
            try:
                self.spawn([compiler_so] + cc_args + [src, '-o', obj] + post_args)
            except errors.DistutilsExecError as e:
                raise errors.CompileError(str(e))

        return objects

    def compile(self, sources, **kwargs):
        # Split CUDA C sources and others.
        cu_sources = []
        other_sources = []
        for source in sources:
            if os.path.splitext(source)[1] == '.cu':
                cu_sources.append(source)
            else:
                other_sources.append(source)

        # Compile source files other than CUDA C ones.
        other_objects = msvccompiler.MSVCCompiler.compile(
            self, other_sources, **kwargs)

        # Compile CUDA C sources.
        cu_objects = self._compile_cu(cu_sources, **kwargs)

        # Return compiled object filenames.
        return other_objects + cu_objects


class cuda_build_ext(setuptools_build_ext):
    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if CUDA is not None:
            def wrap_new_compiler(func):
                def _wrap_new_compiler(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except errors.DistutilsPlatformError:
                        if not sys.platform == 'win32':
                            CCompiler = _UnixCCompiler
                        else:
                            CCompiler = _MSVCCompiler
                        return CCompiler(
                            None, kwargs['dry_run'], kwargs['force'])
                return _wrap_new_compiler
            ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
            # Intentionally causes DistutilsPlatformError in
            # ccompiler.new_compiler() function to hook.
            self.compiler = 'nvidia'

        setuptools_build_ext.run(self)


CUDA = locate_cuda()
build_ext = cuda_build_ext if CUDA else setuptools_build_ext
