# Adapted from https://github.com/rmcgibbo/npcuda-example
import logging
import os

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

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            logging.info('The nvcc binary could not be located in your $PATH. Either add it to '
                         'your path, or set $CUDAHOME to enable CUDA extensions')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home,
                  'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib64':   os.path.join(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            logging.warning('The CUDA %s path could not be located in %s', k, v)
            return None

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])

            # override compiler args for nvcc
            postargs = ['-gencode=arch=compute_30,code=sm_30',
                        '-gencode=arch=compute_35,code=sm_35',
                        '-gencode=arch=compute_50,code=sm_50',
                        '-gencode=arch=compute_52,code=sm_52',
                        '-gencode=arch=compute_52,code=compute_52',
                        '--ptxas-options=-v', '-O2',
                        '-c', '--compiler-options', "'-fPIC'"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class cuda_build_ext(setuptools_build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        setuptools_build_ext.build_extensions(self)


CUDA = locate_cuda()
build_ext = cuda_build_ext if CUDA else setuptools_build_ext
