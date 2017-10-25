from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

setup(
  name = 'Test app',
  cmdclass = {'build_ext': build_ext},
  ext_modules=[
    	Extension('RegTreeError',
              sources=["RegTreeError.pyx"],
	      libraries=["m"],
	      include_dirs=[numpy.get_include()],
#              extra_compile_args=['-O3',"-march=native", "-fopenmp"],
#	      extra_link_args=["-fopenmp"]
		),
	Extension('RegTree',
	      sources=["RegTree.pyx"],
	      libraries=["m"],
	      include_dirs=[numpy.get_include()],
#	      extra_compile_args=['-O3',"-march=native", "-fopenmp"],
#	      extra_link_args=["-fopenmp"]
		),
	Extension('GBRegTree',
              sources=["GBRegTree.pyx"],
	      include_dirs=[numpy.get_include()],
#              extra_compile_args=['-O3',"-march=native","-fopenmp" ],
#	      extra_link_args=["-fopenmp"]
		),
],
#              language='c++'
  
)

