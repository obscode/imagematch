from distutils.core import setup
from glob import glob
import os
scripts = glob('bin/*')

setup(
      name='imagematch',
      version='0.1.0',
      author='Chris Burns',
      author_email='cburns@carnegiescience.edu',
      packages=['imagematch'],
      scripts=scripts,
      package_data={'imagematch':['data/*']},
      description='A package for rectifying and PSF-matching astronomical data',
      requires=[
         'astropy',
         'scipy',
         'numpy',
         'vtk',
      ],
      url='http://code.obs.carnegiescience.edu/',
      license='MIT',
      classifiers=[
         'Development Status :: 4 - Beta',
         'Environment :: Console',
         'Intended Audience :: Science/Research',
         'License :: OSI Approved :: MIT License',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3.7',
         'Topic :: Scientific/Engineering :: Astronomy'],
      )

