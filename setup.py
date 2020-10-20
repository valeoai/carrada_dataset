from setuptools import setup
from setuptools import find_packages


setup(name='carrada',
      version='0.1.0',
      license='GPL-3.0',
      install_requires=['filterpy', 'matplotlib', 'numba', 'numpy',
                        'pandas', 'Pillow', 'pytest', 'scikit-image',
                        'scikit-learn', 'scipy', 'torchvision', 'xmltodict'],
      packages=find_packages())
