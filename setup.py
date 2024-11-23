from setuptools import setup, find_packages

setup(
  name = 'spmsimu',
  version = '0.0.1',
  packages = find_packages(),
  license='MIT',
  description = 'SPM Simulator.',
  long_description = 'Python simulator for generating realistic scan images.\nIt offers simulated scan images based on ground-truth patterns or user-input images taken into account of most scanning parameters.',
  author = ["Richard (Yu) Liu"],
  email = ['yu93liu@gmail.com'],
  url = 'https://github.com/RichardLiuCoding/spmsimu',
  download_url = 'https://github.com/RichardLiuCoding/spmsimu.git',
  keywords = ['SPM', 'Python', 'Simulator', 'Machine learning', 'User training'],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'numba',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
